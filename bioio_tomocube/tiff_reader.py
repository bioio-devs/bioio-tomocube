#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reader for TIFF volumes exported by the Tomocube HTX processing server.

The HTX ``ProcessingServer`` exports one ImageJ-style multi-page TIFF per
modality and channel.  For a single acquisition (one timepoint) the files are
laid out as::

    <base>.TP01_HT3D_0.00.TIFF                 # refractive-index volume
    <base>.TP01/
        <base>.TP01_FL3D_CH0_0.00.TIFF         # fluorescence volume, channel 0
        <base>.TP01_FL3D_CH1_0.00.TIFF
        <base>.TP01_FL2D_CH0_0.00.TIFF         # fluorescence max projection
        ...

Pointing :class:`TiffReader` at *any* of these files discovers the sibling
files of the same acquisition and exposes each modality/channel as a scene
using the same names as the TCF reader (``"3D"``, ``"3DFL/CH0"``, …).

Known quirks of the export that this reader compensates for:

* ``HT3D`` stores refractive index multiplied by 10 000 as ``uint16``.
* ``FL3D`` volumes are resampled onto the HT Z grid (same page count as the
  ``HT3D`` file) but the ImageJ header still reports the *original* FL slice
  count and spacing.  ``tifffile`` trusts that header and reports the wrong
  shape, so this reader always enumerates TIFF pages directly and, when an
  ``HT3D`` sibling with the same page count exists, uses the HT Z spacing.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from bioio_base import constants, exceptions, io, types
from bioio_base.dimensions import DimensionNames
from bioio_base.standard_metadata import StandardMetadata
from fsspec.spec import AbstractFileSystem
from ome_types.model import OME

from bioio_tomocube.ome_utils import TiffOmeScene, build_tiff_ome
from bioio_tomocube.reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

TIFF_EXTENSIONS = (".tiff", ".tif")

#: ``HT3D``/``HT2D`` store refractive index * 10 000 as uint16.
RI_SCALE = 1.0e-4

_MODALITY_TO_SCENE = {
    "HT3D": "3D",
    "HT2D": "2DMIP",
    "FL3D": "3DFL",
    "FL2D": "2DFLMIP",
}
_HT_MODALITIES = frozenset({"HT3D", "HT2D"})
_VOLUME_MODALITIES = frozenset({"HT3D", "FL3D"})
_SCENE_ORDER = ("3D", "2DMIP", "3DFL", "2DFLMIP")

# <base>[.TPnn]_<MODALITY>[_CHn]_<suffix>.TIFF
_FILENAME_RE = re.compile(
    r"^(?P<stem>(?P<base>.+?)(?:\.TP(?P<tp>\d+))?)"
    r"_(?P<modality>HT3D|HT2D|FL3D|FL2D)"
    r"(?:_CH(?P<channel>\d+))?"
    r"_(?P<suffix>(?!CH\d+\.)[^_/]+?)"  # a channel token is never the suffix
    r"\.(?P<ext>tiff?)$",
    re.IGNORECASE,
)
_TP_DIR_RE_TEMPLATE = r"^{base}\.TP\d+$"
_WELL_RE = re.compile(r"^(?P<row>[A-Za-z]{1,2})(?P<column>\d{1,3})$")

_TIFF_DATETIME_FMT = "%Y:%m:%d %H:%M:%S"
_RESUNIT_TO_UM = {2: 25_400.0, 3: 10_000.0}  # inch, centimeter
_IMAGEJ_UNIT_TO_UM = {
    "cm": 10_000.0,
    "mm": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "nm": 1.0e-3,
    "inch": 25_400.0,
}
_TAGS_TO_SKIP = frozenset({"StripOffsets", "StripByteCounts", "TileOffsets"})

_SPATIAL_DIMS = (
    DimensionNames.SpatialZ,
    DimensionNames.SpatialY,
    DimensionNames.SpatialX,
)

# Sentinel distinguishing "not yet cached" from None (a valid return value)
_NOT_CACHED: Any = object()


###############################################################################
# File-name parsing and acquisition discovery
###############################################################################


class TomocubeFile(NamedTuple):
    """One exported TIFF file, as parsed from its file name."""

    path: str
    stem: str  # ``<base>.TPnn`` (or ``<base>`` when there is no TP part)
    base: str  # acquisition name shared by all timepoints
    timepoint: Optional[int]
    modality: str  # ``HT3D``, ``HT2D``, ``FL3D`` or ``FL2D``
    channel: Optional[int]
    suffix: str

    @property
    def scene(self) -> str:
        """Scene name in TCF-reader nomenclature (e.g. ``"3DFL/CH0"``)."""
        scene = _MODALITY_TO_SCENE[self.modality]
        if self.channel is not None:
            return f"{scene}/CH{self.channel}"
        return scene

    @property
    def is_ht(self) -> bool:
        return self.modality in _HT_MODALITIES


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _dirname(path: str) -> str:
    head, sep, _ = path.rstrip("/").rpartition("/")
    return head if sep else ""


def parse_filename(path: str) -> Optional[TomocubeFile]:
    """Parse a Tomocube export file name; return ``None`` if it does not match."""
    match = _FILENAME_RE.match(_basename(path))
    if match is None:
        return None
    tp = match.group("tp")
    ch = match.group("channel")
    return TomocubeFile(
        path=path,
        stem=match.group("stem"),
        base=match.group("base"),
        timepoint=int(tp) if tp is not None else None,
        modality=match.group("modality").upper(),
        channel=int(ch) if ch is not None else None,
        suffix=match.group("suffix"),
    )


def _scene_sort_key(scene: str) -> Tuple[int, int]:
    prefix, _, ch = scene.partition("/CH")
    order = _SCENE_ORDER.index(prefix) if prefix in _SCENE_ORDER else len(_SCENE_ORDER)
    return order, int(ch) if ch else -1


def _list_dir(fs: AbstractFileSystem, directory: str) -> Tuple[List[str], List[str]]:
    """Return ``(files, sub_directories)`` for *directory*; empty lists on error."""
    files: List[str] = []
    dirs: List[str] = []
    if not directory:
        directory = "/"
    try:
        entries = fs.ls(directory, detail=True)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as err:
        log.debug("Could not list %r: %s", directory, err)
        return files, dirs
    for entry in entries:
        if isinstance(entry, dict):
            name = str(entry.get("name", ""))
            is_dir = entry.get("type") == "directory"
        else:
            name = str(entry)
            is_dir = False
        if not name:
            continue
        (dirs if is_dir else files).append(name.rstrip("/"))
    return files, dirs


def discover_acquisition(
    fs: AbstractFileSystem, path: str, timelapse: bool = False
) -> Dict[str, List[TomocubeFile]]:
    """Find every exported file belonging to the acquisition *path* is part of.

    Parameters
    ----------
    fs, path:
        Filesystem and path of any one exported TIFF (the *anchor*).
    timelapse:
        When ``False`` only files of the anchor's timepoint are returned.
        When ``True`` all timepoints sharing the anchor's ``<base>`` are
        collected, sorted by timepoint, so each scene gains a T dimension.

    Returns
    -------
    Dict[str, List[TomocubeFile]]
        Mapping of scene name to the files that make up that scene, ordered
        by timepoint.
    """
    anchor = parse_filename(path)
    if anchor is None:
        raise exceptions.UnsupportedFileFormatError(
            "bioio-tomocube",
            path,
            "File name does not follow the Tomocube export pattern "
            "<name>[.TPnn]_<HT3D|HT2D|FL3D|FL2D>[_CHn]_<suffix>.TIFF",
        )

    # The HT3D file sits next to a folder named ``<base>.TPnn`` holding the FL
    # files.  If the anchor is one of those FL files, step up one level.
    parent = _dirname(path)
    root = _dirname(parent) if _basename(parent) == anchor.stem else parent

    files, dirs = _list_dir(fs, root)
    candidates = list(files)
    tp_dir_re = re.compile(_TP_DIR_RE_TEMPLATE.format(base=re.escape(anchor.base)))
    for directory in dirs:
        name = _basename(directory)
        if timelapse:
            wanted = name == anchor.base or tp_dir_re.match(name) is not None
        else:
            wanted = name == anchor.stem
        if wanted:
            candidates.extend(_list_dir(fs, directory)[0])

    scenes: Dict[str, List[TomocubeFile]] = {}
    seen: set = set()
    for candidate in candidates:
        info = parse_filename(candidate)
        if info is None or info.base != anchor.base:
            continue
        if not timelapse and info.timepoint != anchor.timepoint:
            continue
        key = (info.scene, info.timepoint, _basename(candidate))
        if key in seen:
            continue
        seen.add(key)
        scenes.setdefault(info.scene, []).append(info)

    # Listing may be unavailable on some filesystems; always include the anchor.
    anchor_key = (anchor.scene, anchor.timepoint, _basename(path))
    if anchor_key not in seen:
        scenes.setdefault(anchor.scene, []).append(anchor)

    for members in scenes.values():
        members.sort(
            key=lambda f: (f.timepoint is not None, f.timepoint or 0, f.path),
        )
    return dict(sorted(scenes.items(), key=lambda kv: _scene_sort_key(kv[0])))


###############################################################################
# TIFF header parsing
###############################################################################


class _PlaneLocation(NamedTuple):
    """Where one Z plane lives inside a TIFF file."""

    page_index: int
    offset: int  # byte offset of the raw pixel data; ``-1`` → use tifffile
    nbytes: int


class _FileHeader(NamedTuple):
    """Everything needed from one TIFF file, read once."""

    file: TomocubeFile
    n_pages: int
    size_y: int
    size_x: int
    dtype: np.dtype  # as stored (may be non-native byte order)
    planes: Tuple[_PlaneLocation, ...]
    datetime: Optional[datetime]
    pixel_size_x: Optional[float]
    pixel_size_y: Optional[float]
    z_spacing: Optional[float]  # from the ImageJ header, µm
    tags: Dict[str, Any]
    imagej: Dict[str, Any]


def _plain(value: Any) -> Any:
    """Convert a tifffile tag value into a JSON-friendly Python object."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "name") and hasattr(value, "value"):  # IntEnum
        return value.name
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    return value


def _resolution_to_um(resolution: Any, unit: Any) -> Optional[float]:
    """Convert a TIFF ``XResolution``/``YResolution`` rational into µm/pixel."""
    try:
        scale = _RESUNIT_TO_UM[int(unit)]
        numerator, denominator = resolution
        numerator = float(numerator)
        denominator = float(denominator)
    except (KeyError, TypeError, ValueError):
        return None
    if numerator == 0:
        return None
    return scale * denominator / numerator


def _imagej_spacing_um(imagej: Dict[str, Any]) -> Optional[float]:
    """Return the ImageJ ``spacing`` (Z step) converted to µm, if present."""
    spacing = imagej.get("spacing")
    if spacing is None:
        return None
    unit = str(imagej.get("unit", "")).strip().lower()
    factor = _IMAGEJ_UNIT_TO_UM.get(unit)
    if factor is None:
        log.warning("Unknown ImageJ unit %r; cannot convert Z spacing.", unit)
        return None
    return float(spacing) * factor


def _parse_tiff_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), _TIFF_DATETIME_FMT)
    except ValueError:
        return None


def _read_header(fs: AbstractFileSystem, file: TomocubeFile) -> _FileHeader:
    """Open one TIFF, enumerate its pages and collect the tags we care about."""
    with fs.open(file.path, "rb") as fobj:
        with tifffile.TiffFile(fobj) as tif:
            pages = list(tif.pages)
            first = pages[0]
            if first.ndim != 2 or first.samplesperpixel != 1:
                raise exceptions.UnsupportedFileFormatError(
                    "bioio-tomocube",
                    file.path,
                    "Only single-sample 2-D pages are supported "
                    f"(got shape {first.shape}).",
                )
            size_y, size_x = int(first.shape[0]), int(first.shape[1])
            stored_dtype = np.dtype(first.dtype).newbyteorder(tif.byteorder)
            expected_nbytes = size_y * size_x * stored_dtype.itemsize

            planes: List[_PlaneLocation] = []
            for index, page in enumerate(pages):
                if tuple(page.shape) != (size_y, size_x) or page.dtype != first.dtype:
                    raise exceptions.UnsupportedFileFormatError(
                        "bioio-tomocube",
                        file.path,
                        f"Page {index} has shape {page.shape}/{page.dtype}, "
                        f"expected {(size_y, size_x)}/{first.dtype}.",
                    )
                contiguous = (
                    int(page.compression) == 1
                    and int(page.predictor) == 1
                    and int(page.fillorder) == 1
                    and len(page.dataoffsets) == 1
                    and int(page.databytecounts[0]) == expected_nbytes
                )
                if contiguous:
                    planes.append(
                        _PlaneLocation(
                            index, int(page.dataoffsets[0]), expected_nbytes
                        )
                    )
                else:
                    planes.append(_PlaneLocation(index, -1, 0))

            tags = {
                tag.name: _plain(tag.value)
                for tag in first.tags.values()
                if tag.name not in _TAGS_TO_SKIP
            }
            imagej = dict(tif.imagej_metadata or {})

    unit = tags.get("ResolutionUnit")
    unit_code = {"INCH": 2, "CENTIMETER": 3}.get(str(unit), unit)
    return _FileHeader(
        file=file,
        n_pages=len(planes),
        size_y=size_y,
        size_x=size_x,
        dtype=stored_dtype,
        planes=tuple(planes),
        datetime=_parse_tiff_datetime(tags.get("DateTime")),
        pixel_size_x=_resolution_to_um(tags.get("XResolution"), unit_code),
        pixel_size_y=_resolution_to_um(tags.get("YResolution"), unit_code),
        z_spacing=_imagej_spacing_um(imagej),
        tags=tags,
        imagej=imagej,
    )


###############################################################################
# Scene assembly
###############################################################################


class _SceneInfo(NamedTuple):
    scene: str
    headers: Tuple[_FileHeader, ...]  # one per timepoint
    size_z: Optional[int]  # ``None`` for 2-D scenes (dims ``TYX``)
    size_y: int
    size_x: int
    dtype: np.dtype  # dtype of the *returned* array
    scale: Optional[float]  # multiply raw values by this (RI conversion)
    pixel_sizes: types.PhysicalPixelSizes
    z_spacing_source: Optional[str]
    time_interval: Optional[timedelta]


def _mean_interval(datetimes: Tuple[Optional[datetime], ...]) -> Optional[timedelta]:
    if len(datetimes) < 2 or any(dt is None for dt in datetimes):
        return None
    first, last = datetimes[0], datetimes[-1]
    assert first is not None and last is not None
    return (last - first) / (len(datetimes) - 1)


def _build_scene_info(
    scene: str,
    headers: Tuple[_FileHeader, ...],
    all_headers: Dict[str, Tuple[_FileHeader, ...]],
    refractive_index: bool,
) -> _SceneInfo:
    first = headers[0]
    for header in headers[1:]:
        same = (
            header.n_pages == first.n_pages
            and header.size_y == first.size_y
            and header.size_x == first.size_x
            and header.dtype == first.dtype
        )
        if not same:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube",
                header.file.path,
                f"Timepoints of scene {scene!r} differ in shape or dtype "
                f"({header.n_pages}x{header.size_y}x{header.size_x} "
                f"{header.dtype} vs {first.n_pages}x{first.size_y}x"
                f"{first.size_x} {first.dtype}).",
            )

    modality = first.file.modality
    is_volume = modality in _VOLUME_MODALITIES or first.n_pages > 1
    size_z: Optional[int] = first.n_pages if is_volume else None

    scale: Optional[float] = None
    out_dtype = first.dtype.newbyteorder("=")
    if refractive_index and first.file.is_ht:
        scale = RI_SCALE
        out_dtype = np.dtype(np.float32)

    # Z spacing: FL3D headers carry the *pre-resampling* FL spacing even though
    # the pages are on the HT grid.  Prefer the HT3D sibling's spacing when the
    # page counts agree.
    z_spacing = first.z_spacing if is_volume else None
    z_source: Optional[str] = "ImageJ header" if z_spacing is not None else None
    if is_volume and not first.file.is_ht:
        ht_headers = all_headers.get("3D")
        if ht_headers:
            ht_by_tp = {h.file.timepoint: h for h in ht_headers}
            ht = ht_by_tp.get(first.file.timepoint, ht_headers[0])
            if ht.n_pages == first.n_pages and ht.z_spacing is not None:
                if z_spacing is not None and not np.isclose(z_spacing, ht.z_spacing):
                    log.info(
                        "Scene %s: ImageJ header Z spacing %.4f µm differs from "
                        "the HT3D grid (%.4f µm) the pages are stored on; "
                        "using the HT3D spacing.",
                        scene,
                        z_spacing,
                        ht.z_spacing,
                    )
                z_spacing = ht.z_spacing
                z_source = "HT3D sibling"

    return _SceneInfo(
        scene=scene,
        headers=headers,
        size_z=size_z,
        size_y=first.size_y,
        size_x=first.size_x,
        dtype=out_dtype,
        scale=scale,
        pixel_sizes=types.PhysicalPixelSizes(
            Z=z_spacing, Y=first.pixel_size_y, X=first.pixel_size_x
        ),
        z_spacing_source=z_source,
        time_interval=_mean_interval(tuple(h.datetime for h in headers)),
    )


def _parse_acquisition_name(base: str) -> Dict[str, Any]:
    """Best-effort split of ``YYMMDD.HHMMSS.<experiment>.<index>.<group>.<well>``."""
    parts = base.split(".")
    result: Dict[str, Any] = {"base": base}
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        try:
            result["started"] = datetime.strptime(
                f"{parts[0]}.{parts[1]}", "%y%m%d.%H%M%S"
            )
        except ValueError:
            pass
        parts = parts[2:]
    if parts:
        well = _WELL_RE.match(parts[-1])
        if well is not None:
            result["well"] = parts[-1]
            result["row"] = well.group("row").upper()
            result["column"] = int(well.group("column"))
            parts = parts[:-1]
    if len(parts) >= 3 and parts[-2].isdigit():
        result["group"] = parts[-1]
        result["index"] = int(parts[-2])
        parts = parts[:-2]
    if parts:
        result["experiment"] = ".".join(parts)
    return result


###############################################################################
# Pixel readers (module-level so they pickle cleanly into dask graphs)
###############################################################################


def _to_output(arr: np.ndarray, scale: Optional[float]) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("="))
    if scale is not None:
        return arr.astype(np.float32) * np.float32(scale)
    return arr


def _read_plane(
    fs: AbstractFileSystem,
    path: str,
    location: _PlaneLocation,
    shape: Tuple[int, int],
    dtype: str,
    scale: Optional[float],
) -> np.ndarray:
    """Read one Z plane; opens and closes the file (safe on dask workers)."""
    with fs.open(path, "rb") as fobj:
        if location.offset >= 0:
            fobj.seek(location.offset)
            buffer = fobj.read(location.nbytes)
            arr = np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(shape)
        else:
            with tifffile.TiffFile(fobj) as tif:
                arr = tif.pages[location.page_index].asarray()
    return _to_output(arr, scale)


def _read_volume(
    fs: AbstractFileSystem, header: _FileHeader, scale: Optional[float]
) -> np.ndarray:
    """Read every page of one file with a single open; returns ``ZYX``."""
    shape = (header.size_y, header.size_x)
    planes: List[np.ndarray] = []
    with fs.open(header.file.path, "rb") as fobj:
        tif: Optional[tifffile.TiffFile] = None
        try:
            for location in header.planes:
                if location.offset >= 0:
                    fobj.seek(location.offset)
                    buffer = fobj.read(location.nbytes)
                    planes.append(
                        np.frombuffer(buffer, dtype=header.dtype).reshape(shape)
                    )
                else:
                    if tif is None:
                        tif = tifffile.TiffFile(fobj)
                    planes.append(tif.pages[location.page_index].asarray())
        finally:
            if tif is not None:
                tif.close()
    return _to_output(np.stack(planes, axis=0), scale)


###############################################################################


class TiffReader(Reader):
    """Read Tomocube HTX TIFF exports.

    Parameters
    ----------
    image : Path or str
        Path to any exported ``.TIFF`` of an acquisition (``_HT3D_``,
        ``_FL3D_CHn_`` or ``_FL2D_CHn_``).  Sibling files of the same
        acquisition are discovered automatically.  Any fsspec-compatible URI
        is accepted.
    fs_kwargs : Dict[str, Any]
        Keyword arguments forwarded to the fsspec filesystem.  Default: ``{}``.
    timelapse : bool
        When ``True``, every ``<base>.TPnn`` timepoint found next to *image*
        is stacked along ``T``.  Default: ``False`` (single timepoint, T=1).
    refractive_index : bool
        When ``True`` the ``3D``/``2DMIP`` scenes are returned as ``float32``
        refractive index (raw ``uint16`` × ``1e-4``).  Default: ``False``
        (raw ``uint16`` as stored).

    Notes
    -----
    Scenes mirror the TCF reader:

    * ``"3D"`` — refractive-index volume (``TZYX``)
    * ``"2DMIP"`` — refractive-index max projection (``TYX``), if exported
    * ``"3DFL/CH0"``, ``"3DFL/CH1"``, … — fluorescence volumes (``TZYX``)
    * ``"2DFLMIP/CH0"``, … — fluorescence max projections (``TYX``)

    The default scene is the one matching the file passed as *image*.
    """

    _physical_pixel_sizes: Optional[types.PhysicalPixelSizes]

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        if not path.lower().endswith(TIFF_EXTENSIONS):
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube", path, "File does not have a .TIFF/.TIF extension."
            )
        if parse_filename(path) is None:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube",
                path,
                "File name does not follow the Tomocube export pattern "
                "<name>[.TPnn]_<HT3D|HT2D|FL3D|FL2D>[_CHn]_<suffix>.TIFF",
            )
        try:
            with fs.open(path, "rb") as fobj:
                with tifffile.TiffFile(fobj) as tif:
                    tags = tif.pages[0].tags
                    model = str(tags.get("Model").value) if "Model" in tags else ""
                    software = (
                        str(tags.get("Software").value) if "Software" in tags else ""
                    )
        except Exception as err:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube", path, f"Not a readable TIFF file: {err}"
            )
        lowered = software.lower()
        if not (model.upper().startswith("HT") or "tomo" in lowered or "htx" in lowered):
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube",
                path,
                f"TIFF Model/Software tags ({model!r}/{software!r}) do not "
                "identify a Tomocube export.",
            )
        return True

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Optional[Dict[str, Any]] = None,
        timelapse: bool = False,
        refractive_index: bool = False,
    ) -> None:
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs or {},
        )
        self._is_supported_image(self._fs, self._path)
        anchor = parse_filename(self._path)
        assert anchor is not None  # guaranteed by _is_supported_image
        self._anchor: TomocubeFile = anchor
        self._timelapse = timelapse
        self._refractive_index = refractive_index

        self._files: Optional[Dict[str, List[TomocubeFile]]] = None
        self._scenes: Optional[Tuple[str, ...]] = None
        self._acquisition: Optional[Dict[str, _SceneInfo]] = None
        self._ome: Optional[OME] = None

        # Default to the scene of the file that was actually passed in.
        self._current_scene_index = self.scenes.index(self._anchor.scene)

    # ------------------------------------------------------------------
    # Discovery / caching
    # ------------------------------------------------------------------

    @property
    def files(self) -> Dict[str, List[TomocubeFile]]:
        """Scene name → files (one per timepoint) discovered for this image."""
        if self._files is None:
            self._files = discover_acquisition(
                self._fs, self._path, timelapse=self._timelapse
            )
        return self._files

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            self._scenes = tuple(self.files.keys())
        return self._scenes

    def _load_acquisition(self) -> Dict[str, _SceneInfo]:
        """Parse every discovered file's header once and cache per scene."""
        if self._acquisition is None:
            headers: Dict[str, Tuple[_FileHeader, ...]] = {
                scene: tuple(_read_header(self._fs, f) for f in members)
                for scene, members in self.files.items()
            }
            self._acquisition = {
                scene: _build_scene_info(
                    scene, hdrs, headers, self._refractive_index
                )
                for scene, hdrs in headers.items()
            }
        return self._acquisition

    def _scene_info(self) -> _SceneInfo:
        return self._load_acquisition()[self.current_scene]

    # ------------------------------------------------------------------
    # Array construction
    # ------------------------------------------------------------------

    def _build_xarray(self, delayed: bool) -> xr.DataArray:
        info = self._scene_info()
        plane_shape = (info.size_y, info.size_x)
        dim_names = [DimensionNames.Time] + list(
            _SPATIAL_DIMS[-(3 if info.size_z is not None else 2) :]
        )

        array_data: Any
        if delayed:
            timepoints: List[da.Array] = []
            for header in info.headers:
                planes = [
                    da.from_delayed(
                        dask.delayed(_read_plane)(
                            self._fs,
                            header.file.path,
                            location,
                            plane_shape,
                            header.dtype.str,
                            info.scale,
                        ),
                        shape=plane_shape,
                        dtype=info.dtype,
                    )
                    for location in header.planes
                ]
                if info.size_z is None:
                    timepoints.append(planes[0])
                else:
                    timepoints.append(da.stack(planes, axis=0))
            array_data = da.stack(timepoints, axis=0)
            log.debug(
                "scene=%r shape=%s chunks=%s",
                info.scene,
                array_data.shape,
                array_data.chunks,
            )
        else:
            volumes = [
                _read_volume(self._fs, header, info.scale) for header in info.headers
            ]
            if info.size_z is None:
                volumes = [v[0] for v in volumes]
            array_data = np.stack(volumes, axis=0)

        return xr.DataArray(
            array_data,
            dims=dim_names,
            attrs={
                constants.METADATA_UNPROCESSED: self.tiff_metadata,
                constants.METADATA_PROCESSED: self.ome_metadata,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        return self._build_xarray(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._build_xarray(delayed=False)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        return self._scene_info().dtype

    @property
    def tiff_metadata(self) -> Dict[str, Any]:
        """Raw TIFF/ImageJ metadata and file provenance for the current scene.

        Structured as::

            {
                "format": "tomocube-tiff",
                "scene": "3DFL/CH0",
                "files": [...],            # one path per timepoint
                "timepoints": [...],       # TP index per file, or None
                "acquisition": {...},      # parsed from the file name
                "tags": {...},             # TIFF tags of the first page
                "imagej": {...},           # parsed ImageJ description
                "z_spacing_source": "HT3D sibling" | "ImageJ header" | None,
                "refractive_index_scale": 1e-4 | None,
            }
        """
        info = self._scene_info()
        first = info.headers[0]
        acquisition = _parse_acquisition_name(first.file.base)
        acquisition.update(
            {
                "stem": first.file.stem,
                "modality": first.file.modality,
                "channel": first.file.channel,
                "suffix": first.file.suffix,
            }
        )
        return {
            "format": "tomocube-tiff",
            "scene": info.scene,
            "files": [h.file.path for h in info.headers],
            "timepoints": [h.file.timepoint for h in info.headers],
            "acquisition": acquisition,
            "tags": dict(first.tags),
            "imagej": dict(first.imagej),
            "z_spacing_source": info.z_spacing_source,
            "refractive_index_scale": info.scale,
        }

    @property
    def ome_metadata(self) -> OME:
        """OME metadata with one ``Image`` per scene (aligned with ``scenes``)."""
        if self._ome is None:
            acquisition = self._load_acquisition()
            ome_scenes: List[TiffOmeScene] = []
            for scene in self.scenes:
                info = acquisition[scene]
                first = info.headers[0]
                channel_name = (
                    "HT"
                    if first.file.is_ht
                    else f"FL CH{first.file.channel or 0}"
                )
                ome_scenes.append(
                    TiffOmeScene(
                        name=scene,
                        size_x=info.size_x,
                        size_y=info.size_y,
                        size_z=info.size_z or 1,
                        size_t=len(info.headers),
                        dtype=info.dtype,
                        pixel_sizes=info.pixel_sizes,
                        datetimes=tuple(h.datetime for h in info.headers),
                        channel_name=channel_name,
                        title=first.file.stem,
                    )
                )
            first_header = acquisition[self.scenes[0]].headers[0]
            self._ome = build_tiff_ome(
                ome_scenes,
                model=str(first_header.tags.get("Model", "")) or None,
                software=str(first_header.tags.get("Software", "")) or None,
                user=str(first_header.tags.get("Artist", "")) or None,
            )
        return self._ome

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """Physical pixel sizes in micrometres (see :attr:`tiff_metadata`)."""
        if self._physical_pixel_sizes is None:
            self._physical_pixel_sizes = self._scene_info().pixel_sizes
        return self._physical_pixel_sizes

    @property
    def time_interval(self) -> Optional[timedelta]:
        """Mean interval between timepoints from the TIFF ``DateTime`` tags."""
        return self._scene_info().time_interval

    @property
    def standard_metadata(self) -> StandardMetadata:
        sm = super().standard_metadata
        info = self._scene_info()
        first = info.headers[0]
        acquisition = _parse_acquisition_name(first.file.base)
        if "row" in acquisition:
            sm.row = acquisition["row"]
        if "column" in acquisition:
            sm.column = acquisition["column"]
        artist = str(first.tags.get("Artist", "")).strip()
        if artist:
            sm.imaged_by = artist
        return sm
