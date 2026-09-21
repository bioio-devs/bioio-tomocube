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
files of the same acquisition.  The export places every modality on the same
pixel grid, so the modalities are combined into the **channels** of one image:

* scene ``"3D"`` — ``TCZYX`` with channels ``HT``, ``FL_CH0``, ``FL_CH1``, …
* scene ``"2DMIP"`` — ``TCYX`` max projections with the same channel naming

Known quirks of the export that this reader compensates for:

* ``HT3D`` stores refractive index multiplied by 10 000 as ``uint16``.
* ``FL3D`` volumes are resampled onto the HT Z grid (same page count as the
  ``HT3D`` file) but the ImageJ header still reports the *original* FL slice
  count and spacing.  ``tifffile`` trusts that header and reports the wrong
  shape, so this reader always enumerates TIFF pages directly and takes the
  Z spacing from the ``HT`` channel.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

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

#: Channel name used for the holotomography (refractive index) modality.
HT_CHANNEL = "HT"

_MODALITY_TO_SCENE = {
    "HT3D": "3D",
    "FL3D": "3D",
    "HT2D": "2DMIP",
    "FL2D": "2DMIP",
}
_HT_MODALITIES = frozenset({"HT3D", "HT2D"})
_VOLUME_MODALITIES = frozenset({"HT3D", "FL3D"})
_SCENE_ORDER = ("3D", "2DMIP")

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
        """Scene this file contributes to (``"3D"`` or ``"2DMIP"``)."""
        return _MODALITY_TO_SCENE[self.modality]

    @property
    def is_ht(self) -> bool:
        return self.modality in _HT_MODALITIES

    @property
    def channel_name(self) -> str:
        """Channel name (``"HT"``, ``"FL_CH0"``, …) within the scene."""
        if self.is_ht:
            return HT_CHANNEL
        return f"FL_CH{self.channel or 0}"


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


def _channel_sort_key(name: str) -> Tuple[int, int]:
    """``HT`` first, then fluorescence channels by index."""
    if name == HT_CHANNEL:
        return (0, -1)
    match = re.search(r"(\d+)$", name)
    return (1, int(match.group(1)) if match else 0)


def _scene_sort_key(scene: str) -> Tuple[int, str]:
    prefix = scene.split("/", 1)[0]
    order = _SCENE_ORDER.index(prefix) if prefix in _SCENE_ORDER else len(_SCENE_ORDER)
    return order, scene


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
) -> List[TomocubeFile]:
    """Find every exported file belonging to the acquisition *path* is part of.

    Parameters
    ----------
    fs, path:
        Filesystem and path of any one exported TIFF (the *anchor*).
    timelapse:
        When ``False`` only files of the anchor's timepoint are returned.
        When ``True`` all timepoints sharing the anchor's ``<base>`` are
        collected so each scene gains a T dimension.

    Returns
    -------
    List[TomocubeFile]
        Every matching file, sorted by scene, timepoint and channel.
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

    found: List[TomocubeFile] = []
    seen: set = set()
    for candidate in candidates:
        info = parse_filename(candidate)
        if info is None or info.base != anchor.base:
            continue
        if not timelapse and info.timepoint != anchor.timepoint:
            continue
        key = (info.scene, info.channel_name, info.timepoint, _basename(candidate))
        if key in seen:
            continue
        seen.add(key)
        found.append(info)

    # Listing may be unavailable on some filesystems; always include the anchor.
    anchor_key = (anchor.scene, anchor.channel_name, anchor.timepoint, _basename(path))
    if anchor_key not in seen:
        found.append(anchor)

    found.sort(
        key=lambda f: (
            _scene_sort_key(f.scene),
            f.timepoint is not None,
            f.timepoint or 0,
            _channel_sort_key(f.channel_name),
        )
    )
    return found


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

    @property
    def grid(self) -> Tuple[int, int, int, str]:
        """Signature that must match for files to share one array."""
        return (self.n_pages, self.size_y, self.size_x, self.dtype.str)


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
                        _PlaneLocation(index, int(page.dataoffsets[0]), expected_nbytes)
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
    channels: Tuple[str, ...]
    timepoints: Tuple[Optional[int], ...]
    headers: Tuple[Tuple[_FileHeader, ...], ...]  # indexed ``[t][c]``
    size_z: Optional[int]  # ``None`` for 2-D scenes (dims ``TCYX``)
    size_y: int
    size_x: int
    dtype: np.dtype  # dtype of the *returned* array
    scales: Tuple[Optional[float], ...]  # per channel; multiply raw values
    pixel_sizes: types.PhysicalPixelSizes
    z_spacing_source: Optional[str]
    time_interval: Optional[timedelta]

    @property
    def datetimes(self) -> Tuple[Optional[datetime], ...]:
        """One acquisition time per T (from the HT channel when present)."""
        c_idx = self.channels.index(HT_CHANNEL) if HT_CHANNEL in self.channels else 0
        return tuple(row[c_idx].datetime for row in self.headers)


def _mean_interval(datetimes: Sequence[Optional[datetime]]) -> Optional[timedelta]:
    if len(datetimes) < 2 or any(dt is None for dt in datetimes):
        return None
    first, last = datetimes[0], datetimes[-1]
    assert first is not None and last is not None
    return (last - first) / (len(datetimes) - 1)


def _tp_sort_key(tp: Optional[int]) -> Tuple[bool, int]:
    return (tp is None, tp or 0)


def _build_scene_info(
    scene: str, headers: Sequence[_FileHeader], refractive_index: bool
) -> _SceneInfo:
    """Combine one scene's files (all channels, all timepoints) into a grid.

    All *headers* must share the same pixel grid; callers guarantee this via
    :func:`_group_by_grid`.  Channels are the union over timepoints; only
    timepoints at which **every** channel is present are kept.
    """
    channels = tuple(
        sorted({h.file.channel_name for h in headers}, key=_channel_sort_key)
    )
    by_tp: Dict[Optional[int], Dict[str, _FileHeader]] = {}
    for header in headers:
        by_tp.setdefault(header.file.timepoint, {})[header.file.channel_name] = header

    complete = [tp for tp, chans in by_tp.items() if set(chans) == set(channels)]
    dropped = sorted((tp for tp in by_tp if tp not in complete), key=_tp_sort_key)
    if not complete:
        raise exceptions.UnsupportedFileFormatError(
            "bioio-tomocube",
            headers[0].file.path,
            f"Scene {scene!r}: no timepoint has all channels {channels}.",
        )
    if dropped:
        log.warning(
            "Scene %s: dropping timepoint(s) %s because not every channel %s "
            "was exported for them.",
            scene,
            dropped,
            channels,
        )
    timepoints = tuple(sorted(complete, key=_tp_sort_key))
    grid = tuple(tuple(by_tp[tp][c] for c in channels) for tp in timepoints)

    first = grid[0][0]
    modality = first.file.modality
    is_volume = modality in _VOLUME_MODALITIES or first.n_pages > 1
    size_z: Optional[int] = first.n_pages if is_volume else None

    scales = tuple(
        RI_SCALE if (refractive_index and c == HT_CHANNEL) else None for c in channels
    )
    out_dtype = first.dtype.newbyteorder("=")
    if any(s is not None for s in scales):
        out_dtype = np.dtype(np.float32)

    # Z spacing: prefer the HT header.  FL3D headers carry the pre-resampling
    # FL spacing even though the pages are on the HT grid.
    z_spacing: Optional[float] = None
    z_source: Optional[str] = None
    if is_volume:
        ht = next((h for h in grid[0] if h.file.is_ht), None)
        if ht is not None and ht.z_spacing is not None:
            z_spacing, z_source = ht.z_spacing, "HT header"
            for other in grid[0]:
                if (
                    other is not ht
                    and other.z_spacing is not None
                    and not np.isclose(other.z_spacing, ht.z_spacing)
                ):
                    log.info(
                        "Scene %s channel %s: ImageJ header Z spacing %.4f µm "
                        "differs from the HT grid (%.4f µm) the pages are stored "
                        "on; using the HT spacing.",
                        scene,
                        other.file.channel_name,
                        other.z_spacing,
                        ht.z_spacing,
                    )
        else:
            fallback = next((h for h in grid[0] if h.z_spacing is not None), None)
            if fallback is not None:
                z_spacing, z_source = fallback.z_spacing, "ImageJ header"

    info = _SceneInfo(
        scene=scene,
        channels=channels,
        timepoints=timepoints,
        headers=grid,
        size_z=size_z,
        size_y=first.size_y,
        size_x=first.size_x,
        dtype=out_dtype,
        scales=scales,
        pixel_sizes=types.PhysicalPixelSizes(
            Z=z_spacing, Y=first.pixel_size_y, X=first.pixel_size_x
        ),
        z_spacing_source=z_source,
        time_interval=None,
    )
    return info._replace(time_interval=_mean_interval(info.datetimes))


def _group_by_grid(
    scene: str, headers: Sequence[_FileHeader]
) -> Dict[str, List[_FileHeader]]:
    """Split a scene's files into groups that can share one array.

    Normally every modality of an export sits on the same grid and a single
    group named *scene* results.  If grids differ, each channel becomes its
    own scene named ``"<scene>/<channel>"`` and a warning is logged.
    """
    grids = {h.grid for h in headers}
    if len(grids) == 1:
        return {scene: list(headers)}

    by_channel: Dict[str, List[_FileHeader]] = {}
    for header in headers:
        by_channel.setdefault(header.file.channel_name, []).append(header)
    log.warning(
        "Scene %s: channels %s are not on a common pixel grid (%s); exposing "
        "them as separate scenes instead of stacking along C.",
        scene,
        sorted(by_channel, key=_channel_sort_key),
        sorted(grids),
    )
    result: Dict[str, List[_FileHeader]] = {}
    for channel in sorted(by_channel, key=_channel_sort_key):
        members = by_channel[channel]
        if len({h.grid for h in members}) != 1:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-tomocube",
                members[0].file.path,
                f"Channel {channel!r} of scene {scene!r} changes shape or dtype "
                "between timepoints.",
            )
        result[f"{scene}/{channel}"] = members
    return result


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


def _to_output(arr: np.ndarray, scale: Optional[float], dtype: np.dtype) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("="))
    if scale is not None:
        arr = arr.astype(np.float32) * np.float32(scale)
    return arr.astype(dtype, copy=False)


def _read_plane(
    fs: AbstractFileSystem,
    path: str,
    location: _PlaneLocation,
    shape: Tuple[int, int],
    stored_dtype: str,
    scale: Optional[float],
    out_dtype: np.dtype,
) -> np.ndarray:
    """Read one Z plane; opens and closes the file (safe on dask workers)."""
    with fs.open(path, "rb") as fobj:
        if location.offset >= 0:
            fobj.seek(location.offset)
            buffer = fobj.read(location.nbytes)
            arr = np.frombuffer(buffer, dtype=np.dtype(stored_dtype)).reshape(shape)
        else:
            with tifffile.TiffFile(fobj) as tif:
                arr = tif.pages[location.page_index].asarray()
    return _to_output(arr, scale, out_dtype)


def _read_volume(
    fs: AbstractFileSystem,
    header: _FileHeader,
    scale: Optional[float],
    out_dtype: np.dtype,
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
    return _to_output(np.stack(planes, axis=0), scale, out_dtype)


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
        When ``True`` the ``HT`` channel is converted to refractive index
        (raw ``uint16`` × ``1e-4``) and the whole scene is returned as
        ``float32``.  Default: ``False`` (raw ``uint16`` as stored).

    Notes
    -----
    The export places every modality on one pixel grid, so modalities become
    **channels** rather than separate scenes:

    * ``"3D"`` — ``TCZYX``; channels ``HT``, ``FL_CH0``, ``FL_CH1``, …
    * ``"2DMIP"`` — ``TCYX`` max projections with the same channel naming,
      present when ``HT2D``/``FL2D`` files were exported

    Only timepoints at which every channel of a scene exists are kept; others
    are dropped with a warning.  Should an export ever place modalities on
    different grids they are exposed as ``"3D/HT"``, ``"3D/FL_CH0"``, … instead.

    The default scene is the one containing the file passed as *image*.
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
        if not (
            model.upper().startswith("HT") or "tomo" in lowered or "htx" in lowered
        ):
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

        self._acquisition: Optional[Dict[str, _SceneInfo]] = None
        self._scenes: Optional[Tuple[str, ...]] = None
        self._ome: Optional[OME] = None

        # Default to the scene containing the file that was actually passed in.
        self._current_scene_index = self.scenes.index(self._anchor_scene())

    # ------------------------------------------------------------------
    # Discovery / caching
    # ------------------------------------------------------------------

    def _load_acquisition(self) -> Dict[str, _SceneInfo]:
        """Discover files, parse every header once and assemble the scenes."""
        if self._acquisition is None:
            files = discover_acquisition(
                self._fs, self._path, timelapse=self._timelapse
            )
            headers = [_read_header(self._fs, f) for f in files]
            by_scene: Dict[str, List[_FileHeader]] = {}
            for header in headers:
                by_scene.setdefault(header.file.scene, []).append(header)

            acquisition: Dict[str, _SceneInfo] = {}
            for scene in sorted(by_scene, key=_scene_sort_key):
                for name, members in _group_by_grid(scene, by_scene[scene]).items():
                    acquisition[name] = _build_scene_info(
                        name, members, self._refractive_index
                    )
            self._acquisition = acquisition
        return self._acquisition

    def _anchor_scene(self) -> str:
        """Name of the scene that contains the file passed to the constructor."""
        anchor_name = _basename(self._path)
        for name, info in self._load_acquisition().items():
            for row in info.headers:
                if any(_basename(h.file.path) == anchor_name for h in row):
                    return name
        # Anchor timepoint may have been dropped; fall back to its scene group.
        candidates = [
            s for s in self.scenes if s.split("/", 1)[0] == self._anchor.scene
        ]
        return candidates[0] if candidates else self.scenes[0]

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            self._scenes = tuple(self._load_acquisition().keys())
        return self._scenes

    def _scene_info(self) -> _SceneInfo:
        return self._load_acquisition()[self.current_scene]

    @property
    def files(self) -> Dict[str, Dict[str, List[TomocubeFile]]]:
        """``scene → channel → files`` (one per timepoint) backing this image."""
        result: Dict[str, Dict[str, List[TomocubeFile]]] = {}
        for name, info in self._load_acquisition().items():
            result[name] = {
                channel: [row[c_idx].file for row in info.headers]
                for c_idx, channel in enumerate(info.channels)
            }
        return result

    # ------------------------------------------------------------------
    # Array construction
    # ------------------------------------------------------------------

    def _build_xarray(self, delayed: bool) -> xr.DataArray:
        info = self._scene_info()
        plane_shape = (info.size_y, info.size_x)
        dim_names = [DimensionNames.Time, DimensionNames.Channel]
        if info.size_z is not None:
            dim_names.append(DimensionNames.SpatialZ)
        dim_names += [DimensionNames.SpatialY, DimensionNames.SpatialX]

        array_data: Any
        if delayed:
            timepoints: List[da.Array] = []
            for row in info.headers:
                channels: List[da.Array] = []
                for header, scale in zip(row, info.scales):
                    planes = [
                        da.from_delayed(
                            dask.delayed(_read_plane)(
                                self._fs,
                                header.file.path,
                                location,
                                plane_shape,
                                header.dtype.str,
                                scale,
                                info.dtype,
                            ),
                            shape=plane_shape,
                            dtype=info.dtype,
                        )
                        for location in header.planes
                    ]
                    channels.append(
                        planes[0] if info.size_z is None else da.stack(planes, axis=0)
                    )
                timepoints.append(da.stack(channels, axis=0))
            array_data = da.stack(timepoints, axis=0)
            log.debug(
                "scene=%r shape=%s chunks=%s",
                info.scene,
                array_data.shape,
                array_data.chunks,
            )
        else:
            timepoint_arrays: List[np.ndarray] = []
            for row in info.headers:
                volumes = [
                    _read_volume(self._fs, header, scale, info.dtype)
                    for header, scale in zip(row, info.scales)
                ]
                if info.size_z is None:
                    volumes = [v[0] for v in volumes]
                timepoint_arrays.append(np.stack(volumes, axis=0))
            array_data = np.stack(timepoint_arrays, axis=0)

        return xr.DataArray(
            array_data,
            dims=dim_names,
            coords={DimensionNames.Channel: list(info.channels)},
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
    def channel_names(self) -> Optional[List[str]]:
        return list(self._scene_info().channels)

    @property
    def tiff_metadata(self) -> Dict[str, Any]:
        """Raw TIFF/ImageJ metadata and file provenance for the current scene.

        Structured as::

            {
                "format": "tomocube-tiff",
                "scene": "3D",
                "channels": ["HT", "FL_CH0", "FL_CH1"],
                "timepoints": [1, 2, ...],          # TP index per T, or None
                "files": {"HT": [...], "FL_CH0": [...], ...},  # one per T
                "acquisition": {...},               # parsed from the file name
                "tags": {"HT": {...}, ...},         # TIFF tags, first page, T=0
                "imagej": {"HT": {...}, ...},       # parsed ImageJ description
                "z_spacing_source": "HT header" | "ImageJ header" | None,
                "refractive_index_scale": {"HT": 1e-4, "FL_CH0": None, ...},
            }
        """
        info = self._scene_info()
        first_row = info.headers[0]
        acquisition = _parse_acquisition_name(first_row[0].file.base)
        acquisition["stem"] = first_row[0].file.stem
        acquisition["suffix"] = first_row[0].file.suffix
        acquisition["modalities"] = {
            channel: header.file.modality
            for channel, header in zip(info.channels, first_row)
        }
        return {
            "format": "tomocube-tiff",
            "scene": info.scene,
            "channels": list(info.channels),
            "timepoints": list(info.timepoints),
            "files": {
                channel: [row[c_idx].file.path for row in info.headers]
                for c_idx, channel in enumerate(info.channels)
            },
            "acquisition": acquisition,
            "tags": {
                channel: dict(header.tags)
                for channel, header in zip(info.channels, first_row)
            },
            "imagej": {
                channel: dict(header.imagej)
                for channel, header in zip(info.channels, first_row)
            },
            "z_spacing_source": info.z_spacing_source,
            "refractive_index_scale": dict(zip(info.channels, info.scales)),
        }

    @property
    def ome_metadata(self) -> OME:
        """OME metadata with one ``Image`` per scene (aligned with ``scenes``)."""
        if self._ome is None:
            acquisition = self._load_acquisition()
            ome_scenes = [
                TiffOmeScene(
                    name=scene,
                    size_x=info.size_x,
                    size_y=info.size_y,
                    size_z=info.size_z or 1,
                    size_t=len(info.headers),
                    dtype=info.dtype,
                    pixel_sizes=info.pixel_sizes,
                    datetimes=info.datetimes,
                    channel_names=info.channels,
                    title=info.headers[0][0].file.stem,
                )
                for scene, info in ((s, acquisition[s]) for s in self.scenes)
            ]
            first_header = acquisition[self.scenes[0]].headers[0][0]
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
        first = self._scene_info().headers[0][0]
        acquisition = _parse_acquisition_name(first.file.base)
        if "row" in acquisition:
            sm.row = acquisition["row"]
        if "column" in acquisition:
            sm.column = acquisition["column"]
        artist = str(first.tags.get("Artist", "")).strip()
        if artist:
            sm.imaged_by = artist
        return sm
