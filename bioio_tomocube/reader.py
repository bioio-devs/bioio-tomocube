import logging
from datetime import timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import dask
import dask.array as da
import h5py
import numpy as np
import xarray as xr
from bioio_base import constants, exceptions, io, reader, types
from bioio_base.dimensions import DimensionNames
from bioio_base.standard_metadata import StandardMetadata
from fsspec.spec import AbstractFileSystem
from ome_types.model import OME

from bioio_tomocube.ome_utils import (
    _deserialize_attr,
    _detect_channel_idx,
    _frame_key,
    _scene_group_key as _scene_group_key_util,
    build_ome,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# TCF HDF5 layout constants
_DATA_ROOT = "Data"
_FL_CHANNELS_ATTR = "Channels"
_BF_IMGTYPE = "BF"
_FL3D_IMGTYPE = "3DFL"

# Canonical spatial dim order (outermost → innermost)
_SPATIAL_DIMS = (
    DimensionNames.SpatialZ,
    DimensionNames.SpatialY,
    DimensionNames.SpatialX,
)

# BF imgtype returns PIL Images rather than arrays; skip it entirely
_SKIP_IMGTYPES = frozenset({_BF_IMGTYPE})

# Sentinel distinguishing "not yet cached" from None (a valid return value)
_NOT_CACHED: Any = object()


def _build_metadata(f: "h5py.File", scene_group_key: str) -> Dict[str, Any]:
    """Return structured metadata from an open TCF HDF5 file."""
    meta: Dict[str, Any] = {
        "root": {k: _deserialize_attr(v) for k, v in f.attrs.items()},
        "scene": {
            k: _deserialize_attr(v)
            for k, v in f[f"{_DATA_ROOT}/{scene_group_key}"].attrs.items()
        },
    }
    if "Info/Device" in f:
        meta["device"] = {
            k: _deserialize_attr(v) for k, v in f["Info/Device"].attrs.items()
        }
    if "Info/Imaging" in f:
        meta["imaging"] = {
            k: _deserialize_attr(v) for k, v in f["Info/Imaging"].attrs.items()
        }
    return meta


###############################################################################


class _SceneInfo(NamedTuple):
    """Everything read from the HDF5 file for the current scene in one pass."""

    n_frames: int
    size_y: int
    size_x: int
    size_z: Optional[int]
    channel_idx: Optional[int]
    stage_x: Optional[float]
    stage_y: Optional[float]
    tcf_metadata: Dict[str, Any]
    ome: OME


###############################################################################


class Reader(reader.Reader):
    """Read Tomocube TCF holotomography files.

    Parameters
    ----------
    image : Path or str
        Path to a ``.TCF`` file.  Any fsspec-compatible URI is accepted.
    fs_kwargs : Dict[str, Any]
        Keyword arguments forwarded to the fsspec filesystem (e.g. S3
        credentials).  Default: ``{}``.

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file does not have a ``.TCF`` extension.

    Notes
    -----
    Each imaging modality in the file is exposed as a separate **scene**:

    * ``"3D"`` — 3-D refractive-index volume (float32)
    * ``"2DMIP"`` — 2-D max-intensity projection (float32)
    * ``"3DFL/CH0"``, ``"3DFL/CH1"``, … — 3-D fluorescence channels
    """

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, **kwargs: Any
    ) -> bool:
        if path.upper().endswith(".TCF"):
            return True
        raise exceptions.UnsupportedFileFormatError(
            "bioio-tomocube", path, "File does not have a .TCF extension."
        )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs or {},
        )
        self._is_supported_image(self._fs, self._path)
        self._scenes: Optional[Tuple[str, ...]] = None
        self._scene_info: Optional[_SceneInfo] = None
        self._time_interval: Any = _NOT_CACHED  # None is a valid value

    def _reset_self(self) -> None:
        super()._reset_self()
        self._scene_info = None
        self._time_interval = _NOT_CACHED

    @property
    def scenes(self) -> Tuple[str, ...]:
        """All scene IDs in the file.

        Each HDF5 ``Data/`` key becomes a scene, except ``BF`` (brightfield,
        which stores PIL images rather than arrays).  ``3DFL`` is expanded
        into one scene per channel: ``"3DFL/CH0"``, ``"3DFL/CH1"``, …
        """
        if self._scenes is None:
            result: List[str] = []
            with self._fs.open(self._path, "rb") as fobj:
                with h5py.File(fobj, "r") as f:
                    for key in f[_DATA_ROOT]:
                        if key in _SKIP_IMGTYPES:
                            continue
                        if key == _FL3D_IMGTYPE:
                            n_ch = int(
                                _deserialize_attr(
                                    f[f"{_DATA_ROOT}/{_FL3D_IMGTYPE}"].attrs.get(
                                        _FL_CHANNELS_ATTR, np.array([1])
                                    )
                                )
                            )
                            for ch in range(n_ch):
                                result.append(f"{_FL3D_IMGTYPE}/CH{ch}")
                        else:
                            result.append(key)
            self._scenes = tuple(result)
        return self._scenes

    @staticmethod
    def _scene_group_key(scene: str) -> str:
        return _scene_group_key_util(scene)

    @staticmethod
    def _read_frame(
        f: "h5py.File",
        scene_group_key: str,
        channel_idx: Optional[int],
        frame_idx: int,
    ) -> np.ndarray:
        """Read one frame from an already-open HDF5 file handle."""
        key = _frame_key(scene_group_key, channel_idx, frame_idx)
        return np.asarray(f[key][()], dtype=np.float32)

    @staticmethod
    def _read_single_frame(
        path: str,
        scene: str,
        frame_idx: int,
        fs: AbstractFileSystem,
        channel_idx: Optional[int],
    ) -> np.ndarray:
        """Open the file, read one frame, and close — safe for dask workers."""
        scene_group_key = Reader._scene_group_key(scene)
        with fs.open(path, "rb") as fobj:
            with h5py.File(fobj, "r") as f:
                return Reader._read_frame(f, scene_group_key, channel_idx, frame_idx)

    def _load_scene_info(self) -> _SceneInfo:
        """Open the file once and cache all scene-level metadata and shape attrs."""
        if self._scene_info is None:
            scene = self.scenes[self.current_scene_index]
            scene_group_key = self._scene_group_key(scene)
            with self._fs.open(self._path, "rb") as fobj:
                with h5py.File(fobj, "r") as f:
                    group = f[f"{_DATA_ROOT}/{scene_group_key}"]
                    n_frames = int(_deserialize_attr(group.attrs["DataCount"]))
                    size_y = int(_deserialize_attr(group.attrs["SizeY"]))
                    size_x = int(_deserialize_attr(group.attrs["SizeX"]))
                    size_z_raw = group.attrs.get("SizeZ")
                    size_z = (
                        int(_deserialize_attr(size_z_raw))
                        if size_z_raw is not None
                        else None
                    )
                    channel_idx = _detect_channel_idx(f, scene, scene_group_key)
                    fkey = _frame_key(scene_group_key, channel_idx, 0)
                    stage_x = stage_y = None
                    if fkey in f:
                        raw_x = f[fkey].attrs.get("PositionX")
                        raw_y = f[fkey].attrs.get("PositionY")
                        if raw_x is not None:
                            stage_x = float(_deserialize_attr(raw_x))
                        if raw_y is not None:
                            stage_y = float(_deserialize_attr(raw_y))
                    tcf_meta = _build_metadata(f, scene_group_key)
                    ome = build_ome(f, self.scenes)
            self._scene_info = _SceneInfo(
                n_frames=n_frames,
                size_y=size_y,
                size_x=size_x,
                size_z=size_z,
                channel_idx=channel_idx,
                stage_x=stage_x,
                stage_y=stage_y,
                tcf_metadata=tcf_meta,
                ome=ome,
            )
        return self._scene_info

    def _build_xarray(self, delayed: bool) -> xr.DataArray:
        """Core builder shared by ``_read_delayed`` and ``_read_immediate``."""
        info = self._load_scene_info()
        scene = self.scenes[self.current_scene_index]
        scene_group_key = self._scene_group_key(scene)

        data_ndim = 3 if info.size_z is not None else 2
        spatial_shape: Tuple[int, ...] = (
            (info.size_z, info.size_y, info.size_x)
            if info.size_z is not None
            else (info.size_y, info.size_x)
        )
        spatial_dims = _SPATIAL_DIMS[-data_ndim:]
        dim_names = [DimensionNames.Time] + list(spatial_dims)

        if delayed:
            frames = [
                da.from_delayed(
                    dask.delayed(Reader._read_single_frame)(
                        self._path, scene, i, self._fs, info.channel_idx
                    ),
                    shape=spatial_shape,
                    dtype=np.float32,
                )
                for i in range(info.n_frames)
            ]
            array_data: Any = da.stack(frames, axis=0)
            log.debug(
                "scene=%r  shape=%s  chunks=%s",
                scene,
                array_data.shape,
                array_data.chunks,
            )
        else:
            with self._fs.open(self._path, "rb") as fobj:
                with h5py.File(fobj, "r") as f:
                    array_data = np.stack([
                        self._read_frame(f, scene_group_key, info.channel_idx, i)
                        for i in range(info.n_frames)
                    ])

        return xr.DataArray(
            array_data,
            dims=dim_names,
            attrs={
                constants.METADATA_UNPROCESSED: info.tcf_metadata,
                constants.METADATA_PROCESSED: info.ome,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        return self._build_xarray(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._build_xarray(delayed=False)

    @property
    def dtype(self) -> np.dtype:
        """All TCF pixel data is float32 — no file read needed."""
        return np.dtype(np.float32)

    @property
    def ome_metadata(self) -> OME:
        """OME metadata for the current scene, cached without building xarray."""
        return self._load_scene_info().ome

    @property
    def tcf_metadata(self) -> Dict[str, Any]:
        """Raw HDF5 attribute dictionary for the current scene.

        Structured as::

            {
                "root":    {...},   # file-level attrs (Title, CreateDate, …)
                "scene":   {...},   # Data/<scene> attrs (SizeX/Y/Z, …)
                "device":  {...},   # Info/Device attrs (Magnification, NA, …)
                "imaging": {...},   # Info/Imaging attrs (CameraGain, …)
            }
        """
        return self._load_scene_info().tcf_metadata

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """Physical pixel sizes in micrometres, read from the HDF5 scene group."""
        if self._physical_pixel_sizes is None:
            try:
                scene = self.scenes[self.current_scene_index]
                scene_group_key = self._scene_group_key(scene)
                with self._fs.open(self._path, "rb") as fobj:
                    with h5py.File(fobj, "r") as f:
                        group = f[f"{_DATA_ROOT}/{scene_group_key}"]
                        res_z_raw = group.attrs.get("ResolutionZ")
                        res_y_raw = group.attrs.get("ResolutionY")
                        res_x_raw = group.attrs.get("ResolutionX")
                self._physical_pixel_sizes = types.PhysicalPixelSizes(
                    Z=(
                        float(_deserialize_attr(res_z_raw))
                        if res_z_raw is not None
                        else None
                    ),
                    Y=(
                        float(_deserialize_attr(res_y_raw))
                        if res_y_raw is not None
                        else None
                    ),
                    X=(
                        float(_deserialize_attr(res_x_raw))
                        if res_x_raw is not None
                        else None
                    ),
                )
            except Exception as err:
                log.warning("Failed to read physical pixel sizes: %s", err)
                self._physical_pixel_sizes = types.PhysicalPixelSizes(
                    Z=None, Y=None, X=None
                )
        return self._physical_pixel_sizes

    @property
    def time_interval(self) -> Optional[timedelta]:
        """Return the acquisition time interval between frames, or ``None``."""
        if self._time_interval is _NOT_CACHED:
            scene = self.scenes[self.current_scene_index]
            scene_group_key = self._scene_group_key(scene)
            try:
                with self._fs.open(self._path, "rb") as fobj:
                    with h5py.File(fobj, "r") as f:
                        raw = f[f"{_DATA_ROOT}/{scene_group_key}"].attrs.get(
                            "TimeInterval"
                        )
                if raw is None:
                    self._time_interval = None
                else:
                    seconds = float(_deserialize_attr(raw))
                    self._time_interval = (
                        timedelta(seconds=seconds) if seconds > 0 else None
                    )
            except Exception as err:
                log.warning("Failed to read time interval: %s", err)
                self._time_interval = None
        return self._time_interval

    @property
    def standard_metadata(self) -> StandardMetadata:
        """StandardMetadata with stage position patched in from HDF5 frame attrs."""
        sm = super().standard_metadata
        info = self._load_scene_info()
        if info.stage_x is not None:
            sm.stage_position_x = info.stage_x
        if info.stage_y is not None:
            sm.stage_position_y = info.stage_y
        return sm
