from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import dask
import dask.array as da
import h5py
import numpy as np
import xarray as xr
from bioio_base import constants, exceptions, io, types
from bioio_base.dimensions import DimensionNames
from bioio_base.reader import Reader as BaseReader
from fsspec.spec import AbstractFileSystem

from bioio_tomocube.metadata import attr, group_attrs

###############################################################################

# Brightfield groups hold encoded images rather than arrays; not supported.
_SKIPPED_GROUPS = frozenset({"BF"})


class _SceneInfo(NamedTuple):
    n_frames: int
    spatial_shape: Tuple[int, ...]  # ZYX or YX
    pixel_sizes: types.PhysicalPixelSizes
    stage_x: Optional[float]
    stage_y: Optional[float]
    tcf_metadata: Dict[str, Any]


def _read_frame(
    fs: AbstractFileSystem, path: str, scene: str, index: int
) -> np.ndarray:
    """Open the file, read one frame as float32, close. Safe on dask workers."""
    with fs.open(path, "rb") as fobj, h5py.File(fobj, "r") as f:
        return np.asarray(f[f"Data/{scene}/{index:06d}"][()], dtype=np.float32)


class Reader(BaseReader):
    """Read Tomocube ``.TCF`` files.

    Parameters
    ----------
    image : Path or str
        Path to a ``.TCF`` file. Any fsspec-compatible URI is accepted.
    fs_kwargs : Dict[str, Any]
        Keyword arguments forwarded to the fsspec filesystem.
    """

    _physical_pixel_sizes: Optional[types.PhysicalPixelSizes]

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        if path.upper().endswith(".TCF"):
            return True
        raise exceptions.UnsupportedFileFormatError(
            "bioio-tomocube", path, "File does not have a .TCF extension."
        )

    def __init__(
        self, image: types.PathLike, fs_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        self._fs, self._path = io.pathlike_to_fs(
            image, enforce_exists=True, fs_kwargs=fs_kwargs or {}
        )
        self._is_supported_image(self._fs, self._path)
        self._scenes: Optional[Tuple[str, ...]] = None
        self._scene_info: Optional[_SceneInfo] = None

    def _reset_self(self) -> None:
        super()._reset_self()
        self._scene_info = None

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            found: List[str] = []
            with self._fs.open(self._path, "rb") as fobj, h5py.File(fobj, "r") as f:
                for name, group in f["Data"].items():
                    if name in _SKIPPED_GROUPS:
                        continue
                    children = list(group.values())
                    if children and isinstance(children[0], h5py.Group):
                        found.extend(f"{name}/{ch}" for ch in group)
                    else:
                        found.append(name)
            self._scenes = tuple(found)
        return self._scenes

    def _load_scene_info(self) -> _SceneInfo:
        """Open the file once per scene and cache everything but pixels."""
        if self._scene_info is None:
            scene = self.current_scene
            modality = scene.split("/")[0]
            with self._fs.open(self._path, "rb") as fobj, h5py.File(fobj, "r") as f:
                mod = group_attrs(f[f"Data/{modality}"])
                first = f[f"Data/{scene}/000000"]
                shape = (
                    (int(mod["SizeZ"]), int(mod["SizeY"]), int(mod["SizeX"]))
                    if "SizeZ" in mod
                    else (int(mod["SizeY"]), int(mod["SizeX"]))
                )
                res = [mod.get(f"Resolution{axis}") for axis in "ZYX"]
                meta: Dict[str, Any] = {
                    "root": group_attrs(f),
                    "scene": mod,
                    "info": {k: group_attrs(v) for k, v in f["Info"].items()},
                }
                if "/" in scene:
                    meta["channel"] = group_attrs(f[f"Data/{scene}"])
                self._scene_info = _SceneInfo(
                    n_frames=int(mod["DataCount"]),
                    spatial_shape=shape,
                    pixel_sizes=types.PhysicalPixelSizes(
                        *(float(r) if r is not None else None for r in res)
                    ),
                    stage_x=attr(first, "PositionX"),
                    stage_y=attr(first, "PositionY"),
                    tcf_metadata=meta,
                )
        return self._scene_info

    def _build_xarray(self, delayed: bool) -> xr.DataArray:
        info = self._load_scene_info()
        scene = self.current_scene
        spatial_dims = [
            DimensionNames.SpatialZ,
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
        ][-len(info.spatial_shape) :]

        if delayed:
            data: Any = da.stack(
                [
                    da.from_delayed(
                        dask.delayed(_read_frame)(self._fs, self._path, scene, i),
                        shape=info.spatial_shape,
                        dtype=np.float32,
                    )
                    for i in range(info.n_frames)
                ]
            )
        else:
            with self._fs.open(self._path, "rb") as fobj, h5py.File(fobj, "r") as f:
                data = np.stack(
                    [
                        np.asarray(f[f"Data/{scene}/{i:06d}"][()], dtype=np.float32)
                        for i in range(info.n_frames)
                    ]
                )

        return xr.DataArray(
            data,
            dims=[DimensionNames.Time, *spatial_dims],
            attrs={
                constants.METADATA_UNPROCESSED: info.tcf_metadata,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        return self._build_xarray(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._build_xarray(delayed=False)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        return self._load_scene_info().pixel_sizes
