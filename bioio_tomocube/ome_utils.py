#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import h5py
import numpy as np
from bioio_base.types import PhysicalPixelSizes
from ome_types.model import (
    OME,
    Channel,
    Experimenter,
    ExperimenterRef,
    Image,
    Instrument,
    InstrumentRef,
    Microscope,
    Objective,
    Objective_Immersion,
    Pixels,
    Pixels_DimensionOrder,
    PixelType,
    Plane,
    UnitsLength,
    UnitsTime,
)

log = logging.getLogger(__name__)

_DATA_ROOT = "Data"
_FL3D_IMGTYPE = "3DFL"


def _deserialize_attr(val: Any) -> Any:
    """Convert an h5py attribute value to a plain Python scalar or list."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return _deserialize_attr(val.flat[0])
        return [_deserialize_attr(s) for s in val.flat]
    if isinstance(val, np.generic):
        return _deserialize_attr(val.item())
    return val


def _scene_group_key(scene: str) -> str:
    """Return the HDF5 ``Data/<key>`` for *scene*."""
    if scene.startswith(f"{_FL3D_IMGTYPE}/CH"):
        return _FL3D_IMGTYPE
    return scene


def _detect_channel_idx(
    f: "h5py.File", scene: str, scene_group_key: str
) -> Optional[int]:
    """Return the CH index for channel-based frame layouts, or ``None``."""
    if scene.startswith(f"{_FL3D_IMGTYPE}/CH"):
        return int(scene.removeprefix(f"{_FL3D_IMGTYPE}/CH"))
    first_key = next(iter(f[f"{_DATA_ROOT}/{scene_group_key}"].keys()), None)
    if first_key is not None and first_key.startswith("CH"):
        return 0
    return None


def _frame_key(scene_group_key: str, channel_idx: Optional[int], frame_idx: int) -> str:
    """Return the HDF5 path for one frame dataset."""
    if channel_idx is not None:
        return f"{_DATA_ROOT}/{scene_group_key}/CH{channel_idx}/{frame_idx:06d}"
    return f"{_DATA_ROOT}/{scene_group_key}/{frame_idx:06d}"


def _ri_to_immersion(ri: float) -> Optional[Objective_Immersion]:
    """Map a refractive-index value to the closest OME Immersion enum entry."""
    if abs(ri - 1.0) < 0.01:
        return Objective_Immersion.AIR
    if abs(ri - 1.333) < 0.01:
        return Objective_Immersion.WATER
    if abs(ri - 1.515) < 0.02:
        return Objective_Immersion.OIL
    if abs(ri - 1.47) < 0.02:
        return Objective_Immersion.GLYCEROL
    return Objective_Immersion.OTHER


def _build_image(
    f: "h5py.File",
    scene: str,
    image_index: int,
    dev: Dict[str, Any],
    root: Dict[str, Any],
    has_instrument: bool,
    has_experimenter: bool,
) -> Image:
    """Build one OME ``Image`` for a single TCF scene.

    Parameters
    ----------
    f:
        Open ``h5py.File`` handle.
    scene:
        Scene name (e.g. ``"3D"``, ``"2DMIP"``, ``"3DFL/CH0"``).
    image_index:
        Position of this image in the OME images list — must equal
        ``scenes.index(scene)`` so ``ome.images[i]`` aligns with
        ``Reader.scenes[i]``.
    dev:
        Deserialized ``Info/Device`` attribute dict.
    root:
        Deserialized root HDF5 attribute dict.
    has_instrument:
        Whether an ``Instrument`` will be present in the parent OME; controls
        whether an ``InstrumentRef`` is added to this image.
    has_experimenter:
        Whether an ``Experimenter`` will be present in the parent OME; controls
        whether an ``ExperimenterRef`` is added to this image.
    """
    scene_group_key = _scene_group_key(scene)
    group = f[f"{_DATA_ROOT}/{scene_group_key}"]
    sg = {k: _deserialize_attr(v) for k, v in group.attrs.items()}
    channel_idx = _detect_channel_idx(f, scene, scene_group_key)

    # Per-frame RecordingTime → absolute timestamps for Plane delta_t
    frame_times: List[Optional[datetime]] = []
    n_frames = int(sg.get("DataCount", 1))
    for i in range(n_frames):
        fkey = _frame_key(scene_group_key, channel_idx, i)
        if fkey in f:
            rt = _deserialize_attr(f[fkey].attrs.get("RecordingTime", ""))
            try:
                frame_times.append(
                    datetime.strptime(rt, "%Y-%m-%d %H:%M:%S.%f") if rt else None
                )
            except ValueError:
                frame_times.append(None)
        else:
            frame_times.append(None)

    # Acquisition datetime from file-level CreateDate
    acq_date: Optional[datetime] = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            acq_date = datetime.strptime(root.get("CreateDate", ""), fmt)
            break
        except ValueError:
            pass

    size_x = int(sg.get("SizeX", 1))
    size_y = int(sg.get("SizeY", 1))
    size_z = int(sg.get("SizeZ", 1)) if "SizeZ" in sg else 1

    res_x = sg.get("ResolutionX")
    res_y = sg.get("ResolutionY")
    res_z = sg.get("ResolutionZ")

    wave_um = dev.get("Wavelength")
    emission_nm: Optional[float] = float(wave_um) * 1000.0 if wave_um else None

    # Plane objects carry per-frame delta_t for timelapse helpers
    planes: List[Plane] = []
    t0 = frame_times[0] if frame_times else None
    for t_idx, ft in enumerate(frame_times):
        delta_t: Optional[float] = None
        if ft is not None and t0 is not None:
            delta_t = (ft - t0).total_seconds()
        elif sg.get("TimeInterval") is not None:
            delta_t = float(sg["TimeInterval"]) * t_idx
        planes.append(
            Plane(
                the_t=t_idx,
                the_z=0,
                the_c=0,
                **(
                    {"delta_t": delta_t, "delta_t_unit": UnitsTime.SECOND}
                    if delta_t is not None
                    else {}
                ),
            )
        )

    ch_kwargs: Dict[str, Any] = {
        "id": f"Channel:{image_index}:0",
        "samples_per_pixel": 1,
    }
    if emission_nm is not None:
        ch_kwargs["emission_wavelength"] = emission_nm
        ch_kwargs["emission_wavelength_unit"] = UnitsLength.NANOMETER

    px_kwargs: Dict[str, Any] = {
        "id": f"Pixels:{image_index}",
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "size_t": n_frames,
        "size_c": 1,
        "type": PixelType.FLOAT,
        "dimension_order": Pixels_DimensionOrder.XYZCT,
        "channels": [Channel(**ch_kwargs)],
        "planes": planes,
    }
    if res_x is not None:
        px_kwargs["physical_size_x"] = float(res_x)
        px_kwargs["physical_size_x_unit"] = UnitsLength.MICROMETER
    if res_y is not None:
        px_kwargs["physical_size_y"] = float(res_y)
        px_kwargs["physical_size_y_unit"] = UnitsLength.MICROMETER
    if res_z is not None:
        px_kwargs["physical_size_z"] = float(res_z)
        px_kwargs["physical_size_z_unit"] = UnitsLength.MICROMETER
    if sg.get("TimeInterval") is not None:
        px_kwargs["time_increment"] = float(sg["TimeInterval"])
        px_kwargs["time_increment_unit"] = UnitsTime.SECOND

    img_kwargs: Dict[str, Any] = {
        "id": f"Image:{image_index}",
        "name": root.get("Title", scene),
        "pixels": Pixels(**px_kwargs),
    }
    if acq_date is not None:
        img_kwargs["acquisition_date"] = acq_date
    if has_instrument:
        img_kwargs["instrument_ref"] = InstrumentRef(id="Instrument:0")
    if has_experimenter:
        img_kwargs["experimenter_ref"] = ExperimenterRef(id="Experimenter:0")

    return Image(**img_kwargs)


def build_ome(f: "h5py.File", scenes: Tuple[str, ...]) -> OME:
    """Construct a multi-image OME from an open TCF HDF5 file.

    One :class:`ome_types.model.Image` is built per scene so that
    ``ome.images[i]`` corresponds to ``scenes[i]``.  This alignment lets the
    bioio-base ``standard_metadata`` helper index into the OME with
    ``Reader.current_scene_index`` correctly.

    Parameters
    ----------
    f:
        Open ``h5py.File`` handle.
    scenes:
        Ordered scene names as returned by :pyattr:`Reader.scenes`.

    Returns
    -------
    OME
        A valid ``ome_types.model.OME`` instance with one image per scene,
        a shared instrument (objective), and an experimenter when the file
        carries a non-empty ``UserID`` attribute.
    """
    root = {k: _deserialize_attr(v) for k, v in f.attrs.items()}

    dev: Dict[str, Any] = {}
    if "Info/Device" in f:
        dev = {k: _deserialize_attr(v) for k, v in f["Info/Device"].attrs.items()}

    # Shared instrument / objective (same optics for every scene)
    instrument: Optional[Instrument] = None
    mag = dev.get("Magnification")
    na = dev.get("NA")
    ri = dev.get("RI")
    if mag is not None or na is not None:
        obj_kwargs: Dict[str, Any] = {"id": "Objective:0"}
        if mag is not None:
            obj_kwargs["nominal_magnification"] = float(mag)
        if na is not None:
            obj_kwargs["lens_na"] = float(na)
        if ri is not None:
            obj_kwargs["immersion"] = _ri_to_immersion(float(ri))
        instrument = Instrument(id="Instrument:0", objectives=[Objective(**obj_kwargs)])

    user_id = root.get("UserID", "").strip()
    experimenters: List[Experimenter] = []
    if user_id:
        experimenters.append(Experimenter(id="Experimenter:0", user_name=user_id))

    images = [
        _build_image(
            f,
            scene,
            i,
            dev,
            root,
            has_instrument=instrument is not None,
            has_experimenter=bool(experimenters),
        )
        for i, scene in enumerate(scenes)
    ]

    ome_kwargs: Dict[str, Any] = {"images": images}
    if instrument is not None:
        ome_kwargs["instruments"] = [instrument]
    if experimenters:
        ome_kwargs["experimenters"] = experimenters

    return OME(**ome_kwargs)


###############################################################################
# TIFF export → OME
###############################################################################

_NUMPY_TO_OME_PIXEL_TYPE = {
    "int8": PixelType.INT8,
    "int16": PixelType.INT16,
    "int32": PixelType.INT32,
    "uint8": PixelType.UINT8,
    "uint16": PixelType.UINT16,
    "uint32": PixelType.UINT32,
    "float32": PixelType.FLOAT,
    "float64": PixelType.DOUBLE,
}


class TiffOmeScene(NamedTuple):
    """Description of one TIFF-export scene, consumed by :func:`build_tiff_ome`."""

    name: str
    size_x: int
    size_y: int
    size_z: int
    size_t: int
    dtype: np.dtype
    pixel_sizes: PhysicalPixelSizes
    datetimes: Sequence[Optional[datetime]]
    channel_name: str
    title: Optional[str] = None


def _build_tiff_image(
    scene: TiffOmeScene, image_index: int, has_instrument: bool, has_experimenter: bool
) -> Image:
    """Build one OME ``Image`` for a TIFF-export scene."""
    planes: List[Plane] = []
    t0 = next((dt for dt in scene.datetimes if dt is not None), None)
    for t_idx in range(scene.size_t):
        dt = scene.datetimes[t_idx] if t_idx < len(scene.datetimes) else None
        plane_kwargs: Dict[str, Any] = {"the_t": t_idx, "the_z": 0, "the_c": 0}
        if dt is not None and t0 is not None:
            plane_kwargs["delta_t"] = (dt - t0).total_seconds()
            plane_kwargs["delta_t_unit"] = UnitsTime.SECOND
        planes.append(Plane(**plane_kwargs))

    px_kwargs: Dict[str, Any] = {
        "id": f"Pixels:{image_index}",
        "size_x": scene.size_x,
        "size_y": scene.size_y,
        "size_z": scene.size_z,
        "size_t": scene.size_t,
        "size_c": 1,
        "type": _NUMPY_TO_OME_PIXEL_TYPE.get(
            np.dtype(scene.dtype).name, PixelType.UINT16
        ),
        "dimension_order": Pixels_DimensionOrder.XYZCT,
        "channels": [
            Channel(
                id=f"Channel:{image_index}:0",
                name=scene.channel_name,
                samples_per_pixel=1,
            )
        ],
        "planes": planes,
    }
    for axis in ("x", "y", "z"):
        value = getattr(scene.pixel_sizes, axis.upper())
        if value is not None:
            px_kwargs[f"physical_size_{axis}"] = float(value)
            px_kwargs[f"physical_size_{axis}_unit"] = UnitsLength.MICROMETER
    if scene.size_t > 1 and t0 is not None:
        last = scene.datetimes[-1]
        if last is not None:
            px_kwargs["time_increment"] = (last - t0).total_seconds() / (
                scene.size_t - 1
            )
            px_kwargs["time_increment_unit"] = UnitsTime.SECOND

    img_kwargs: Dict[str, Any] = {
        "id": f"Image:{image_index}",
        "name": scene.title or scene.name,
        "pixels": Pixels(**px_kwargs),
    }
    if t0 is not None:
        img_kwargs["acquisition_date"] = t0
    if has_instrument:
        img_kwargs["instrument_ref"] = InstrumentRef(id="Instrument:0")
    if has_experimenter:
        img_kwargs["experimenter_ref"] = ExperimenterRef(id="Experimenter:0")
    return Image(**img_kwargs)


def build_tiff_ome(
    scenes: Sequence[TiffOmeScene],
    model: Optional[str] = None,
    software: Optional[str] = None,
    user: Optional[str] = None,
) -> OME:
    """Construct a multi-image OME for a set of Tomocube TIFF-export scenes.

    One :class:`ome_types.model.Image` is built per scene, in order, so that
    ``ome.images[i]`` corresponds to ``Reader.scenes[i]``.

    Parameters
    ----------
    scenes:
        Ordered scene descriptions.
    model:
        TIFF ``Model`` tag (e.g. ``"HTX"``); becomes the OME microscope model.
    software:
        TIFF ``Software`` tag; recorded on the microscope as its serial-free
        description via ``lot_number`` is not appropriate, so it is ignored
        unless *model* is missing, in which case it is used as the model.
    user:
        TIFF ``Artist`` tag; becomes the OME experimenter user name.
    """
    instrument: Optional[Instrument] = None
    microscope_model = model or software
    if microscope_model:
        instrument = Instrument(
            id="Instrument:0",
            microscope=Microscope(manufacturer="Tomocube", model=microscope_model),
        )

    experimenters: List[Experimenter] = []
    if user and user.strip():
        experimenters.append(Experimenter(id="Experimenter:0", user_name=user.strip()))

    images = [
        _build_tiff_image(
            scene,
            i,
            has_instrument=instrument is not None,
            has_experimenter=bool(experimenters),
        )
        for i, scene in enumerate(scenes)
    ]

    ome_kwargs: Dict[str, Any] = {"images": images}
    if instrument is not None:
        ome_kwargs["instruments"] = [instrument]
    if experimenters:
        ome_kwargs["experimenters"] = experimenters
    return OME(**ome_kwargs)
