"""HDF5 attribute helpers and OME metadata for TCF files."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from ome_types.model import (
    OME,
    Channel,
    Experimenter,
    ExperimenterRef,
    Image,
    Instrument,
    InstrumentRef,
    Objective,
    Pixels,
    Pixels_DimensionOrder,
    PixelType,
    Plane,
    UnitsLength,
    UnitsTime,
)

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _plain(value: Any) -> Any:
    """h5py attribute value to a plain Python scalar, string or list."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _plain(value.flat[0]) if value.size == 1 else [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def group_attrs(obj: Any) -> Dict[str, Any]:
    return {key: _plain(value) for key, value in obj.attrs.items()}


def attr(obj: Any, key: str) -> Any:
    value = obj.attrs.get(key)
    return None if value is None else _plain(value)


def _frame_times(f: h5py.File, scene: str, n_frames: int) -> List[Optional[datetime]]:
    times: List[Optional[datetime]] = []
    for i in range(n_frames):
        stamp = attr(f[f"Data/{scene}/{i:06d}"], "RecordingTime")
        try:
            times.append(datetime.strptime(stamp, _TIME_FORMAT) if stamp else None)
        except ValueError:
            times.append(None)
    return times


def _build_image(f: h5py.File, scene: str, index: int, refs: Dict[str, Any]) -> Image:
    modality = scene.split("/")[0]
    mod = group_attrs(f[f"Data/{modality}"])
    n_frames = int(mod["DataCount"])
    times = _frame_times(f, scene, n_frames)
    t0 = next((t for t in times if t is not None), None)

    planes = []
    for t_idx, stamp in enumerate(times):
        kwargs: Dict[str, Any] = {"the_t": t_idx, "the_z": 0, "the_c": 0}
        if stamp is not None and t0 is not None:
            kwargs["delta_t"] = (stamp - t0).total_seconds()
            kwargs["delta_t_unit"] = UnitsTime.SECOND
        planes.append(Plane(**kwargs))

    channel_kwargs: Dict[str, Any] = {
        "id": f"Channel:{index}:0",
        "samples_per_pixel": 1,
    }
    if "/" in scene:  # fluorescence channel group carries wavelengths in µm
        channel = group_attrs(f[f"Data/{scene}"])
        channel_kwargs["name"] = scene.split("/")[1]
        for key, field in (("Excitation", "excitation"), ("Emission", "emission")):
            if channel.get(key):
                channel_kwargs[f"{field}_wavelength"] = float(channel[key]) * 1000.0
                channel_kwargs[f"{field}_wavelength_unit"] = UnitsLength.NANOMETER

    pixels_kwargs: Dict[str, Any] = {
        "id": f"Pixels:{index}",
        "size_x": int(mod["SizeX"]),
        "size_y": int(mod["SizeY"]),
        "size_z": int(mod.get("SizeZ", 1)),
        "size_c": 1,
        "size_t": n_frames,
        "type": PixelType.FLOAT,
        "dimension_order": Pixels_DimensionOrder.XYZCT,
        "channels": [Channel(**channel_kwargs)],
        "planes": planes,
    }
    for axis in "xyz":
        value = mod.get(f"Resolution{axis.upper()}")
        if value is not None:
            pixels_kwargs[f"physical_size_{axis}"] = float(value)
            pixels_kwargs[f"physical_size_{axis}_unit"] = UnitsLength.MICROMETER

    image_kwargs: Dict[str, Any] = {
        "id": f"Image:{index}",
        "name": attr(f, "Title") or scene,
        "pixels": Pixels(**pixels_kwargs),
        **refs,
    }
    if t0 is not None:
        image_kwargs["acquisition_date"] = t0
    return Image(**image_kwargs)


def build_ome(f: h5py.File, scenes: Tuple[str, ...]) -> OME:
    """One OME ``Image`` per scene, in order, so ``ome.images[i]`` is ``scenes[i]``."""
    refs: Dict[str, Any] = {}
    ome_kwargs: Dict[str, Any] = {}

    # Info/Device "NA" is the condenser, not the objective, so only magnification.
    magnification = (
        attr(f["Info/Device"], "Magnification") if "Info/Device" in f else None
    )
    if magnification is not None:
        objective = Objective(
            id="Objective:0", nominal_magnification=float(magnification)
        )
        ome_kwargs["instruments"] = [
            Instrument(id="Instrument:0", objectives=[objective])
        ]
        refs["instrument_ref"] = InstrumentRef(id="Instrument:0")

    user = (attr(f, "UserID") or "").strip()
    if user:
        ome_kwargs["experimenters"] = [
            Experimenter(id="Experimenter:0", user_name=user)
        ]
        refs["experimenter_ref"] = ExperimenterRef(id="Experimenter:0")

    ome_kwargs["images"] = [_build_image(f, s, i, refs) for i, s in enumerate(scenes)]
    return OME(**ome_kwargs)
