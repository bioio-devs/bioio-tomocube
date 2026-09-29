"""Crop real TCF acquisitions into small test fixtures.

Reads the source files below from the HT-X1 Plus demo tree on VAST, keeps the
first few frames of each modality and a 64 x 64 window at the image centre,
updates the size attributes, and writes the result to ``tests/resources``.
Thumbnails are dropped; everything else (attrs, ``Info``) is copied verbatim.

    python scripts/make_fixtures.py
"""

import pathlib

import h5py

VAST = pathlib.Path(
    "/allen/aics/microscopes/Jie/Microscope demo/2025 Tomocube HT-X1 Plus/Data"
)
if not VAST.exists():  # macOS mount
    VAST = pathlib.Path.home() / "allen" / VAST.relative_to("/allen")
OUT = pathlib.Path(__file__).parent.parent / "bioio_tomocube" / "tests" / "resources"
CROP = 64

# (source, frames to keep per Data/ group)
SOURCES = [
    (  # snapshot-style capture: one frame, HT + two FL channels of mixed dtype
        "6 Well Mito/251003.112316.6 Well Mito.018.Group1.A1.TP01/"
        "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF",
        {"3D": 1, "2DMIP": 1, "3DFL": 1, "2DFLMIP": 1},
    ),
    (  # timelapse: 16 frames at 60 s in one file; keep three
        "6 Well Mito/251003.104445.6 Well Mito.003.Group1.A1.T001P01/"
        "251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF",
        {"3D": 3, "2DMIP": 3, "3DFL": 3, "2DFLMIP": 3},
    ),
    (  # FL acquired every third HT frame: modalities have different T counts
        "myosin 24well plate/251006.162843.myosin 24well plate.005.Group8.C5.T001P02/"
        "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF",
        {"3D": 4, "2DMIP": 4, "3DFL": 2, "2DFLMIP": 2},
    ),
]


def _copy_frames(src: h5py.Group, dst: h5py.Group, n_frames: int) -> None:
    for name in sorted(src)[:n_frames]:
        data = src[name]
        y0 = (data.shape[-2] - CROP) // 2
        x0 = (data.shape[-1] - CROP) // 2
        out = dst.create_dataset(name, data=data[..., y0 : y0 + CROP, x0 : x0 + CROP])
        out.attrs.update(data.attrs)


def crop(source: pathlib.Path, frames: dict, target: pathlib.Path) -> None:
    with h5py.File(source, "r") as f, h5py.File(target, "w") as g:
        g.attrs.update(f.attrs)
        f.copy("Info", g)
        data = g.create_group("Data")
        for modality, n_frames in frames.items():
            src = f["Data"][modality]
            dst = data.create_group(modality)
            dst.attrs.update(src.attrs)
            dst.attrs["SizeX"] = dst.attrs["SizeY"] = CROP
            dst.attrs["DataCount"] = n_frames
            first = next(iter(src.values()))
            if isinstance(first, h5py.Group):  # 3DFL/CHn/<frame>
                for ch_name, ch in src.items():
                    ch_out = dst.create_group(ch_name)
                    ch_out.attrs.update(ch.attrs)
                    _copy_frames(ch, ch_out, n_frames)
            else:
                _copy_frames(src, dst, n_frames)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, frames in SOURCES:
        source = VAST / rel
        target = OUT / source.name
        crop(source, frames, target)
        print(f"{target.name}: {target.stat().st_size / 2**20:.1f} MiB")
