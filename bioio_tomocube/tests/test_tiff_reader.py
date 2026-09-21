#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the HTX TIFF-export backend.

Fixtures under ``resources/tiff`` are the real export files from a 5-timepoint
acquisition cropped to 32x32 pixels with every TIFF tag preserved, including
the FL3D ImageJ header that claims ``slices=12`` while the file holds 70 pages.
Only TP01 and TP05 are included; TP05 additionally has FL2D max projections.
"""

from datetime import datetime, timedelta

import dask.array as da
import fsspec
import numpy as np
import pytest
import tifffile
from bioio import BioImage
from bioio_base import exceptions, test_utilities
from ome_types.model import OME

from bioio_tomocube import Reader, TiffReader
from bioio_tomocube.tiff_reader import (
    RI_SCALE,
    discover_acquisition,
    parse_filename,
)

from .conftest import LOCAL_RESOURCES_DIR

TIFF_DIR = LOCAL_RESOURCES_DIR / "tiff"
BASE = "251003.112316.6 Well Mito.018.Group1.A1"

HT_TP01 = TIFF_DIR / f"{BASE}.TP01_HT3D_0.00.TIFF"
HT_TP05 = TIFF_DIR / f"{BASE}.TP05_HT3D_0.00.TIFF"
FL3D_CH0_TP01 = TIFF_DIR / f"{BASE}.TP01" / f"{BASE}.TP01_FL3D_CH0_0.00.TIFF"
FL3D_CH1_TP05 = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL3D_CH1_0.00.TIFF"
FL2D_CH0_TP05 = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL2D_CH0_0.00.TIFF"

PX_XY = 0.1625607173804458  # 128 / 7873981 cm
PX_Z_HT = 0.87449127  # ImageJ spacing 0.000087449127 cm
PX_Z_FL_HEADER = 1.04453123  # what the FL3D ImageJ header (wrongly) claims

CHANNELS_3D = ["HT", "FL_CH0", "FL_CH1"]
CHANNELS_2D = ["FL_CH0", "FL_CH1"]  # no HT2D export in this dataset
TP01_SCENES = ("3D",)
TP05_SCENES = ("3D", "2DMIP")


# ---------------------------------------------------------------------------
# File-name parsing and discovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            f"{BASE}.TP01_HT3D_0.00.TIFF",
            (BASE, 1, "HT3D", None, "0.00", "3D", "HT"),
        ),
        (
            f"{BASE}.TP05_FL3D_CH1_0.00.TIFF",
            (BASE, 5, "FL3D", 1, "0.00", "3D", "FL_CH1"),
        ),
        (
            f"{BASE}.TP05_FL2D_CH0_0.00.TIFF",
            (BASE, 5, "FL2D", 0, "0.00", "2DMIP", "FL_CH0"),
        ),
        (
            "some_experiment_HT3D_0.00.tif",
            ("some_experiment", None, "HT3D", None, "0.00", "3D", "HT"),
        ),
        (
            "exp.001.A1.TP12_HT2D_1.50.TIFF",
            ("exp.001.A1", 12, "HT2D", None, "1.50", "2DMIP", "HT"),
        ),
    ],
)
def test_parse_filename(name, expected):
    info = parse_filename(f"/some/dir/{name}")
    assert info is not None
    assert (
        info.base,
        info.timepoint,
        info.modality,
        info.channel,
        info.suffix,
        info.scene,
        info.channel_name,
    ) == expected


@pytest.mark.parametrize(
    "name",
    ["image.tiff", "HT3D.TIFF", f"{BASE}.TP01_BF_0.00.TIFF", "x_FL3D_CH0.TIFF"],
)
def test_parse_filename_rejects(name):
    assert parse_filename(name) is None


def _summary(files):
    return [(f.scene, f.timepoint, f.channel_name) for f in files]


def test_discover_single_timepoint_from_ht():
    fs = fsspec.filesystem("file")
    found = discover_acquisition(fs, str(HT_TP01))
    assert _summary(found) == [
        ("3D", 1, "HT"),
        ("3D", 1, "FL_CH0"),
        ("3D", 1, "FL_CH1"),
    ]


def test_discover_single_timepoint_from_fl():
    """Anchoring on an FL file inside the TP folder must still find HT3D."""
    fs = fsspec.filesystem("file")
    found = discover_acquisition(fs, str(FL3D_CH1_TP05))
    assert _summary(found) == [
        ("3D", 5, "HT"),
        ("3D", 5, "FL_CH0"),
        ("3D", 5, "FL_CH1"),
        ("2DMIP", 5, "FL_CH0"),
        ("2DMIP", 5, "FL_CH1"),
    ]
    assert found[0].path.endswith(f"{BASE}.TP05_HT3D_0.00.TIFF")


def test_discover_timelapse():
    fs = fsspec.filesystem("file")
    found = discover_acquisition(fs, str(HT_TP01), timelapse=True)
    assert _summary(found) == [
        ("3D", 1, "HT"),
        ("3D", 1, "FL_CH0"),
        ("3D", 1, "FL_CH1"),
        ("3D", 5, "HT"),
        ("3D", 5, "FL_CH0"),
        ("3D", 5, "FL_CH1"),
        ("2DMIP", 5, "FL_CH0"),
        ("2DMIP", 5, "FL_CH1"),
    ]


# ---------------------------------------------------------------------------
# Core reader checks — shape, dtype, dims, channels, pixel sizes, metadata type
# ---------------------------------------------------------------------------

_READER_PARAMS = [
    (
        HT_TP01,
        "3D",
        TP01_SCENES,
        (1, 3, 70, 32, 32),
        "TCZYX",
        CHANNELS_3D,
        (PX_Z_HT, PX_XY, PX_XY),
    ),
    (
        FL3D_CH1_TP05,
        "3D",
        TP05_SCENES,
        (1, 3, 70, 32, 32),
        "TCZYX",
        CHANNELS_3D,
        (PX_Z_HT, PX_XY, PX_XY),
    ),
    (
        FL2D_CH0_TP05,
        "2DMIP",
        TP05_SCENES,
        (1, 2, 32, 32),
        "TCYX",
        CHANNELS_2D,
        (None, PX_XY, PX_XY),
    ),
    (
        HT_TP05,
        "2DMIP",
        TP05_SCENES,
        (1, 2, 32, 32),
        "TCYX",
        CHANNELS_2D,
        (None, PX_XY, PX_XY),
    ),
]


@pytest.mark.parametrize(
    "filename, set_scene, expected_scenes, expected_shape, expected_dims_order, "
    "expected_channel_names, expected_physical_pixel_sizes",
    _READER_PARAMS,
)
def test_reader(
    filename,
    set_scene,
    expected_scenes,
    expected_shape,
    expected_dims_order,
    expected_channel_names,
    expected_physical_pixel_sizes,
):
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=filename,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.dtype(np.uint16),
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
        reader_kwargs={},
    )


def test_reader_dispatch_returns_tiff_backend():
    rdr = Reader(HT_TP01)
    assert isinstance(rdr, TiffReader)
    assert isinstance(rdr, Reader)


@pytest.mark.parametrize(
    "filename, expected_scene",
    [
        (HT_TP01, "3D"),
        (FL3D_CH0_TP01, "3D"),
        (FL3D_CH1_TP05, "3D"),
        (HT_TP05, "3D"),
        (FL2D_CH0_TP05, "2DMIP"),
    ],
)
def test_default_scene_matches_anchor_file(filename, expected_scene):
    """The default scene is the one containing the file that was passed in."""
    rdr = Reader(filename)
    assert rdr.current_scene == expected_scene


def test_multi_scene():
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=HT_TP05,
        first_scene_id="3D",
        first_scene_shape=(1, 3, 70, 32, 32),
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id="2DMIP",
        second_scene_shape=(1, 2, 32, 32),
        second_scene_dtype=np.dtype(np.uint16),
        allow_same_scene_data=False,
        reader_kwargs={},
    )


def test_unsupported_tiff():
    """A .tif that is not a Tomocube export must raise UnsupportedFileFormatError."""
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(LOCAL_RESOURCES_DIR / "unsupported.tif")


def test_non_tomocube_tiff_rejected(tmp_path):
    """A valid TIFF whose name matches but lacks HTX tags must be rejected."""
    path = tmp_path / "fake.TP01_HT3D_0.00.TIFF"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(path)


# ---------------------------------------------------------------------------
# Pixel data — channel stacking, the FL3D header trap, RI scaling, laziness
# ---------------------------------------------------------------------------


def _tiff_pages(path):
    with tifffile.TiffFile(path) as tif:
        return np.stack([p.asarray() for p in tif.pages])


def test_channels_are_stacked_from_separate_files():
    """C index i of the 3D scene must equal the i-th modality's file."""
    rdr = Reader(HT_TP01)
    data = rdr.data
    assert data.shape == (1, 3, 70, 32, 32)
    assert rdr.channel_names == CHANNELS_3D
    np.testing.assert_array_equal(data[0, 0], _tiff_pages(HT_TP01))
    np.testing.assert_array_equal(data[0, 1], _tiff_pages(FL3D_CH0_TP01))
    np.testing.assert_array_equal(
        data[0, 2],
        _tiff_pages(TIFF_DIR / f"{BASE}.TP01" / f"{BASE}.TP01_FL3D_CH1_0.00.TIFF"),
    )
    np.testing.assert_array_equal(rdr.dask_data.compute(), data)
    assert list(rdr.xarray_dask_data.coords["C"].values) == CHANNELS_3D


def test_fl3d_reads_all_pages_not_imagej_slices():
    """FL channels must yield all 70 stored pages even though the header says 12."""
    with tifffile.TiffFile(FL3D_CH0_TP01) as tif:
        assert tif.imagej_metadata["slices"] == 12
        assert len(tif.pages) == 70

    fl0 = Reader(FL3D_CH0_TP01).data[0, 1]
    assert fl0.shape == (70, 32, 32)
    nonzero = [z for z in range(70) if fl0[z].any()]
    assert nonzero[0] == 30 and nonzero[-1] == 43 and len(nonzero) == 14


def test_z_spacing_comes_from_ht_channel():
    """FL pages are on the HT grid, so Z spacing is the HT header's value."""
    rdr = Reader(FL3D_CH0_TP01)
    assert rdr.physical_pixel_sizes.Z == pytest.approx(PX_Z_HT)
    meta = rdr.tiff_metadata
    assert meta["z_spacing_source"] == "HT header"
    assert meta["imagej"]["HT"]["spacing"] * 1e4 == pytest.approx(PX_Z_HT)
    assert meta["imagej"]["FL_CH0"]["spacing"] * 1e4 == pytest.approx(PX_Z_FL_HEADER)


def test_ht_raw_values_are_ri_times_1e4():
    rdr = Reader(HT_TP01)
    ht = rdr.data[0, 0]
    assert ht.dtype == np.uint16
    assert 12_000 < ht.min() and ht.max() < 16_000


def test_refractive_index_option():
    raw = Reader(HT_TP01).data
    rdr = Reader(HT_TP01, refractive_index=True)
    assert rdr.dtype == np.float32
    data = rdr.data
    assert data.dtype == np.float32
    # HT channel is scaled, FL channels are only cast
    np.testing.assert_allclose(data[0, 0], raw[0, 0].astype(np.float32) * RI_SCALE)
    np.testing.assert_array_equal(data[0, 1:], raw[0, 1:].astype(np.float32))
    assert 1.30 < float(data[0, 0].min()) < float(data[0, 0].max()) < 1.45
    assert rdr.tiff_metadata["refractive_index_scale"] == {
        "HT": RI_SCALE,
        "FL_CH0": None,
        "FL_CH1": None,
    }
    assert rdr.ome_metadata.images[0].pixels.type.value == "float"

    # a scene without an HT channel is untouched
    rdr = Reader(FL2D_CH0_TP05, refractive_index=True)
    assert rdr.dtype == np.uint16


def test_dask_backed_and_lazy():
    rdr = Reader(HT_TP01)
    lazy = rdr.xarray_dask_data
    assert isinstance(lazy.data, da.Array)
    assert lazy.dims == ("T", "C", "Z", "Y", "X")
    assert lazy.data.chunks[1] == (1, 1, 1)  # one chunk per channel
    assert lazy.data.chunks[2] == (1,) * 70  # one chunk per Z plane
    assert lazy[0, 2, 35].compute().shape == (32, 32)


# ---------------------------------------------------------------------------
# Timelapse
# ---------------------------------------------------------------------------


def test_timelapse_shapes_and_order():
    rdr = Reader(HT_TP01, timelapse=True)
    assert rdr.scenes == TP05_SCENES
    assert rdr.shape == (2, 3, 70, 32, 32)
    assert rdr.tiff_metadata["timepoints"] == [1, 5]
    assert [f.timepoint for f in rdr.files["3D"]["HT"]] == [1, 5]

    # T=0 must be TP01 and T=1 must be TP05, for every channel
    np.testing.assert_array_equal(rdr.data[0], Reader(HT_TP01).data[0])
    np.testing.assert_array_equal(rdr.data[1], Reader(HT_TP05).data[0])

    # the MIP scene only exists at TP05, so it keeps T=1
    rdr.set_scene("2DMIP")
    assert rdr.shape == (1, 2, 32, 32)
    assert rdr.tiff_metadata["timepoints"] == [5]
    assert rdr.time_interval is None


def test_timelapse_time_interval_and_planes():
    rdr = Reader(HT_TP01, timelapse=True)
    # TP01 12:36:59 -> TP05 12:42:53 is 354 s; only two timepoints are fixtures
    assert rdr.time_interval == timedelta(seconds=354)
    planes = rdr.ome_metadata.images[0].pixels.planes
    assert [(p.the_t, p.the_c, p.delta_t) for p in planes] == [
        (0, 0, 0.0),
        (0, 1, 0.0),
        (0, 2, 0.0),
        (1, 0, 354.0),
        (1, 1, 354.0),
        (1, 2, 354.0),
    ]
    sm = rdr.standard_metadata.to_dict()
    assert sm["Timelapse"] is True
    assert sm["Timelapse Interval"] == timedelta(seconds=354)


def test_incomplete_timepoint_is_dropped(tmp_path, caplog):
    """A timepoint missing one channel is dropped with a warning, not zero-filled."""
    # Build TP01 (all channels) and TP05 with only HT + FL_CH0.
    root = tmp_path / "acq"
    (root / f"{BASE}.TP01").mkdir(parents=True)
    (root / f"{BASE}.TP05").mkdir(parents=True)
    for src in [HT_TP01, HT_TP05]:
        (root / src.name).write_bytes(src.read_bytes())
    for src in (TIFF_DIR / f"{BASE}.TP01").iterdir():
        (root / f"{BASE}.TP01" / src.name).write_bytes(src.read_bytes())
    src = TIFF_DIR / f"{BASE}.TP05" / f"{BASE}.TP05_FL3D_CH0_0.00.TIFF"
    (root / f"{BASE}.TP05" / src.name).write_bytes(src.read_bytes())

    with caplog.at_level("WARNING", logger="bioio_tomocube.tiff_reader"):
        rdr = Reader(root / HT_TP01.name, timelapse=True)
        assert rdr.scenes == ("3D",)
        assert rdr.shape == (1, 3, 70, 32, 32)
        assert rdr.tiff_metadata["timepoints"] == [1]
    assert any("dropping timepoint(s) [5]" in r.message for r in caplog.records)


def test_mismatched_grids_fall_back_to_per_channel_scenes(tmp_path, caplog):
    """Channels on different pixel grids are exposed as separate scenes."""
    root = tmp_path / "acq"
    (root / f"{BASE}.TP01").mkdir(parents=True)
    (root / HT_TP01.name).write_bytes(HT_TP01.read_bytes())
    # FL_CH0 on a different grid (fewer pages), FL_CH1 unchanged
    with tifffile.TiffFile(FL3D_CH0_TP01) as tif:
        pages = [p.asarray() for p in tif.pages[:10]]
    tifffile.imwrite(
        root / f"{BASE}.TP01" / FL3D_CH0_TP01.name,
        np.stack(pages),
        extratags=[(272, "s", 0, "HTX", True)],
    )
    ch1 = TIFF_DIR / f"{BASE}.TP01" / f"{BASE}.TP01_FL3D_CH1_0.00.TIFF"
    (root / f"{BASE}.TP01" / ch1.name).write_bytes(ch1.read_bytes())

    with caplog.at_level("WARNING", logger="bioio_tomocube.tiff_reader"):
        rdr = Reader(root / HT_TP01.name)
    assert rdr.scenes == ("3D/HT", "3D/FL_CH0", "3D/FL_CH1")
    assert rdr.current_scene == "3D/HT"
    assert rdr.shape == (1, 1, 70, 32, 32)
    rdr.set_scene("3D/FL_CH0")
    assert rdr.shape == (1, 1, 10, 32, 32)
    assert rdr.channel_names == ["FL_CH0"]
    assert any("not on a common pixel grid" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_ome_metadata_structure():
    rdr = Reader(HT_TP05)
    ome = rdr.ome_metadata
    assert isinstance(ome, OME)
    assert len(ome.images) == len(rdr.scenes) == 2
    img3d, img2d = ome.images
    assert img3d.pixels.size_c == 3
    assert [c.name for c in img3d.pixels.channels] == CHANNELS_3D
    assert [c.id for c in img3d.pixels.channels] == [
        "Channel:0:0",
        "Channel:0:1",
        "Channel:0:2",
    ]
    assert img2d.pixels.size_c == 2
    assert [c.name for c in img2d.pixels.channels] == CHANNELS_2D
    assert img2d.pixels.size_z == 1 and img2d.pixels.physical_size_z is None
    assert ome.instruments[0].microscope.model == "HTX"
    assert ome.instruments[0].microscope.manufacturer == "Tomocube"
    assert ome.experimenters[0].user_name == "Default"
    assert img3d.acquisition_date == datetime(2025, 10, 3, 12, 42, 53)
    assert img3d.pixels.physical_size_x == pytest.approx(PX_XY)
    assert img3d.pixels.physical_size_z == pytest.approx(PX_Z_HT)


def test_tiff_metadata_contents():
    rdr = Reader(HT_TP01)
    meta = rdr.tiff_metadata
    assert meta["format"] == "tomocube-tiff"
    assert meta["scene"] == "3D"
    assert meta["channels"] == CHANNELS_3D
    assert meta["timepoints"] == [1]
    assert meta["files"]["HT"] == [str(HT_TP01)]
    assert meta["files"]["FL_CH0"] == [str(FL3D_CH0_TP01)]
    assert meta["tags"]["HT"]["Model"] == "HTX"
    assert meta["tags"]["HT"]["Software"] == "HTX ProcessingServer 2.1.24"
    assert meta["tags"]["HT"]["XResolution"] == (7873981, 128)
    assert meta["imagej"]["HT"]["slices"] == 70
    assert meta["imagej"]["FL_CH0"]["slices"] == 12
    assert meta["z_spacing_source"] == "HT header"
    acq = meta["acquisition"]
    assert acq["base"] == BASE
    assert acq["started"] == datetime(2025, 10, 3, 11, 23, 16)
    assert acq["experiment"] == "6 Well Mito"
    assert acq["index"] == 18
    assert acq["group"] == "Group1"
    assert acq["well"] == "A1"
    assert acq["modalities"] == {"HT": "HT3D", "FL_CH0": "FL3D", "FL_CH1": "FL3D"}


def test_standard_metadata():
    rdr = Reader(FL3D_CH0_TP01)
    sm = rdr.standard_metadata.to_dict()
    expected = {
        "Dimensions Present": "TCZYX",
        "Image Size T": 1,
        "Image Size C": 3,
        "Image Size X": 32,
        "Image Size Y": 32,
        "Image Size Z": 70,
        "Imaging Datetime": datetime(2025, 10, 3, 12, 36, 59),
        "Imaged By": "Default",
        "Pixel Size X": PX_XY,
        "Pixel Size Y": PX_XY,
        "Pixel Size Z": PX_Z_HT,
        "Row": "A",
        "Column": 1,
        "Timelapse": False,
        "Timelapse Interval": None,
    }
    for key, value in expected.items():
        if isinstance(value, float):
            assert sm[key] == pytest.approx(value), key
        else:
            assert sm[key] == value, key


# ---------------------------------------------------------------------------
# bioio integration
# ---------------------------------------------------------------------------


def test_bioimage_auto_selects_tomocube_plugin():
    img = BioImage(HT_TP01)
    assert isinstance(img.reader, TiffReader)
    assert img.dims.order == "TCZYX"
    assert img.shape == (1, 3, 70, 32, 32)
    assert img.channel_names == CHANNELS_3D
    assert img.physical_pixel_sizes.Z == pytest.approx(PX_Z_HT)
    np.testing.assert_array_equal(
        img.get_image_data("ZYX", T=0, C=1), _tiff_pages(FL3D_CH0_TP01)
    )


def test_bioimage_forwards_backend_kwargs():
    img = BioImage(HT_TP01, timelapse=True, refractive_index=True)
    assert img.shape == (2, 3, 70, 32, 32)
    assert img.dtype == np.float32


# ---------------------------------------------------------------------------
# Remote (fsspec) read via the in-memory filesystem
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mem_fs():
    """Mirror the TP05 acquisition (HT3D + TP05 folder) into memory://."""
    mem = fsspec.filesystem("memory")
    root = "/tomocube"
    paths = []
    for local in [HT_TP05, *sorted((TIFF_DIR / f"{BASE}.TP05").iterdir())]:
        rel = local.relative_to(TIFF_DIR).as_posix()
        target = f"{root}/{rel}"
        mem.pipe(target, local.read_bytes())
        paths.append(target)
    yield mem, f"memory://{root}/{HT_TP05.name}"
    for p in paths:
        mem.rm(p)


def test_remote_scenes_and_shapes(mem_fs):
    _, url = mem_fs
    rdr = Reader(url)
    assert isinstance(rdr, TiffReader)
    assert rdr.scenes == TP05_SCENES
    assert rdr.shape == (1, 3, 70, 32, 32)
    assert rdr.channel_names == CHANNELS_3D
    assert rdr.physical_pixel_sizes.Z == pytest.approx(PX_Z_HT)
    assert isinstance(rdr.xarray_dask_data.data, da.Array)
    frame = rdr.xarray_dask_data[0, 2, 40].compute()
    np.testing.assert_array_equal(frame, _tiff_pages(FL3D_CH1_TP05)[40])
