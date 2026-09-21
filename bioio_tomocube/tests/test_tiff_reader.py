#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the HTX TIFF-export backend.

Fixtures under ``resources/tiff`` are the real export files from a 5-timepoint
acquisition cropped to 32x32 pixels with every TIFF tag preserved, including
the FL3D ImageJ header that claims ``slices=12`` while the file holds 70 pages.
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

TP01_SCENES = ("3D", "3DFL/CH0", "3DFL/CH1")
TP05_SCENES = ("3D", "3DFL/CH0", "3DFL/CH1", "2DFLMIP/CH0", "2DFLMIP/CH1")


# ---------------------------------------------------------------------------
# File-name parsing and discovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            f"{BASE}.TP01_HT3D_0.00.TIFF",
            (BASE, 1, "HT3D", None, "0.00", "3D"),
        ),
        (
            f"{BASE}.TP05_FL3D_CH1_0.00.TIFF",
            (BASE, 5, "FL3D", 1, "0.00", "3DFL/CH1"),
        ),
        (
            f"{BASE}.TP05_FL2D_CH0_0.00.TIFF",
            (BASE, 5, "FL2D", 0, "0.00", "2DFLMIP/CH0"),
        ),
        (
            "some_experiment_HT3D_0.00.tif",
            ("some_experiment", None, "HT3D", None, "0.00", "3D"),
        ),
        (
            "exp.001.A1.TP12_HT2D_1.50.TIFF",
            ("exp.001.A1", 12, "HT2D", None, "1.50", "2DMIP"),
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
    ) == expected


@pytest.mark.parametrize(
    "name",
    ["image.tiff", "HT3D.TIFF", f"{BASE}.TP01_BF_0.00.TIFF", "x_FL3D_CH0.TIFF"],
)
def test_parse_filename_rejects(name):
    assert parse_filename(name) is None


def test_discover_single_timepoint_from_ht():
    fs = fsspec.filesystem("file")
    found = discover_acquisition(fs, str(HT_TP01))
    assert tuple(found) == TP01_SCENES
    assert all(len(v) == 1 for v in found.values())
    assert all(v[0].timepoint == 1 for v in found.values())


def test_discover_single_timepoint_from_fl():
    """Anchoring on an FL file inside the TP folder must still find HT3D."""
    fs = fsspec.filesystem("file")
    found = discover_acquisition(fs, str(FL3D_CH1_TP05))
    assert tuple(found) == TP05_SCENES
    assert found["3D"][0].path.endswith(f"{BASE}.TP05_HT3D_0.00.TIFF")


def test_discover_timelapse():
    fs = fsspec.filesystem("file")
    found = discover_acquisition(fs, str(HT_TP01), timelapse=True)
    assert tuple(found) == TP05_SCENES
    assert [f.timepoint for f in found["3D"]] == [1, 5]
    assert [f.timepoint for f in found["3DFL/CH0"]] == [1, 5]
    assert [f.timepoint for f in found["2DFLMIP/CH0"]] == [5]


# ---------------------------------------------------------------------------
# Core reader checks — shape, dtype, dims, pixel sizes, metadata type
# ---------------------------------------------------------------------------

_READER_PARAMS = [
    (HT_TP01, "3D", TP01_SCENES, (1, 70, 32, 32), "TZYX", (PX_Z_HT, PX_XY, PX_XY)),
    (
        HT_TP01,
        "3DFL/CH0",
        TP01_SCENES,
        (1, 70, 32, 32),
        "TZYX",
        (PX_Z_HT, PX_XY, PX_XY),
    ),
    (
        FL3D_CH1_TP05,
        "3DFL/CH1",
        TP05_SCENES,
        (1, 70, 32, 32),
        "TZYX",
        (PX_Z_HT, PX_XY, PX_XY),
    ),
    (
        FL2D_CH0_TP05,
        "2DFLMIP/CH0",
        TP05_SCENES,
        (1, 32, 32),
        "TYX",
        (None, PX_XY, PX_XY),
    ),
    (
        HT_TP05,
        "2DFLMIP/CH1",
        TP05_SCENES,
        (1, 32, 32),
        "TYX",
        (None, PX_XY, PX_XY),
    ),
]


@pytest.mark.parametrize(
    "filename, set_scene, expected_scenes, expected_shape, expected_dims_order, "
    "expected_physical_pixel_sizes",
    _READER_PARAMS,
)
def test_reader(
    filename,
    set_scene,
    expected_scenes,
    expected_shape,
    expected_dims_order,
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
        expected_channel_names=None,
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
        (FL3D_CH0_TP01, "3DFL/CH0"),
        (FL3D_CH1_TP05, "3DFL/CH1"),
        (FL2D_CH0_TP05, "2DFLMIP/CH0"),
    ],
)
def test_default_scene_matches_anchor_file(filename, expected_scene):
    """The default scene is the modality of the file that was passed in."""
    rdr = Reader(filename)
    assert rdr.current_scene == expected_scene


def test_multi_scene():
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=HT_TP05,
        first_scene_id="3D",
        first_scene_shape=(1, 70, 32, 32),
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id="2DFLMIP/CH0",
        second_scene_shape=(1, 32, 32),
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
# Pixel data — the FL3D header trap, RI scaling, delayed == immediate
# ---------------------------------------------------------------------------


def test_fl3d_reads_all_pages_not_imagej_slices():
    """FL3D must yield all 70 stored pages even though the header says 12."""
    with tifffile.TiffFile(FL3D_CH0_TP01) as tif:
        assert tif.imagej_metadata["slices"] == 12
        assert len(tif.pages) == 70

    rdr = Reader(FL3D_CH0_TP01)
    data = rdr.data
    assert data.shape == (1, 70, 32, 32)
    nonzero = [z for z in range(70) if data[0, z].any()]
    assert nonzero[0] == 30 and nonzero[-1] == 43 and len(nonzero) == 14


def test_fl3d_uses_ht_z_spacing():
    """FL3D pages are on the HT grid, so Z spacing comes from the HT3D sibling."""
    rdr = Reader(FL3D_CH0_TP01)
    assert rdr.physical_pixel_sizes.Z == pytest.approx(PX_Z_HT)
    meta = rdr.tiff_metadata
    assert meta["z_spacing_source"] == "HT3D sibling"
    assert meta["imagej"]["spacing"] * 1e4 == pytest.approx(PX_Z_FL_HEADER)


def test_pixel_values_match_tifffile():
    """Fast contiguous-strip path must return exactly what tifffile decodes."""
    rdr = Reader(HT_TP01)
    expected = np.stack([p.asarray() for p in tifffile.TiffFile(HT_TP01).pages])
    np.testing.assert_array_equal(rdr.data[0], expected)
    np.testing.assert_array_equal(rdr.dask_data[0].compute(), expected)


def test_ht_raw_values_are_ri_times_1e4():
    rdr = Reader(HT_TP01)
    data = rdr.data
    assert data.dtype == np.uint16
    assert 12_000 < data.min() and data.max() < 16_000


def test_refractive_index_option():
    raw = Reader(HT_TP01).data
    rdr = Reader(HT_TP01, refractive_index=True)
    assert rdr.dtype == np.float32
    ri = rdr.data
    assert ri.dtype == np.float32
    np.testing.assert_allclose(ri, raw.astype(np.float32) * RI_SCALE, rtol=1e-6)
    assert 1.30 < float(ri.min()) < float(ri.max()) < 1.45
    assert rdr.tiff_metadata["refractive_index_scale"] == RI_SCALE
    assert rdr.ome_metadata.images[0].pixels.type.value == "float"

    # FL scenes are never scaled
    rdr.set_scene("3DFL/CH0")
    assert rdr.dtype == np.uint16
    assert rdr.tiff_metadata["refractive_index_scale"] is None


def test_dask_backed_and_lazy():
    rdr = Reader(HT_TP01)
    lazy = rdr.xarray_dask_data
    assert isinstance(lazy.data, da.Array)
    assert lazy.dims == ("T", "Z", "Y", "X")
    # one chunk per Z plane
    assert lazy.data.chunks[1] == (1,) * 70
    assert lazy[0, 35].compute().shape == (32, 32)


# ---------------------------------------------------------------------------
# Timelapse
# ---------------------------------------------------------------------------


def test_timelapse_shapes_and_order():
    rdr = Reader(HT_TP01, timelapse=True)
    assert rdr.scenes == TP05_SCENES
    rdr.set_scene("3D")
    assert rdr.shape == (2, 70, 32, 32)
    assert [f.timepoint for f in rdr.files["3D"]] == [1, 5]
    assert rdr.tiff_metadata["timepoints"] == [1, 5]

    # T=0 must be TP01 and T=1 must be TP05
    np.testing.assert_array_equal(rdr.data[0], Reader(HT_TP01).data[0])
    np.testing.assert_array_equal(rdr.data[1], Reader(HT_TP05).data[0])

    # scenes present in only one timepoint keep T=1
    rdr.set_scene("2DFLMIP/CH0")
    assert rdr.shape == (1, 32, 32)
    assert rdr.time_interval is None


def test_timelapse_time_interval_and_planes():
    rdr = Reader(HT_TP01, timelapse=True)
    rdr.set_scene("3D")
    # TP01 12:36:59 -> TP05 12:42:53 is 354 s across 4 intervals, but only two
    # of the five timepoints are present as fixtures, so the mean is 354 s.
    assert rdr.time_interval == timedelta(seconds=354)
    planes = rdr.ome_metadata.images[0].pixels.planes
    assert [p.delta_t for p in planes] == [0.0, 354.0]
    sm = rdr.standard_metadata.to_dict()
    assert sm["Timelapse"] is True
    assert sm["Timelapse Interval"] == timedelta(seconds=354)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_ome_metadata_structure():
    rdr = Reader(HT_TP05)
    ome = rdr.ome_metadata
    assert isinstance(ome, OME)
    assert len(ome.images) == len(rdr.scenes)
    assert [img.pixels.channels[0].name for img in ome.images] == [
        "HT",
        "FL CH0",
        "FL CH1",
        "FL CH0",
        "FL CH1",
    ]
    assert ome.instruments[0].microscope.model == "HTX"
    assert ome.instruments[0].microscope.manufacturer == "Tomocube"
    assert ome.experimenters[0].user_name == "Default"
    img = ome.images[0]
    assert img.acquisition_date == datetime(2025, 10, 3, 12, 42, 53)
    assert img.pixels.physical_size_x == pytest.approx(PX_XY)
    assert img.pixels.physical_size_z == pytest.approx(PX_Z_HT)
    assert ome.images[3].pixels.physical_size_z is None


def test_tiff_metadata_contents():
    rdr = Reader(HT_TP01)
    meta = rdr.tiff_metadata
    assert meta["format"] == "tomocube-tiff"
    assert meta["scene"] == "3D"
    assert meta["files"] == [str(HT_TP01)]
    assert meta["tags"]["Model"] == "HTX"
    assert meta["tags"]["Software"] == "HTX ProcessingServer 2.1.24"
    assert meta["tags"]["XResolution"] == (7873981, 128)
    assert meta["imagej"]["slices"] == 70
    assert meta["z_spacing_source"] == "ImageJ header"
    acq = meta["acquisition"]
    assert acq["base"] == BASE
    assert acq["started"] == datetime(2025, 10, 3, 11, 23, 16)
    assert acq["experiment"] == "6 Well Mito"
    assert acq["index"] == 18
    assert acq["group"] == "Group1"
    assert acq["well"] == "A1"
    assert acq["modality"] == "HT3D"
    assert acq["channel"] is None


def test_standard_metadata():
    rdr = Reader(FL3D_CH0_TP01)
    sm = rdr.standard_metadata.to_dict()
    expected = {
        "Dimensions Present": "TZYX",
        "Image Size T": 1,
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
    assert img.shape == (1, 1, 70, 32, 32)
    assert img.physical_pixel_sizes.Z == pytest.approx(PX_Z_HT)


def test_bioimage_forwards_backend_kwargs():
    img = BioImage(HT_TP01, timelapse=True, refractive_index=True)
    assert img.shape == (2, 1, 70, 32, 32)
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
    rdr.set_scene("3DFL/CH1")
    assert rdr.shape == (1, 70, 32, 32)
    assert rdr.physical_pixel_sizes.Z == pytest.approx(PX_Z_HT)
    assert isinstance(rdr.xarray_dask_data.data, da.Array)
    frame = rdr.xarray_dask_data[0, 40].compute()
    np.testing.assert_array_equal(frame, Reader(FL3D_CH1_TP05).data[0, 40])
