#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from ome_types.model import OME

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR

_READER_PARAMS = [
    (
        "sample.TCF",
        "3D",
        ("2DMIP", "3D"),
        (10, 208, 296, 296),
        np.float32,
        "TZYX",
        None,
        (0.190974358974359, 0.09551566211946805, 0.09551566211946805),
    ),
    (
        "sample.TCF",
        "2DMIP",
        ("2DMIP", "3D"),
        (10, 296, 296),
        np.float32,
        "TYX",
        None,
        (None, 0.09551566211946805, 0.09551566211946805),
    ),
    (
        "mito_T008P01.TCF",
        "2DFLMIP",
        ("2DFLMIP", "2DMIP", "3D", "3DFL/CH0"),
        (1, 1890, 1890),
        np.float32,
        "TYX",
        None,
        (None, 0.1217217817902565, 0.1217217817902565),
    ),
    (
        "mito_T008P01.TCF",
        "2DMIP",
        ("2DFLMIP", "2DMIP", "3D", "3DFL/CH0"),
        (1, 692, 692),
        np.float32,
        "TYX",
        None,
        (None, 0.33200404047966003, 0.33200404047966003),
    ),
    (
        "mito_T008P01.TCF",
        "3D",
        ("2DFLMIP", "2DMIP", "3D", "3DFL/CH0"),
        (1, 132, 692, 692),
        np.float32,
        "TZYX",
        None,
        (1.1029136180877686, 0.33200404047966003, 0.33200404047966003),
    ),
    (
        "mito_T008P01.TCF",
        "3DFL/CH0",
        ("2DFLMIP", "2DMIP", "3D", "3DFL/CH0"),
        (1, 65, 1890, 1890),
        np.float32,
        "TZYX",
        None,
        (1.0416406393051147, 0.1217217817902565, 0.1217217817902565),
    ),
]

_FILE_SCENE_PARAMS = [(p[0], p[1]) for p in _READER_PARAMS]


# ---------------------------------------------------------------------------
# Core reader checks — shape, dtype, dims, pixel sizes, metadata type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, set_scene, expected_scenes, expected_shape, expected_dtype, "
    "expected_dims_order, expected_channel_names, expected_physical_pixel_sizes",
    _READER_PARAMS,
)
def test_reader(
    filename,
    set_scene,
    expected_scenes,
    expected_shape,
    expected_dtype,
    expected_dims_order,
    expected_channel_names,
    expected_physical_pixel_sizes,
):
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=LOCAL_RESOURCES_DIR / filename,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
        reader_kwargs={},
    )


def test_unsupported_format():
    """Non-TCF files must raise UnsupportedFileFormatError."""
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(LOCAL_RESOURCES_DIR / "unsupported.tif")


# ---------------------------------------------------------------------------
# Multi-scene cache invalidation
# ---------------------------------------------------------------------------


def test_multi_scene():
    """Scene switch must invalidate caches and return the correct data."""
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=LOCAL_RESOURCES_DIR / "sample.TCF",
        first_scene_id="3D",
        first_scene_shape=(10, 208, 296, 296),
        first_scene_dtype=np.dtype("float32"),
        second_scene_id="2DMIP",
        second_scene_shape=(10, 296, 296),
        second_scene_dtype=np.dtype("float32"),
        allow_same_scene_data=False,
        reader_kwargs={},
    )


# ---------------------------------------------------------------------------
# OME metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename, scene", _FILE_SCENE_PARAMS)
def test_ome_metadata_structure(filename, scene):
    """OME must be an OME instance with one image per scene."""
    rdr = Reader(LOCAL_RESOURCES_DIR / filename)
    rdr.set_scene(scene)
    ome = rdr.ome_metadata
    assert isinstance(ome, OME)
    assert len(ome.images) == len(rdr.scenes)


def test_ome_metadata_instrument():
    """sample.TCF OME must carry objective magnification and NA."""
    rdr = Reader(LOCAL_RESOURCES_DIR / "sample.TCF")
    rdr.set_scene("3D")
    obj = rdr.ome_metadata.instruments[0].objectives[0]

    assert pytest.approx(obj.nominal_magnification, abs=0.01) == 58.33
    assert pytest.approx(obj.lens_na, abs=0.001) == 1.2


def test_ome_metadata_acquisition_date():
    """sample.TCF OME acquisition_date must parse from CreateDate (2021)."""
    rdr = Reader(LOCAL_RESOURCES_DIR / "sample.TCF")
    rdr.set_scene("3D")
    acq = rdr.ome_metadata.images[0].acquisition_date

    assert isinstance(acq, datetime)
    assert acq.year == 2021


def test_ome_metadata_planes():
    """sample.TCF Plane list must have delta_t=0 at T=0 and positive after."""
    rdr = Reader(LOCAL_RESOURCES_DIR / "sample.TCF")
    rdr.set_scene("3D")
    planes = rdr.ome_metadata.images[0].pixels.planes

    assert len(planes) == 10
    assert planes[0].delta_t == 0.0
    assert all(p.delta_t > 0.0 for p in planes[1:])


# ---------------------------------------------------------------------------
# Standard metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, scene, expected",
    [
        (
            "sample.TCF",
            "3D",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 10,
                "Image Size X": 296,
                "Image Size Y": 296,
                "Image Size Z": 208,
                "Imaging Datetime": datetime(2021, 7, 23, 17, 51, 55),
                "Objective": "58x/1.2Water",
                "Pixel Size X": 0.09551566211946805,
                "Pixel Size Y": 0.09551566211946805,
                "Pixel Size Z": 0.190974358974359,
                "Stage Position X": 0.706384,
                "Stage Position Y": 1.53143,
                "Timelapse": True,
                "Timelapse Interval": timedelta(microseconds=992556),
            },
        ),
        (
            "sample.TCF",
            "2DMIP",
            {
                "Dimensions Present": "TYX",
                "Image Size T": 10,
                "Image Size X": 296,
                "Image Size Y": 296,
                "Image Size Z": None,
                "Imaging Datetime": datetime(2021, 7, 23, 17, 51, 55),
                "Objective": "58x/1.2Water",
                "Pixel Size X": 0.09551566211946805,
                "Pixel Size Y": 0.09551566211946805,
                "Pixel Size Z": None,
                "Stage Position X": 0.706384,
                "Stage Position Y": 1.53143,
                "Timelapse": True,
                "Timelapse Interval": timedelta(microseconds=992556),
            },
        ),
        (
            "mito_T008P01.TCF",
            "3D",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 1,
                "Image Size X": 692,
                "Image Size Y": 692,
                "Image Size Z": 132,
                "Imaging Datetime": datetime(2026, 6, 25, 12, 9, 53),
                "Objective": "40x/0.38Water",
                "Pixel Size X": 0.33200404047966003,
                "Pixel Size Y": 0.33200404047966003,
                "Pixel Size Z": 1.1029136180877686,
                "Stage Position X": 0.531,
                "Stage Position Y": 0.657,
                "Timelapse": False,
                "Timelapse Interval": None,
            },
        ),
        (
            "mito_T008P01.TCF",
            "3DFL/CH0",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 1,
                "Image Size X": 1890,
                "Image Size Y": 1890,
                "Image Size Z": 65,
                "Imaging Datetime": datetime(2026, 6, 25, 12, 9, 53),
                "Objective": "40x/0.38Water",
                "Pixel Size X": 0.1217217817902565,
                "Pixel Size Y": 0.1217217817902565,
                "Pixel Size Z": 1.0416406393051147,
                "Stage Position X": 0.531,
                "Stage Position Y": 0.657,
                "Timelapse": False,
                "Timelapse Interval": None,
            },
        ),
    ],
)
def test_standard_metadata(filename, scene, expected):
    """standard_metadata.to_dict() must match expected values."""
    rdr = Reader(LOCAL_RESOURCES_DIR / filename)
    rdr.set_scene(scene)
    metadata = rdr.standard_metadata.to_dict()

    for key, expected_value in expected.items():
        error_message = f"{key}: Expected {expected_value!r}, got {metadata[key]!r}"
        if isinstance(expected_value, float):
            assert metadata[key] == pytest.approx(expected_value), error_message
        else:
            assert metadata[key] == expected_value, error_message


@pytest.mark.parametrize("scene", ["3D", "2DMIP"])
def test_time_interval(scene):
    """time_interval must be a timedelta of ~1 second for sample.TCF."""
    rdr = Reader(LOCAL_RESOURCES_DIR / "sample.TCF")
    rdr.set_scene(scene)
    assert rdr.time_interval == timedelta(seconds=1.0)
