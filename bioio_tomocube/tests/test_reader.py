#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests against real TCF files from the HT-X1 Plus demo, cropped to 64 x 64 XY
by ``scripts/make_fixtures.py`` with every attribute preserved."""

from datetime import datetime, timedelta

import h5py
import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from ome_types.model import OME

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR

# One frame, HT + two FL channels (CH0 stored uint8, CH1 uint16).
SNAPSHOT = "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF"
# First 3 of 16 timelapse frames, 60 s apart.
TIMELAPSE = "251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF"
# FL acquired every third HT frame: 4 HT frames, 2 FL frames.
MIXED_T = "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF"

TWO_CHANNEL_SCENES = (
    "2DFLMIP/CH0",
    "2DFLMIP/CH1",
    "2DMIP",
    "3D",
    "3DFL/CH0",
    "3DFL/CH1",
)
ONE_CHANNEL_SCENES = ("2DFLMIP/CH0", "2DMIP", "3D", "3DFL/CH0")
HT_PX = (0.874491274356842, 0.16256071627140045, 0.16256071627140045)
FL_PX = (1.044531226158142, 0.1214757040143013, 0.1214757040143013)
LOW_NA_HT_PX = (1.1059743165969849, 0.33200404047966003, 0.33200404047966003)
LOW_NA_FL_PX = (1.044531226158142, 0.12126840651035309, 0.12126840651035309)


@pytest.mark.parametrize(
    "filename, scene, scenes, shape, dims, pixel_sizes",
    [
        (SNAPSHOT, "3D", TWO_CHANNEL_SCENES, (1, 70, 64, 64), "TZYX", HT_PX),
        (SNAPSHOT, "3DFL/CH1", TWO_CHANNEL_SCENES, (1, 12, 64, 64), "TZYX", FL_PX),
        (
            SNAPSHOT,
            "2DMIP",
            TWO_CHANNEL_SCENES,
            (1, 64, 64),
            "TYX",
            (None,) + HT_PX[1:],
        ),
        (
            SNAPSHOT,
            "2DFLMIP/CH1",
            TWO_CHANNEL_SCENES,
            (1, 64, 64),
            "TYX",
            (None,) + FL_PX[1:],
        ),
        (TIMELAPSE, "3D", TWO_CHANNEL_SCENES, (3, 70, 64, 64), "TZYX", HT_PX),
        (MIXED_T, "3D", ONE_CHANNEL_SCENES, (4, 132, 64, 64), "TZYX", LOW_NA_HT_PX),
        (
            MIXED_T,
            "3DFL/CH0",
            ONE_CHANNEL_SCENES,
            (2, 27, 64, 64),
            "TZYX",
            LOW_NA_FL_PX,
        ),
    ],
)
def test_reader(filename, scene, scenes, shape, dims, pixel_sizes):
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=LOCAL_RESOURCES_DIR / filename,
        set_scene=scene,
        expected_scenes=scenes,
        expected_current_scene=scene,
        expected_shape=shape,
        expected_dtype=np.float32,
        expected_dims_order=dims,
        expected_channel_names=None,
        expected_physical_pixel_sizes=pixel_sizes,
        expected_metadata_type=OME,
        reader_kwargs={},
    )


def test_unsupported_format(tmp_path):
    path = tmp_path / "not_a_tcf.txt"
    path.write_text("x")
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(path)


def test_multi_scene():
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=LOCAL_RESOURCES_DIR / SNAPSHOT,
        first_scene_id="3D",
        first_scene_shape=(1, 70, 64, 64),
        first_scene_dtype=np.dtype(np.float32),
        second_scene_id="3DFL/CH0",
        second_scene_shape=(1, 12, 64, 64),
        second_scene_dtype=np.dtype(np.float32),
        allow_same_scene_data=False,
        reader_kwargs={},
    )


def test_pixel_values_match_hdf5():
    """Values are the stored integers cast to float32: HT is RI x 10 000."""
    rdr = Reader(LOCAL_RESOURCES_DIR / SNAPSHOT)
    with h5py.File(LOCAL_RESOURCES_DIR / SNAPSHOT, "r") as f:
        for scene in ("3D", "3DFL/CH1", "2DFLMIP/CH0"):
            rdr.set_scene(scene)
            np.testing.assert_array_equal(rdr.data[0], f[f"Data/{scene}/000000"][()])
    rdr.set_scene("3D")
    assert 13_000 < rdr.data.min() < rdr.data.max() < 14_300


def test_ome_metadata():
    rdr = Reader(LOCAL_RESOURCES_DIR / SNAPSHOT)
    ome = rdr.ome_metadata
    assert len(ome.images) == len(rdr.scenes) == 6
    objective = ome.instruments[0].objectives[0]
    assert (objective.nominal_magnification, objective.lens_na) == (40.0, 0.68)
    assert ome.experimenters[0].user_name == "Default"

    images = dict(zip(rdr.scenes, ome.images))
    assert images["3D"].acquisition_date == datetime(2025, 10, 3, 11, 23, 50, 702000)
    channel = images["3DFL/CH0"].pixels.channels[0]
    assert channel.name == "CH0"
    assert (channel.excitation_wavelength, channel.emission_wavelength) == (470, 525)
    assert images["3DFL/CH1"].pixels.channels[0].excitation_wavelength == 555


def test_ome_planes_carry_frame_times():
    rdr = Reader(LOCAL_RESOURCES_DIR / TIMELAPSE)
    images = dict(zip(rdr.scenes, rdr.ome_metadata.images))
    assert [p.delta_t for p in images["3D"].pixels.planes] == [0.0, 82.771, 165.632]

    rdr = Reader(LOCAL_RESOURCES_DIR / MIXED_T)
    images = dict(zip(rdr.scenes, rdr.ome_metadata.images))
    assert images["3D"].pixels.size_t == 4
    assert images["3DFL/CH0"].pixels.size_t == 2


@pytest.mark.parametrize(
    "filename, scene, expected",
    [
        (
            TIMELAPSE,
            "3D",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 3,
                "Image Size Z": 70,
                "Image Size Y": 64,
                "Image Size X": 64,
                "Imaged By": "Default",
                "Imaging Datetime": datetime(2025, 10, 3, 10, 45, 6, 802000),
                "Objective": "40x/0.68",
                "Pixel Size Z": HT_PX[0],
                "Pixel Size Y": HT_PX[1],
                "Pixel Size X": HT_PX[2],
                "Stage Position X": 0.948,
                "Stage Position Y": -0.834,
                "Timelapse": True,
                "Timelapse Interval": timedelta(seconds=82, microseconds=816000),
                "Total Time Duration": timedelta(seconds=165, microseconds=632000),
            },
        ),
        (
            MIXED_T,
            "3DFL/CH0",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 2,
                "Image Size Z": 27,
                "Objective": "40x/0.38",
                "Pixel Size Z": LOW_NA_FL_PX[0],
                "Pixel Size X": LOW_NA_FL_PX[2],
                "Stage Position X": 0.067,
                "Stage Position Y": -1.343,
                "Timelapse": True,
                "Timelapse Interval": timedelta(seconds=901, microseconds=519000),
            },
        ),
    ],
)
def test_standard_metadata(filename, scene, expected):
    rdr = Reader(LOCAL_RESOURCES_DIR / filename)
    rdr.set_scene(scene)
    metadata = rdr.standard_metadata.to_dict()
    for key, value in expected.items():
        if isinstance(value, float):
            assert metadata[key] == pytest.approx(value), key
        else:
            assert metadata[key] == value, key
