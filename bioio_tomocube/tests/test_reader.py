#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests against real TCF files from the HT-X1 Plus demo, cropped to 64 x 64 XY
with every attribute preserved."""

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities

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
        expected_metadata_type=dict,
        reader_kwargs={},
    )


def test_unsupported_format(tmp_path):
    path = tmp_path / "not_a_tcf.txt"
    path.write_text("x")
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(path)
