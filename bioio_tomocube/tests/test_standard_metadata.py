#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from typing import Any

import pytest

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, set_scene, expected",
    [
        (
            "251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF",
            "3D",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 3,
                "Image Size Z": 70,
                "Image Size Y": 64,
                "Image Size X": 64,
                "Imaged By": "Default",
                "Imaging Datetime": datetime(2025, 10, 3, 10, 45, 6, 802000),
                "Objective": "40x",
                "Pixel Size Z": 0.874491274356842,
                "Pixel Size Y": 0.16256071627140045,
                "Pixel Size X": 0.16256071627140045,
                "Stage Position X": 0.948,
                "Stage Position Y": -0.834,
                "Timelapse": True,
                "Timelapse Interval": timedelta(seconds=82, microseconds=816000),
                "Total Time Duration": timedelta(seconds=165, microseconds=632000),
            },
        ),
        (
            "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF",
            "3DFL/CH0",
            {
                "Dimensions Present": "TZYX",
                "Image Size T": 2,
                "Image Size Z": 27,
                "Objective": "40x",
                "Pixel Size Z": 1.044531226158142,
                "Pixel Size X": 0.12126840651035309,
                "Stage Position X": 0.067,
                "Stage Position Y": -1.343,
                "Timelapse": True,
                "Timelapse Interval": timedelta(seconds=901, microseconds=519000),
            },
        ),
    ],
)
def test_tomocube_standard_metadata(
    filename: str, set_scene: str, expected: dict[str, Any]
) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri)
    reader.set_scene(set_scene)
    metadata = reader.standard_metadata.to_dict()

    for key, expected_value in expected.items():
        error_message = f"{key}: Expected: {expected_value}, Actual: {metadata[key]}"
        if isinstance(expected_value, float):
            assert metadata[key] == pytest.approx(expected_value), error_message
        else:
            assert metadata[key] == expected_value, error_message
