#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from ome_types.model import OME

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes, "
    "expected_metadata_type",
    [
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        (
            "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF",
            "3D",
            ("2DFLMIP/CH0", "2DFLMIP/CH1", "2DMIP", "3D", "3DFL/CH0", "3DFL/CH1"),
            (1, 70, 64, 64),
            np.float32,
            "TZYX",
            None,
            (0.874491274356842, 0.16256071627140045, 0.16256071627140045),
            OME,
        ),
        (
            "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF",
            "3DFL/CH1",
            ("2DFLMIP/CH0", "2DFLMIP/CH1", "2DMIP", "3D", "3DFL/CH0", "3DFL/CH1"),
            (1, 12, 64, 64),
            np.float32,
            "TZYX",
            None,
            (1.044531226158142, 0.1214757040143013, 0.1214757040143013),
            OME,
        ),
        (
            "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF",
            "2DMIP",
            ("2DFLMIP/CH0", "2DFLMIP/CH1", "2DMIP", "3D", "3DFL/CH0", "3DFL/CH1"),
            (1, 64, 64),
            np.float32,
            "TYX",
            None,
            (None, 0.16256071627140045, 0.16256071627140045),
            OME,
        ),
        (
            "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF",
            "2DFLMIP/CH1",
            ("2DFLMIP/CH0", "2DFLMIP/CH1", "2DMIP", "3D", "3DFL/CH0", "3DFL/CH1"),
            (1, 64, 64),
            np.float32,
            "TYX",
            None,
            (None, 0.1214757040143013, 0.1214757040143013),
            OME,
        ),
        (
            "251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF",
            "3D",
            ("2DFLMIP/CH0", "2DFLMIP/CH1", "2DMIP", "3D", "3DFL/CH0", "3DFL/CH1"),
            (3, 70, 64, 64),
            np.float32,
            "TZYX",
            None,
            (0.874491274356842, 0.16256071627140045, 0.16256071627140045),
            OME,
        ),
        (
            "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF",
            "3D",
            ("2DFLMIP/CH0", "2DMIP", "3D", "3DFL/CH0"),
            (4, 132, 64, 64),
            np.float32,
            "TZYX",
            None,
            (1.1059743165969849, 0.33200404047966003, 0.33200404047966003),
            OME,
        ),
        (
            "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF",
            "3DFL/CH0",
            ("2DFLMIP/CH0", "2DMIP", "3D", "3DFL/CH0"),
            (2, 27, 64, 64),
            np.float32,
            "TZYX",
            None,
            (1.044531226158142, 0.12126840651035309, 0.12126840651035309),
            OME,
        ),
    ],
)
def test_tomocube_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[Optional[float], float, float],
    expected_metadata_type: type,
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )


def test_ome_metadata() -> None:
    uri = LOCAL_RESOURCES_DIR / "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF"
    rdr = Reader(uri)
    ome = rdr.ome_metadata
    assert len(ome.images) == len(rdr.scenes) == 6
    objective = ome.instruments[0].objectives[0]
    assert objective.nominal_magnification == 40.0
    assert objective.lens_na is None  # Info/Device NA is the condenser's
    assert ome.experimenters[0].user_name == "Default"

    images = dict(zip(rdr.scenes, ome.images))
    assert images["3D"].acquisition_date == datetime(2025, 10, 3, 11, 23, 50, 702000)
    channel = images["3DFL/CH0"].pixels.channels[0]
    assert channel.name == "CH0"
    assert (channel.excitation_wavelength, channel.emission_wavelength) == (470, 525)
    assert images["3DFL/CH1"].pixels.channels[0].excitation_wavelength == 555


@pytest.mark.parametrize(
    "filename, set_scene, expected_size_t, expected_delta_t",
    [
        # Timelapse: every HT frame carries its offset from the first.
        (
            "251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF",
            "3D",
            3,
            [0.0, 82.771, 165.632],
        ),
        # FL acquired every third HT frame: the two stacks have their own T.
        (
            "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF",
            "3D",
            4,
            None,
        ),
        (
            "251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF",
            "3DFL/CH0",
            2,
            None,
        ),
    ],
)
def test_ome_planes_carry_frame_times(
    filename: str,
    set_scene: str,
    expected_size_t: int,
    expected_delta_t: Optional[List[float]],
) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    rdr = Reader(uri)
    images = dict(zip(rdr.scenes, rdr.ome_metadata.images))
    pixels = images[set_scene].pixels

    assert pixels.size_t == expected_size_t
    if expected_delta_t is not None:
        assert [p.delta_t for p in pixels.planes] == expected_delta_t
