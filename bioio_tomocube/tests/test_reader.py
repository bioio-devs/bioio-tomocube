#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import h5py
import numpy as np
import pytest
from bioio_base import exceptions, test_utilities

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
            dict,
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
            dict,
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
            dict,
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
            dict,
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
            dict,
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
            dict,
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
            dict,
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


def test_multi_scene() -> None:
    uri = LOCAL_RESOURCES_DIR / "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF"
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=uri,
        first_scene_id="3D",
        first_scene_shape=(1, 70, 64, 64),
        first_scene_dtype=np.dtype(np.float32),
        second_scene_id="3DFL/CH0",
        second_scene_shape=(1, 12, 64, 64),
        second_scene_dtype=np.dtype(np.float32),
        allow_same_scene_data=False,
        reader_kwargs={},
    )


@pytest.mark.parametrize("set_scene", ["3D", "3DFL/CH1", "2DFLMIP/CH0"])
def test_pixel_values_match_hdf5(set_scene: str) -> None:
    """Values are the stored integers cast to float32: HT is RI x 10 000."""
    uri = LOCAL_RESOURCES_DIR / "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF"
    rdr = Reader(uri)
    rdr.set_scene(set_scene)

    with h5py.File(uri, "r") as f:
        expected = f[f"Data/{set_scene}/000000"][()]
    np.testing.assert_array_equal(rdr.data[0], expected)

    if set_scene == "3D":
        assert 13_000 < rdr.data.min() < rdr.data.max() < 14_300
