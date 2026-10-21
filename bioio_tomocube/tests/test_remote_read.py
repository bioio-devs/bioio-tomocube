#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reads through fsspec's in-memory filesystem stand in for remote reads."""

from typing import Iterator

import dask.array as da
import fsspec
import numpy as np
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR

LOCAL_URI = LOCAL_RESOURCES_DIR / "251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF"
MEM_PATH = "/snapshot.TCF"
MEM_URI = f"memory://{MEM_PATH}"


@pytest.fixture(scope="module")
def mem_fs() -> Iterator[MemoryFileSystem]:
    mem = fsspec.filesystem("memory")
    mem.pipe(MEM_PATH, LOCAL_URI.read_bytes())
    yield mem
    mem.rm(MEM_PATH)


def test_remote_scenes_and_metadata(mem_fs: MemoryFileSystem) -> None:
    rdr = Reader(MEM_URI)
    assert rdr.scenes == (
        "2DFLMIP/CH0",
        "2DFLMIP/CH1",
        "2DMIP",
        "3D",
        "3DFL/CH0",
        "3DFL/CH1",
    )
    rdr.set_scene("3D")
    assert rdr.shape == (1, 70, 64, 64)
    assert rdr.physical_pixel_sizes.Z == pytest.approx(0.874491274356842)
    assert len(rdr.ome_metadata.images) == 6


def test_remote_read_is_lazy_and_matches_local(mem_fs: MemoryFileSystem) -> None:
    remote = Reader(MEM_URI)
    remote.set_scene("3DFL/CH0")
    assert isinstance(remote.xarray_dask_data.data, da.Array)
    frame = remote.xarray_dask_data[0].compute()
    assert frame.dtype == np.float32

    local = Reader(LOCAL_URI)
    local.set_scene("3DFL/CH0")
    np.testing.assert_array_equal(frame, local.data[0])
