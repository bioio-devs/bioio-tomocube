"""Reads through fsspec's in-memory filesystem stand in for remote reads."""

import dask.array as da
import fsspec
import numpy as np
import pytest

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR
from .test_reader import SNAPSHOT, TWO_CHANNEL_SCENES

_MEM_PATH = "/snapshot.TCF"
_MEM_URL = f"memory://{_MEM_PATH}"


@pytest.fixture(scope="module")
def mem_fs():
    mem = fsspec.filesystem("memory")
    mem.pipe(_MEM_PATH, (LOCAL_RESOURCES_DIR / SNAPSHOT).read_bytes())
    yield mem
    mem.rm(_MEM_PATH)


def test_remote_scenes_and_metadata(mem_fs):
    rdr = Reader(_MEM_URL)
    assert rdr.scenes == TWO_CHANNEL_SCENES
    rdr.set_scene("3D")
    assert rdr.shape == (1, 70, 64, 64)
    assert rdr.physical_pixel_sizes.Z == pytest.approx(0.874491274356842)
    assert len(rdr.ome_metadata.images) == 6


def test_remote_read_is_lazy_and_matches_local(mem_fs):
    remote = Reader(_MEM_URL)
    remote.set_scene("3DFL/CH0")
    assert isinstance(remote.xarray_dask_data.data, da.Array)
    frame = remote.xarray_dask_data[0].compute()
    assert frame.dtype == np.float32

    local = Reader(LOCAL_RESOURCES_DIR / SNAPSHOT)
    local.set_scene("3DFL/CH0")
    np.testing.assert_array_equal(frame, local.data[0])
