import dask.array as da
import fsspec
import numpy as np
import pytest
from ome_types.model import OME

from bioio_tomocube import Reader

from .conftest import LOCAL_RESOURCES_DIR

_MEM_PATH = "/mito_T008P01.TCF"
_MEM_URL = f"memory://{_MEM_PATH}"


@pytest.fixture(scope="module")
def mem_fs():
    """Load mito_T008P01.TCF into the in-process memory filesystem once."""
    mem = fsspec.filesystem("memory")
    with open(LOCAL_RESOURCES_DIR / "mito_T008P01.TCF", "rb") as fh:
        mem.pipe(_MEM_PATH, fh.read())
    yield mem
    mem.rm(_MEM_PATH)


def test_remote_scenes(mem_fs):
    """Reader must enumerate all four scenes from the memory-backed TCF."""
    rdr = Reader(_MEM_URL)
    assert rdr.scenes == ("2DFLMIP", "2DMIP", "3D", "3DFL/CH0")


@pytest.mark.parametrize(
    "scene, expected_shape",
    [
        ("2DFLMIP", (1, 1890, 1890)),
        ("2DMIP", (1, 692, 692)),
        ("3D", (1, 132, 692, 692)),
        ("3DFL/CH0", (1, 65, 1890, 1890)),
    ],
)
def test_remote_shape(mem_fs, scene, expected_shape):
    """Each scene must report the correct shape via the memory filesystem."""
    rdr = Reader(_MEM_URL)
    rdr.set_scene(scene)
    assert rdr.shape == expected_shape


def test_remote_physical_pixel_sizes(mem_fs):
    """3D scene physical pixel sizes must be read correctly from memory FS."""
    rdr = Reader(_MEM_URL)
    rdr.set_scene("3D")
    pps = rdr.physical_pixel_sizes
    assert pytest.approx(pps.Z, abs=1e-6) == 1.1029136180877686
    assert pytest.approx(pps.Y, abs=1e-6) == 0.33200404047966003
    assert pytest.approx(pps.X, abs=1e-6) == 0.33200404047966003


def test_remote_ome_metadata(mem_fs):
    """OME metadata must parse successfully and carry one image per scene."""
    rdr = Reader(_MEM_URL)
    rdr.set_scene("3D")
    ome = rdr.ome_metadata
    assert isinstance(ome, OME)
    assert len(ome.images) == len(rdr.scenes)


def test_remote_dask_backed(mem_fs):
    """xarray_dask_data must be dask-backed when read from the memory filesystem."""
    rdr = Reader(_MEM_URL)
    rdr.set_scene("3D")
    assert isinstance(rdr.xarray_dask_data.data, da.Array)


def test_remote_compute_first_frame(mem_fs):
    """First frame via memory filesystem must have correct shape and dtype."""
    rdr = Reader(_MEM_URL)
    rdr.set_scene("3D")
    frame = rdr.xarray_dask_data[0].compute()
    assert frame.shape == (132, 692, 692)
    assert frame.dtype == np.float32


def test_remote_fl_channel_compute(mem_fs):
    """FL channel frame must compute correctly via the memory filesystem."""
    rdr = Reader(_MEM_URL)
    rdr.set_scene("3DFL/CH0")
    frame = rdr.xarray_dask_data[0].compute()
    assert frame.shape == (65, 1890, 1890)
    assert frame.dtype == np.float32
