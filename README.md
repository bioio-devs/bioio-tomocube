# bioio-tomocube

[![Build Status](https://github.com/bioio-devs/bioio-tomocube/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-tomocube/actions)
[![PyPI version](https://badge.fury.io/py/bioio-tomocube.svg)](https://badge.fury.io/py/bioio-tomocube)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for reading Tomocube TCF holotomography files.

---

## Documentation

For full documentation of the `BioImage` API, see
[bioio on GitHub Pages](https://bioio-devs.github.io/bioio/OVERVIEW.html) and the
[bioio-base repository](https://github.com/bioio-devs/bioio-base).

## Installation

**Stable Release:** `pip install bioio bioio-tomocube`

**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-tomocube.git`

## Overview

This plugin reads `.TCF` files produced by [Tomocube](https://www.tomocube.com) HT-series
holotomography instruments. Pixel data is read directly from the HDF5 structure inside
each TCF file via `h5py`; no proprietary library is required.

Each imaging modality stored in the file is exposed as a separate BioIO **scene**:

| Scene name | Content | Dimensions |
|---|---|---|
| `3D` | 3-D refractive index volume | `TZYX` |
| `2DMIP` | 2-D max intensity projection | `TYX` |
| `2DFLMIP` | 2-D fluorescence max intensity projection | `TYX` |
| `3DFL/CH0`, `3DFL/CH1`, … | 3-D fluorescence channels | `TZYX` |

All pixel data is returned as `float32`. Physical pixel sizes (µm) are read from the
HDF5 metadata and exposed via `physical_pixel_sizes`.

## Example Usage

```python
from bioio import BioImage
import bioio_tomocube

img = BioImage("my_file.TCF", reader=bioio_tomocube.Reader)

# list modalities
print(img.scenes)          # ('2DMIP', '3D', '3DFL/CH0', ...)

# read the refractive-index volume as a dask array
img.set_scene("3D")
data = img.get_image_dask_data("TZYX")

# physical pixel sizes in µm
print(img.physical_pixel_sizes)  # PhysicalPixelSizes(Z=0.22, Y=0.11, X=0.11)
```

## Remote Filesystem Support

The reader accepts any [fsspec](https://filesystem-spec.readthedocs.io)-compatible URI,
including S3, GCS, and plain HTTPS. The filesystem is serialised into the dask graph so
that delayed pixel reads also work remotely without downloading the file first:

```python
# Read directly from an HTTPS endpoint
rdr = bioio_tomocube.Reader("https://example.org/path/to/file.TCF")
rdr.set_scene("3D")
first_frame = rdr.xarray_dask_data[0].compute()
```

## Issues

[Search open issues across all bioio-devs repositories](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
