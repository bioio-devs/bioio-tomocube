# bioio-tomocube

[![Build Status](https://github.com/bioio-devs/bioio-tomocube/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-tomocube/actions)
[![PyPI version](https://badge.fury.io/py/bioio-tomocube.svg)](https://badge.fury.io/py/bioio-tomocube)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for Tomocube `.TCF` holotomography files, the HDF5 container
written by TomoStudio on HT-series instruments. No proprietary library is needed.

For the TIFF export produced by the HTX ProcessingServer use
[bioio-tomocube-tiff](https://github.com/bioio-devs/bioio-tomocube-tiff).

---

## Documentation

For full documentation of the `BioImage` API, see
[bioio on GitHub Pages](https://bioio-devs.github.io/bioio/OVERVIEW.html) and the
[bioio-base repository](https://github.com/bioio-devs/bioio-base).

## Installation

**Stable Release:** `pip install bioio bioio-tomocube`

**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-tomocube.git`

## What you get

One `.TCF` file is one acquisition: every modality and every timepoint of one
position. Each modality is a BioIO **scene**, because the refractive-index and
fluorescence volumes are stored on different pixel grids.

| Scene | Content | Dims |
|---|---|---|
| `3D` | refractive-index volume | `TZYX` |
| `2DMIP` | its max projection, as stored by the instrument | `TYX` |
| `3DFL/CH0`, `3DFL/CH1`, … | fluorescence volumes, one scene per channel | `TZYX` |
| `2DFLMIP/CH0`, `2DFLMIP/CH1`, … | fluorescence max projections | `TYX` |

Pixel values are the stored integers cast to `float32`. The refractive-index scenes
hold RI × 10 000 (a value of 13 370 is RI 1.3370). Fluorescence channels may be stored
as `uint8` or `uint16` in the same file. Physical pixel sizes are in micrometres. `standard_metadata` carries the objective,
stage position and per-frame timing; fluorescence channels carry excitation and
emission wavelengths in the OME metadata.

## Example Usage

```python
from bioio import BioImage

img = BioImage("251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF")
img.scenes                      # ('2DFLMIP/CH0', '2DFLMIP/CH1', '2DMIP', '3D', '3DFL/CH0', '3DFL/CH1')
img.set_scene("3D")
img.shape                       # (16, 70, 1414, 1414)
img.physical_pixel_sizes        # PhysicalPixelSizes(Z=0.874, Y=0.163, X=0.163)
ri = img.get_image_dask_data("ZYX", T=0) * 1e-4

img.set_scene("3DFL/CH1")
img.shape                       # (16, 12, 1894, 1894)
```

Frame counts can differ between scenes of one file: an acquisition may record
fluorescence on every third refractive-index frame.

## Remote Filesystem Support

Any [fsspec](https://filesystem-spec.readthedocs.io)-compatible URI works, and the
filesystem is carried into the dask graph so delayed reads also work remotely:

```python
import bioio_tomocube

rdr = bioio_tomocube.Reader("s3://bucket/path/to/file.TCF")
rdr.set_scene("3D")
first_frame = rdr.xarray_dask_data[0].compute()
```

## Issues

[Search open issues across all bioio-devs repositories](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Test fixtures are tracked with Git LFS: run
`git lfs pull` before `just test`.
