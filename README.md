# bioio-tomocube

[![Build Status](https://github.com/bioio-devs/bioio-tomocube/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-tomocube/actions)
[![PyPI version](https://badge.fury.io/py/bioio-tomocube.svg)](https://badge.fury.io/py/bioio-tomocube)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for Tomocube `.TCF` holotomography files, the HDF5 container
written by TomoStudio on HT-series instruments.
---

## Documentation

For full documentation of the `BioImage` API, see
[bioio on GitHub Pages](https://bioio-devs.github.io/bioio/OVERVIEW.html) and the
[bioio-base repository](https://github.com/bioio-devs/bioio-base).

## Installation

**Stable Release:** `pip install bioio bioio-tomocube`
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-tomocube.git`

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

## Issues

[Search open issues across all bioio-devs repositories](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Test fixtures are tracked with Git LFS: run
`git lfs pull` before `just test`.
