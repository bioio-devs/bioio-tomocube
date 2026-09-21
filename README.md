# bioio-tomocube

[![Build Status](https://github.com/bioio-devs/bioio-tomocube/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-tomocube/actions)
[![PyPI version](https://badge.fury.io/py/bioio-tomocube.svg)](https://badge.fury.io/py/bioio-tomocube)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for Tomocube holotomography data: `.TCF` files and HTX `.TIFF` exports.

---

## Documentation

For full documentation of the `BioImage` API, see
[bioio on GitHub Pages](https://bioio-devs.github.io/bioio/OVERVIEW.html) and the
[bioio-base repository](https://github.com/bioio-devs/bioio-base).

## Installation

**Stable Release:** `pip install bioio bioio-tomocube`

**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-tomocube.git`

## Overview

This plugin reads data produced by [Tomocube](https://www.tomocube.com) HT-series
holotomography instruments in either of the two forms it leaves the instrument in:

| Format | Extension | Written by | Backend |
|---|---|---|---|
| TCF (HDF5 container) | `.TCF` | TomoStudio | `bioio_tomocube.TCFReader` (via `h5py`) |
| TIFF export (one ImageJ-style multi-page TIFF per modality and channel) | `.TIFF` / `.TIF` | HTX ProcessingServer | `bioio_tomocube.TiffReader` (via `tifffile`) |

`bioio_tomocube.Reader` picks the backend from the file extension, so the same code
works for both. No proprietary library is required.

Each imaging modality is exposed as a separate BioIO **scene** with the same names in
both formats:

| Scene name | Content | Dimensions |
|---|---|---|
| `3D` | 3-D refractive index volume | `TZYX` |
| `2DMIP` | 2-D refractive index max intensity projection | `TYX` |
| `3DFL/CH0`, `3DFL/CH1`, … | 3-D fluorescence channels | `TZYX` |
| `2DFLMIP` (TCF) / `2DFLMIP/CH0`, … (TIFF) | 2-D fluorescence max intensity projection | `TYX` |

Physical pixel sizes (µm) are exposed via `physical_pixel_sizes` for both formats.

### TCF files

Pixel data is read directly from the HDF5 structure and returned as `float32`.

### TIFF exports

An HTX export of one acquisition (one timepoint) looks like this on disk:

```
<base>.TP01_HT3D_0.00.TIFF              # refractive-index volume
<base>.TP01/
    <base>.TP01_FL3D_CH0_0.00.TIFF      # fluorescence volume, channel 0
    <base>.TP01_FL3D_CH1_0.00.TIFF
    <base>.TP01_FL2D_CH0_0.00.TIFF      # fluorescence max projection (optional)
```

Pass **any** of those files to the reader; the sibling files of the same acquisition
are discovered automatically and the default scene is the modality of the file you
passed. Pixel data is returned as stored (`uint16`).

Two options are specific to this backend:

* `timelapse=True` stacks every `<base>.TPnn` timepoint found next to the file along
  `T`, ordered by timepoint. `time_interval` and the OME planes are derived from the
  TIFF `DateTime` tags.
* `refractive_index=True` returns the `3D`/`2DMIP` scenes as `float32` refractive
  index. The export stores refractive index × 10 000 as `uint16`.

Quirks of the export that the reader handles for you:

* The `FL3D` volumes are resampled onto the HT Z grid, but their ImageJ header still
  reports the original fluorescence slice count and spacing. `tifffile` alone
  therefore reports the wrong shape. This reader always enumerates the TIFF pages
  directly and uses the `HT3D` sibling's Z spacing for `FL3D` scenes; the header value
  is kept in `reader.tiff_metadata["imagej"]`.
* Fluorescence data occupies only a sub-range of the Z planes; the remaining planes
  are zero.

## Example Usage

```python
from bioio import BioImage
import bioio_tomocube

# TCF
img = BioImage("my_file.TCF", reader=bioio_tomocube.Reader)
print(img.scenes)          # ('2DMIP', '3D', '3DFL/CH0', ...)
img.set_scene("3D")
data = img.get_image_dask_data("TZYX")
print(img.physical_pixel_sizes)  # PhysicalPixelSizes(Z=0.22, Y=0.11, X=0.11)

# TIFF export: one timepoint, raw uint16
img = BioImage("exp.TP01_HT3D_0.00.TIFF")   # auto-selected, no reader= needed
print(img.scenes)          # ('3D', '3DFL/CH0', '3DFL/CH1')

# TIFF export: all timepoints stacked along T, HT volume as refractive index
img = BioImage("exp.TP01_HT3D_0.00.TIFF", timelapse=True, refractive_index=True)
img.set_scene("3D")
ri = img.get_image_dask_data("TZYX")        # float32
print(img.time_interval)                    # mean interval between timepoints
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
