# ADR-001: Split bioio-tomocube into a TCF plugin and a TIFF-export plugin

**Status:** Accepted, implemented 2026-09-28 (groups 1 and 2 below; group 3 open)
**Date:** 2026-09-28
**Deciders:** Brian Whitney; bioio-devs maintainers (PR review)

## Context

`bioio-tomocube` currently reads two unrelated on-disk forms of Tomocube data behind
one entry point:

| Backend | Format | Library | Lines | Tests | Fixtures |
|---|---|---|---|---|---|
| `TCFReader` (`tcf_reader.py`) | `.TCF`, one HDF5 file per acquisition | h5py | 406 | `test_reader.py` (299), `test_remote_read.py` (88) | 2 TCFs of unknown provenance, LFS |
| `TiffReader` (`tiff_reader.py`) | HTX ProcessingServer export, many ImageJ TIFFs per acquisition | tifffile | 1040 | `test_tiff_reader.py` (582) | 10 cropped real TIFFs, LFS |

`reader.py` (104 lines) is a `Reader` whose `__new__` picks a backend from the file
extension. `ome_utils.py` (446 lines) is two halves, one per backend, that share only a
few attribute helpers. `ReaderMetadata` advertises `.TCF`, `.TIFF` and `.TIF`.

Facts established on 2026-09-28 that change the picture:

* **TCF is the native format and it exists in bulk.** The HT-X1 Plus demo tree at
  `/allen/aics/microscopes/Jie/Microscope demo/2025 Tomocube HT-X1 Plus/Data/` holds
  about 100 TCFs, 565 GiB, across 12 experiments. The current `TCFReader` opens every
  type we tried, including a 31.5 GiB, 181-frame timelapse.
* **The TIFF export is a one-off derivative.** The five-timepoint export in
  `lumenoid/tomocube_data_initial` and `rep_learn/tomocube` was produced from five of
  those TCFs. No other HTX export exists on VAST outside that demo folder.
* **FMS support needs TCF.** That is the file the instrument writes and the file that
  will be ingested.
* **The two backends have diverging open work.** The TIFF side has a pending redesign
  (directory as image, `timelapse` default on, padded `T`, `Z=1` on MIPs; HANDOFF.md §5).
  The TCF side has its own list (mixed FL dtypes, per-modality frame counts, `2DFLMIP`
  not split per channel, no `refractive_index` option). None of it crosses the boundary.

Forces:

* **BioIO routes plugins by file extension.** A plugin that declares `.tiff` is offered
  for every TIFF any user opens, alongside `bioio-ome-tiff`, `bioio-tifffile` and
  `bioio-imageio`. The bioio-devs convention is one format per plugin.
* **Ship TCF soon.** PR #1 has been a draft since July; FMS work is waiting on it.
* **Do not lose the TIFF work.** The export quirks (FL3D header lies about slice count,
  data only on pages 30–43, FL resampled onto the HT grid) took real effort to find and
  will matter again the first time someone exports for ImageJ.
* **Dependency footprint.** A TCF-only user should not need tifffile's export quirks;
  a TIFF-only user should not need h5py.

## Decision

Split into two independently released BioIO plugins:

* **`bioio-tomocube`** (this repository) reads `.TCF` only. `TCFReader` becomes the
  plugin's `Reader`. The dispatching `Reader`, the TIFF backend and the TIFF half of
  `ome_utils.py` are removed.
* **`bioio-tomocube-tiff`** (new repository) reads the HTX ProcessingServer TIFF
  export. It receives the TIFF backend, its tests, its LFS fixtures, the layout page and
  HANDOFF.md §3–5 unchanged. It is parked after the move until an export consumer appears.

No shared package. The only code both need is the acquisition-name parser
(`parse_filename`, about 100 lines, currently only in `tiff_reader.py`). It is duplicated
if the TCF side ever needs it.

## Options Considered

### Option A: Keep one plugin (status quo)

| Dimension | Assessment |
|---|---|
| Complexity | Medium: extension dispatch in `__new__`, two code paths in one package |
| Cost | None now; ongoing review friction |
| Team familiarity | Current state |
| Fit with bioio-devs convention | Poor: two formats, contested `.tiff` extension |

**Pros:** nothing to move; one install covers both forms.
**Cons:** every user pays for the TIFF probe and both dependencies; TCF release is
coupled to TIFF redesign; `Reader.__new__` dispatch will draw review comments.

### Option B: Two plugins (chosen)

| Dimension | Assessment |
|---|---|
| Complexity | Low per repo; one-time move cost |
| Cost | Second repo, CI and release line, from the bioio-devs cookiecutter |
| Team familiarity | Standard bioio-devs layout |
| Fit with bioio-devs convention | Good: one format each |

**Pros:** TCF ships on its own; clean extension declarations; honest dependencies;
TIFF work preserved with its own history and fixtures.
**Cons:** a user with both forms installs two packages; the name parser is duplicated;
two repos to keep in step with `bioio-base` bumps.

### Option C: One repository, two entry points

Register two `bioio.readers` entry points from one distribution, one module each.

| Dimension | Assessment |
|---|---|
| Complexity | Medium: two reader modules, shared metadata, one version |
| Cost | Low move cost |
| Team familiarity | Unusual; no bioio-devs plugin does this |
| Fit with bioio-devs convention | Poor |

**Pros:** shared code and CI.
**Cons:** every install still carries both dependencies and still registers the `.tiff`
probe; releases stay coupled; solves the code-structure complaint but none of the
user-facing ones.

### Option D: Drop the TIFF backend

| Dimension | Assessment |
|---|---|
| Complexity | Lowest |
| Cost | Loses 1600 lines of tested code and the export analysis |
| Fit | Fine for FMS; poor for anyone who exports from TA Viewer |

**Pros:** smallest repo.
**Cons:** the export format will be produced again (it is TA Viewer's ImageJ path), and
its quirks are non-obvious.

### Option E: Serve the TIFF export with an existing bioio TIFF reader

Tested 2026-09-28 against the cropped fixtures with `bioio-tifffile` 1.3.1 and
`bioio-tiff-glob` 1.2.0. `bioio-ome-tiff` rejects the files (no OME-XML) and
`bioio-imageio` is 2-D only, so they were not candidates.

| Reader | Result on HTX export |
|---|---|
| `bioio-tifffile` | Single file only. HT3D reads correctly: `ZYX (70, …)`, Z spacing 0.874 µm. **FL3D reads wrong**: it trusts the ImageJ header and returns `Z=12` with spacing 1.045 µm, and tifffile logs a reshape failure. No grouping of channels or timepoints. Because it declares `.tiff`, it is what BioIO picks by default for these files when our plugin is absent, so the wrong shape is silent. |
| `bioio-tiff-glob` | With a ten-line custom indexer (T from `TPnn`, C from modality and `CHn`) it assembles `TCZYX (2, 3, 70, 32, 32)` and enumerates pages, so FL3D gets the right Z. But: the input is a file list or glob, not a path, so BioIO never auto-selects it and FMS cannot hand it a file; indices must be contiguous (TP01 and TP05 alone fail with an `IndexError`); every file must share one shape, so FL2D must be filtered out and `2DMIP` cannot be a scene; `physical_pixel_sizes` comes back all `None`; no OME planes, so `imaging_datetime`, `timelapse_interval` and `total_time_duration` are `None`; no refractive-index scaling; no padding rules. |

| Dimension | Assessment |
|---|---|
| Complexity | Low to adopt, but every gap above becomes user code or a wrapper |
| Cost | None to build; recurring cost in wrong reads and hand-written indexers |
| Fit for FMS | Poor: neither reader takes a path and returns the whole acquisition with metadata |

**Pros:** nothing new to maintain; `bioio-tiff-glob` is a usable ad-hoc recipe.
**Cons:** the export's two defining quirks, the FL3D header and the multi-file grouping,
are exactly what these readers do not handle; a wrapper that fixed them would be the
TIFF backend again, minus the 582 tests.

**Verdict:** not a replacement. Two things carry forward: the `bioio-tiff-glob` indexer
snippet goes in the `bioio-tomocube-tiff` README as the interim path while that plugin is
parked, and the README of both plugins warns that `bioio-tifffile` misreads FL3D volumes.

## Trade-off Analysis

The decisive factor is extension routing. Any design that keeps `.tiff` in this plugin's
declaration (A and C) makes every TCF user a participant in TIFF plugin resolution, and
makes the bioio-devs review harder, regardless of how the code is arranged internally.
Only B and D avoid that, and D throws away work that will be needed. B's costs are
administrative: one more repository of a shape the team already maintains a dozen of.

The duplicated name parser is the only real code smell B introduces. It is small, stable
(the naming scheme is set by the instrument), and cheaper than a third package.

## Consequences

Easier:

* PR #1 shrinks to a TCF reader and becomes mergeable on its own timeline.
* `bioio-tomocube` declares `.TCF` only, depends on h5py only, and never appears in
  TIFF plugin resolution.
* The TIFF redesign in HANDOFF.md §5 can proceed, or wait, without blocking anything.
* Each repository's fixtures, LFS rules and tests are about one format.

Harder:

* Two release lines to bump when `bioio-base` changes.
* Someone reading a mixed folder of TCFs and exports needs both plugins installed.
* `parse_filename` exists twice if the TCF side adopts it.

To revisit:

* Whether `bioio-tomocube-tiff` should be published to PyPI at all, or stay a source
  repository until there is a consumer.
* Whether the TCF reader should combine modalities as channels where grids match,
  as the TIFF reader does. The native file keeps HT and FL on different grids, so today
  the answer is no.

## Standard metadata derivations

FMS ingestion reads `BioImage.standard_metadata`. `bioio-base` fills most fields from
`dims`, `physical_pixel_sizes` and the OME object each reader builds, so the work is
making sure the OME carries the right values and overriding the few fields OME cannot
express. Each plugin owns its own table. Sources marked *sidecar* live outside the file
the reader is given and are read only if present, never required.

| Field | `bioio-tomocube` (TCF) | `bioio-tomocube-tiff` (HTX export) |
|---|---|---|
| `dimensions_present`, `image_size_*` | From `dims` (base). | From `dims` (base). |
| `pixel_size_x/y/z` | `ResolutionX/Y/Z` attrs on `Data/<modality>`. | `XResolution`/`YResolution` tags; Z from the HT3D ImageJ `spacing`, never from FL headers. |
| `channels` | One per scene today. Name `HT`, or `CHn`; excitation, emission and colour from `3DFL/CHn` attrs (`Excitation` 0.47, `Emission` 0.525, `ColorR/G/B`). Fluorophore names (`FITC`, `TRITC`) only in the `.experiment` *sidecar*. | Names from the filename token (`HT`, `FL_CH0`, `FL_CH1`). The export carries no wavelengths; the matching TCF or its `.experiment` *sidecar* is the only source. |
| `imaging_datetime` | `RecordingTime` attr on frame `000000` (local time, no zone). | `DateTime` tag of the first HT3D file (local time, no zone). |
| `timelapse`, `timelapse_interval`, `total_time_duration` | Base derives from OME plane `delta_t`; OME must be built from the per-frame `Time` attr (seconds from first frame). `TimeInterval` group attr is present in older files only. | Base derives from OME planes built from each TPnn's `DateTime`. With padded `T`, missing frames have no plane, so duration and interval still come out right. |
| `objective` | `Info/Device` `Magnification` and `NA`. **Trap:** that `NA` is the condenser (0.68); the objective NA (0.95) is only in `config.dat` *sidecar*. Report `40x` and omit NA unless the sidecar is read. | Not in the export. *Sidecar* or matching TCF only. |
| `imaged_by` | Not in the file. `config.dat` `User_ID` is `Default` on every demo file. Leave `None`. | `Artist` tag (already read; empty on all files seen). |
| `binning` | Not recorded. `None`. | Not recorded. `None`. |
| `row`, `column` | Parse the well token of the acquisition name (`A1` → `A`, `1`). Needs the name parser; this is the duplicated `parse_filename`. Not implemented on the TCF side today. | Already implemented from the base name. |
| `position_index` | From the suffix: `TnnnPnn` → `nn`; `Snnn` and `TPnn` → `None`. | `TPnn` → `None` (timepoint, not position). |
| `stage_position_x/y` | `PositionX/Y` attrs on the frame dataset, millimetres. Already implemented. | Not in the export. `None`. |
| `reflectors` | Not applicable. `None`. | Not applicable. `None`. |

Two consequences for the plan:

* **The name parser moves to the TCF side too.** `row`, `column` and `position_index`
  need it, which settles the duplication question: both plugins carry a copy.
* **Sidecar reading is a TCF-side feature, opt-in by presence.** `config.dat` (INI) and
  `.experiment` (JSON) in the acquisition folder supply channel names, objective NA,
  device serial and user. The reader looks for them beside the TCF and falls back to
  file-only values when absent, as it must for a bare TCF copied out of its folder or
  read over HTTP.

## Action Items

Order matters: copy out before deleting, so nothing is ever only in git history.

**1. Create `bioio-tomocube-tiff`**

1. [x] Generate the repo (copied this repo's skeleton rather than the cookiecutter); package `bioio_tomocube_tiff`.
2. [x] Move `tiff_reader.py` to `reader.py`; `TiffReader` becomes `Reader`.
3. [x] Move `_build_tiff_image`, `build_tiff_ome`, `TiffOmeScene` and the helpers they
   use from `ome_utils.py`.
4. [x] Move `tests/test_tiff_reader.py`, `tests/resources/tiff/` and its `.gitattributes`
   LFS rule; confirm `git lfs ls-files` shows all 8 TIFFs after push.
5. [x] `ReaderMetadata` declares `.TIFF`, `.TIF`; `_is_supported_image` stays gated on
   the HTX `Software` tag and the filename pattern so it declines ordinary TIFFs.
6. [x] Move HANDOFF.md §3–5 and `docs/tomocube-htx-layout.html`; leave a one-line
   pointer here.
7. [x] Dependencies: `tifffile`, `bioio-base`, `dask`, `fsspec`, `numpy`, `ome-types`.
   No h5py.
8. [x] Mark the repo README as parked: reads the one known export; open design in
   its HANDOFF.
9. [x] README: the `bioio-tiff-glob` indexer recipe as the interim path, and a warning
   that `bioio-tifffile` returns the wrong Z for FL3D files (Option E).

**2. Strip `bioio-tomocube` to TCF**

1. [x] Delete `reader.py` (the dispatcher) and `tiff_reader.py`; rename `tcf_reader.py`
   to `reader.py` with `Reader = TCFReader` or a straight rename.
2. [x] Delete the TIFF half of `ome_utils.py`.
3. [x] `__init__.py` exports `Reader`, `ReaderMetadata` only.
4. [x] `ReaderMetadata` declares `.TCF` only; drop `tifffile` from `pyproject.toml`.
5. [x] Delete `tests/test_tiff_reader.py`, `tests/resources/tiff/`, the TIFF LFS rule
   and `unsupported.tif` if nothing else uses it.
6. [x] Trim README to TCF; retitle PR #1 "TCF reader".
   Add the same `bioio-tifffile` FL3D warning for anyone who has only the export.
7. [x] Replace the two fixtures of unknown provenance with cropped copies of demo
   files (h5py copy with sliced datasets, attrs preserved, XY cropped to 64 px):
   * `6 Well Mito/…018.Group1.A1.TP01.TCF`: one frame, HT + 2 FL, mixed FL dtypes.
   * `6 Well Mito/…003.Group1.A1.T001P01.TCF`: first 3 frames of a timelapse.
   * `myosin 24well plate/…005.Group8.C5.T001P02.TCF`: first 4 HT frames and the
     FL frames that fall within them, to cover per-modality frame counts.
8. [x] Record fixture provenance (HANDOFF.md, fixtures section, and `scripts/make_fixtures.py`).

**3. After the split (TCF side, separate PRs)**

1. [x] Split `2DFLMIP` per channel to match `3DFL/CHn` (done in the 2026-09-28 scrub; scene names are now the HDF5 paths under `Data/`).
2. [ ] Decide `T` handling when modalities have different frame counts (the myosin
   case: 181 HT, 61 FL). The padding decision from the TIFF side applies.
3. [ ] Add `refractive_index=True` for parity with the TIFF reader, or scale by default.
4. [ ] Standard metadata, per the table above: OME planes carry `delta_t` from the
   `Time` attr; `objective` reports magnification without the condenser NA; copy
   `parse_filename` and fill `row`, `column`, `position_index`. Channel excitation
   and emission from `CHn` attrs are in the OME as of the 2026-09-28 scrub;
   `imaging_datetime` is now the earliest frame of any scene.
5. [ ] Sidecar enrichment: read `config.dat` and `.experiment` when they sit beside the
   TCF for channel names (FITC, TRITC), objective NA and user; never require them.
6. [ ] Extend `test_standard_metadata` with expected values for every field in the
   table, on the new fixtures.

**3b. After the split (TIFF side, when resumed)**

1. [ ] `position_index` stays `None` for `TPnn`; document that wavelengths are
   unavailable without the source TCF.

**4. Housekeeping**

1. [ ] Add both plugins to the bioio plugin list once released.
2. [ ] Update memory and HANDOFF.md in both repos to point at each other.
