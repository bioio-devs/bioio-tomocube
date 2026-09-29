# HANDOFF — bioio-tomocube

Status as of 2026-09-28. Branch `feat/initial-tcf-reader`, PR #1 against `main`.

This repository reads Tomocube `.TCF` files only. The HTX TIFF-export reader was split
out to `bioio-tomocube-tiff` on 2026-09-28; the reasoning, options and migration
checklist are in `docs/ADR-001-split-plugins.md`.

## Where the data lives

The instrument's own project tree from the 2025 HT-X1 Plus demo, about 100 TCFs and
565 GiB across 12 experiments. Mounted locally under `~/allen/`:

```
/allen/aics/microscopes/Jie/Microscope demo/2025 Tomocube HT-X1 Plus/Data/
    <experiment title>/                                    # "6 Well Mito", "hiPSC", ...
        <YYMMDD.HHMMSS>.<experiment>.<job>.<group>.<well>.<suffix>/
            <same name>.TCF                                # the image
            <same name>-MIP.PNG  config.dat  .experiment  .vessel  .parent
            profile/  thumbnail/  bgImages/  [sequence.dat]
```

Suffixes: `Snnn` is a snapshot (one frame), `TnnnPnn` is timelapse series `nnn` at
position `nn` with every frame in the one file, `TPnn` is a series of separate
single-frame captures. The TIFF export in `/allen/aics/lumenoid/tomocube_data_initial`
was produced from the five `…018.Group1.A1.TP01..TP05.TCF` files here.

Sidecars are not read. `config.dat` (INI) and `.experiment` (JSON) carry what the TCF
lacks: channel names (`FITC`, `TRITC`), objective NA (0.95; the TCF's `Info/Device`
`NA` of 0.68 is the condenser), device serial and user.

## Fixtures

`bioio_tomocube/tests/resources/`, produced by `scripts/make_fixtures.py` from the tree
above: first frames kept, 64 × 64 XY window at the centre, all attributes preserved,
thumbnails dropped. Tracked in Git LFS.

| File | Why |
|---|---|
| `251003.112316.6 Well Mito.018.Group1.A1.TP01.TCF` | one frame, HT + 2 FL, FL channels stored `uint8` and `uint16` |
| `251003.104445.6 Well Mito.003.Group1.A1.T001P01.TCF` | 3 of 16 timelapse frames |
| `251006.162843.myosin 24well plate.005.Group8.C5.T001P02.TCF` | 4 HT frames, 2 FL frames: modalities with different `T` |

## What the reader does

One scene per `Data/` group, fluorescence split per channel (`3DFL/CHn`, `2DFLMIP/CHn`); scene names are the HDF5 paths under `Data/`. Values
are the stored integers as `float32`; HT is RI × 10 000. Pixel sizes from
`ResolutionX/Y/Z`. OME planes carry `delta_t` from per-frame `RecordingTime`, so
`standard_metadata` reports timelapse interval and duration; acquisition date is the
first frame's `RecordingTime`. FL channels carry excitation and emission wavelengths. Stage position from the
frame `PositionX/Y` attrs. Objective from `Info/Device` magnification and NA (the NA is the condenser's, see above).

## Open items

1. Decide `T` alignment when modalities have different frame counts (myosin fixture).
2. `refractive_index=True` option, or scale HT by default.
3. Standard metadata gaps per the ADR table: `row`, `column`, `position_index` from the
   acquisition name; objective without the condenser NA; optional sidecar enrichment.
4. `TPnn` versus position: the TCFs settle it (each `TPnn` is its own single-frame job,
   same well, ~88 s apart), but the instrument operator has not confirmed intent.
5. PR #1 is still a draft. Merge, then release.
