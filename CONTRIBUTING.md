# Contributing to bioio-tomocube

## Prerequisites

**Git LFS** must be installed before you clone. Test resources (`.TCF` sample
files) are stored in LFS. Cloning without LFS causes those files to appear as
text pointer stubs and all fixture-dependent tests to fail.

Install Git LFS: https://git-lfs.github.com/

**just** is used as the task runner: https://just.systems/

## Setting Up a Development Environment

```bash
git clone git@github.com:bioio-devs/bioio-tomocube.git
cd bioio-tomocube
just setup-dev
```

`just setup-dev` runs `pip install -e .[lint,test]` and installs the
pre-commit hooks.

## Available Commands

| Command | Description |
|---|---|
| `just build` | Run lint then tests (mirrors CI) |
| `just test` | Run pytest with coverage |
| `just lint` | Run pre-commit on all files |
| `just benchmark` | Run read-performance benchmarks; writes `output.csv` |
| `just clean` | Remove build artefacts |
| `just install` | Install runtime + lint + test dependencies |

## Making Changes

1. Fork the repository and clone your fork.
2. Create a feature branch: `git checkout -b feat/short-description`.
3. Make your changes.
4. Validate locally: `just build`.
5. Commit and push to your fork.
6. Open a pull request against `main`.

## Running Tests Without the Sample File

Tests decorated with `@_requires_sample` are skipped automatically when
`bioio_tomocube/tests/resources/sample.TCF` is absent. The format-detection
and non-local-filesystem tests run unconditionally.

To run the full test suite, place the sample file at
`bioio_tomocube/tests/resources/sample.TCF`. A public copy is available in
the [TCFile repository](https://github.com/ehgus/TCFile) under `tests/sample.TCF`.

## Adding Test Fixtures

Test `.TCF` files are tracked by Git LFS and live in
`bioio_tomocube/tests/resources/`. Contributors cannot add new fixtures
directly via a pull request — the file must be committed to LFS by a project
maintainer. Open an issue requesting the fixture, attach the file or a link
to it, and a maintainer will add it.

TIFF-export fixtures live in `bioio_tomocube/tests/resources/tiff/` and are
also tracked by Git LFS. They are real HTX ProcessingServer exports
cropped to 32x32 pixels with every TIFF tag and page preserved, so they
reproduce the real files' quirks (for example the FL3D ImageJ header that
reports `slices=12` while the file holds 70 pages). To regenerate them from a
full-size acquisition, crop each page with `tifffile` and copy the
`ImageDescription`, `Model`, `Software`, `DateTime`, `Artist`, `HostComputer`
and resolution tags of the first page verbatim.

## Release Process (Maintainers Only)

```bash
just tag-for-release vX.Y.Z
just release
```

A tag beginning with `v` triggers the automated PyPI publish via CI. Version
numbers are managed by `setuptools-scm` and derived from the git tag.
