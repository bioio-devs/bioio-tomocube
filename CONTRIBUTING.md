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

## Test Fixtures

The `.TCF` files in `bioio_tomocube/tests/resources/` are real acquisitions from
an Allen HT-X1 Plus, cropped to a 64 × 64 window with the first few frames kept
and every HDF5 attribute preserved. `scripts/make_fixtures.py` regenerates them
from the source files on the Allen network; `HANDOFF.md` lists what each one
covers. They are tracked by Git LFS, so run `git lfs pull` after cloning.

Contributors cannot add fixtures through a pull request; the file must be
committed to LFS by a maintainer. Open an issue describing the acquisition
type the new fixture would cover.

## Release Process (Maintainers Only)

```bash
just tag-for-release vX.Y.Z
just release
```

A tag beginning with `v` triggers the automated PyPI publish via CI. Version
numbers are managed by `setuptools-scm` and derived from the git tag.
