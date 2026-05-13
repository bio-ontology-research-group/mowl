# mOWL Development Guide

## Project Overview

mOWL (Machine Learning with Ontologies) is a Python library for building ML models that operate on biomedical ontologies. It wraps the OWL API (Java) via JPype and exposes ontology projectors, corpus generators, and embedding models (ELEmbeddings, ELBoxEmbeddings, etc.) through a unified Python interface.

Key dependency: **a JVM is required at runtime** (OpenJDK 11+). JAR files are bundled in `mowl/lib/`.

## Development Setup

Use the conda environments in `envs/`. Dev variants install extra tooling:

```bash
conda env create -f envs/environment_dev_3_12.yml
conda activate <env-name>
pip install -e .
```

Environments exist for Python 3.10, 3.11, 3.12, and 3.13.

## Running Tests

```bash
pytest -m "not slow"        # fast tests only (used in CI)
pytest                      # full suite including slow tests
pytest tests/models/        # single module
```

The test suite uses a shared `conftest.py` with JVM-initialising fixtures. Tests marked `@pytest.mark.slow` are excluded from CI.

## Documentation

Docs are in `docs/source/` and built with Sphinx + ReadTheDocs.

**Rule: always use `.. testcode::` instead of `.. code-block:: python` in RST files.**
Sphinx doctest runs all `testcode` blocks to verify syntax and that imports/calls match the actual code. `code-block:: python` is static and never checked. Use `code-block:: bash` only for shell commands.

The `doctest_global_setup` in `docs/source/conf.py` initialises the JVM before all blocks run.

To build docs locally:

```bash
cd docs
make html        # full build
make doctest     # run all testcode blocks
```

## CI/CD Overview

| Event | Action |
|---|---|
| PR / push to `main` or `develop` | Run unit tests (Python 3.10–3.13) + flake8 lint |
| Push to `main` | Build package → deploy to TestPyPI |
| Push tag `v*` | Build package → deploy to PyPI + create GitHub Release (Sigstore-signed) |
| 1st of Jan / Apr / Jul / Oct | Open quarterly release checklist issue |

Workflows live in `.github/workflows/`. Dependabot runs weekly for pip dependency updates.

ReadTheDocs builds are triggered automatically on every push to `main`.

## Release Process

The version is derived from git tags via `setuptools_scm` — **do not set version strings manually** in `setup.py` or `setup.cfg`.

### Step-by-step release checklist

1. **Merge** all pending PRs (feature branches) into `main`.
2. **Update `CHANGELOG.md`**: rename `[Unreleased]` to `[X.Y.Z]` with today's date. Add a new empty `[Unreleased]` section at the top. Update the comparison link at the bottom.
3. **Update `docs/source/conf.py`**: set `release` and `version` to `X.Y.Z`.
4. **Commit** the above two files:
   ```bash
   git commit -m "Release vX.Y.Z"
   ```
5. **Tag and push**:
   ```bash
   git tag vX.Y.Z
   git push origin main --tags
   ```
6. GitHub Actions builds the package and publishes to PyPI automatically. Monitor the `python-publish.yml` workflow run.
7. Verify the release on https://pypi.org/project/mowl-borg/ and that ReadTheDocs reflects the new version.

### Versioning policy (semver)

- **PATCH** (`1.0.x`): bug fixes, dependency bumps, no API changes — release on demand, any time.
- **MINOR** (`1.x.0`): new backwards-compatible features (new model, new dataset, new option) — released on the quarterly schedule.
- **MAJOR** (`x.0.0`): breaking API changes (renamed classes, removed parameters, changed return types) — released on the quarterly schedule.

Current released version: **v1.0.3** (see `CHANGELOG.md` and latest git tag).

### Release cadence

**Minor and major releases** follow a quarterly schedule. The workflow `.github/workflows/release-reminder.yml` opens a release-checklist issue automatically on the first day of each quarter (January, April, July, October). When the issue is opened, work through the steps above.

**Patch releases** are cut on demand whenever a critical bug fix needs to reach users before the next quarter. The same steps apply; no checklist issue is required. A patch release does not reset the quarterly clock.
