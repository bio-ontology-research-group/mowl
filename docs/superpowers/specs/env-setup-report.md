# mOWL Development Environment Setup Report

**Date**: 2026-10-29  
**Repository**: `/home/leechuck/Public/software/mowl` (branch `geometric-models`)  
**Virtual Environment**: `/home/leechuck/.venvs/mowl-dev`

## 1. Installed Versions

| Component | Version |
|---|---|
| **Python** | `3.10.4` (`/home/leechuck/.pyenv/versions/3.10.4/bin/python`) |
| **pip** | `26.2.2` |
| **wheel** | `0.48.0` |
| **torch** | `2.14.1+cpu` |
| **JPype1** | `1.4.1` |
| **scipy** | `1.15.3` |
| **pytest** | `9.2.0` |
| **mowl-borg** | `0.2.1` (editable install from `/home/leechuck/Public/software/mowl`) |
| **Java / Gradle** | Java 21 / Gradle 8.7 |

## 2. Exact Commands Executed

### Step 1 & 2: Virtualenv Creation and Python Dependencies
```bash
/home/leechuck/.pyenv/versions/3.10.4/bin/python -m venv /home/leechuck/.venvs/mowl-dev
source /home/leechuck/.venvs/mowl-dev/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install pytest
```

### Step 3: Java Gateway Build
```bash
./build_jars.sh
```
*Result*: `gradle distZip` succeeded in 54s and populated `mowl/lib/` with `gateway.jar` and all OWLAPI/HermiT/ELK/Jena/SLib dependency jars.

### Step 4: Verification & Pre-staging Test Datasets
Before running the verification commands, the test datasets (`family`, `gda_human_el`, `gda_mouse_el`, `ppi_yeast_slim`) had to be downloaded from `https://bio2vec.net/data/mowl/` because the hardcoded `https://bio2vec.cbrc.kaust.edu.sa/data/mowl/...` URLs redirect to an invalid path (`https://bio2vec.net/data/deepgo/mowl/...` → HTTP 404):

```bash
for f in family.tar.gz gda_human_el.tar.gz gda_mouse_el.tar.gz ppi_yeast_slim.tar.gz; do
    curl -sO "https://bio2vec.net/data/mowl/$f" && tar -xzf "$f"
done
```

#### JVM & `FamilyDataset` Verification
```bash
source /home/leechuck/.venvs/mowl-dev/bin/activate
python -c 'import mowl; mowl.init_jvm("4g"); from tests.datasetFactory import FamilyDataset; d=FamilyDataset(); print(len(d.classes))'
```
*Output*: `8` (succeeded).

#### Pytest Suite (`tests/ontology` and `tests/reasoning`)
```bash
source /home/leechuck/.venvs/mowl-dev/bin/activate
pytest -m 'not slow' -p no:faulthandler tests/ontology tests/reasoning -x -q
```
*Test Results*: **30 passed, 0 failed, 5 warnings** in `29.80s`.

## 3. Failures and Issues Encountered

1. **Broken Redirect on Built-in Dataset URLs (`404 Not Found`)**:
   - `mowl.datasets.builtin` (`FamilyDataset`, `GDAHumanELDataset`, `GDAMouseELDataset`, `PPIYeastSlimDataset`) uses `https://bio2vec.cbrc.kaust.edu.sa/data/mowl/<name>.tar.gz`.
   - That server issues an HTTP 302 redirect to `https://bio2vec.net/data/deepgo/mowl/<name>.tar.gz`, which returns `404 Not Found`.
   - The actual archives are hosted at `https://bio2vec.net/data/mowl/<name>.tar.gz`. Pre-downloading `.tar.gz` and extracting them into the repo root allows `RemoteDataset._download()` to find the local file and proceed.
2. **Duplicate Teardown (`tests/__init__.py` + `tests/conftest.py`)**:
   - Both `tests/__init__.py` (`tearDownModule`) and `tests/conftest.py` (`pytest_unconfigure`) call `os.remove()` on `ppi_yeast_slim.tar.gz`, `gda_human_el.tar.gz`, `gda_mouse_el.tar.gz`, and `family.tar.gz` and `shutil.rmtree()` on their extracted directories.
   - When `pytest` finishes running all 30 tests (`30 passed, 0 failed`), `tearDownModule` in `tests/__init__.py` deletes the files first, and then `pytest_unconfigure` in `tests/conftest.py` attempts to remove them again, raising `FileNotFoundError: [Errno 2] No such file or directory: 'ppi_yeast_slim.tar.gz'` at session teardown.
   - To keep the working directory ready for subsequent test runs or standalone `tests.datasetFactory.FamilyDataset()` calls, the four `.tar.gz` archives and extracted directories were re-staged in the repository root.
