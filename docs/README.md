# MOWL documentation 

## Requirements

- sphinx-build
- sphinx_rtd_theme

## Build locally

```bash
make html
cd build/html
python3 -m http.server 8000
```

Navigate to `http://localhost:8000/` to view

## Build remotely

1. Push changes to `mowl/docs`
2. Navigate to `https://readthedocs.org/projects/mowl/` where you `Build version`with the `docs` version.

