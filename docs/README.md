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

or 

```bash
sphinx-build -b html source/ build/
cd build/html
python3 -m http.server 8000
```


Navigate to `http://localhost:8000/` to view

In case new functions/Classes are added: 

```bash
sphinx-apidoc -o source/api ../mowl/
```

## Build remotely

1. Push changes to `mowl/docs`
2. Navigate to `https://readthedocs.org/projects/mowl/` where you `Build version`with the `docs` version.

# Great tutorial
https://sphinx-rtd-tutorial.readthedocs.io/en/latest/index.html


# Testing code snippets in documentation:

```bash
make doctest
```

or 

```bash
sphinx-build -b doctest source/ build/
```

Outputs of the tests can be found at `build/output.txt`

For more information on testing, check [this](https://sphinx-tutorial.readthedocs.io/step-3/)

To check the coverage of the tests in the documentation sources run

```bash
sphinx-build -b coverage source/ build/
```

The coverage results are in `build/python.txt`
