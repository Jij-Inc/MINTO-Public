# MINTO: Jij Management and Insight tool for Optimization

[![PyPI version shields.io](https://img.shields.io/pypi/v/minto.svg)](https://pypi.python.org/pypi/minto/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/minto.svg)](https://pypi.python.org/pypi/minto/)
[![PyPI implementation](https://img.shields.io/pypi/implementation/minto.svg)](https://pypi.python.org/pypi/minto/)
[![PyPI format](https://img.shields.io/pypi/format/minto.svg)](https://pypi.python.org/pypi/minto/)
[![PyPI license](https://img.shields.io/pypi/l/minto.svg)](https://pypi.python.org/pypi/minto/)
[![PyPI download month](https://img.shields.io/pypi/dm/minto.svg)](https://pypi.python.org/pypi/minto/)
[![Downloads](https://pepy.tech/badge/minto)](https://pepy.tech/project/minto)

[![codecov](https://codecov.io/gh/Jij-Inc/minto/graph/badge.svg?token=ZhfvFdt1sJ)](https://codecov.io/gh/Jij-Inc/minto)

`minto` is a Python library designed for developers working on research and development or proof-of-concept experiments using mathematical optimization. Positioned similarly to mlflow in the machine learning field, `minto` provides features such as saving optimization results, automatically computing benchmark metrics, and offering visualization tools for the results.

Primarily supporting Ising optimization problems, plans to extend its support to a wide range of optimization problems, such as MIP solvers, in the future.

## Installation

`minto` can be easily installed using pip.

```shell
pip install minto
```

## Documentation and Support

Documentation: https://jij-inc.github.io/minto/

Tutorials will be provided in the future. Stay tuned!

### Building Documentation Locally

The documentation is built using Jupyter Book and supports both English and Japanese.

#### Quick Start

The easiest way to build and view the documentation locally:

```shell
# Build all documentation (English and Japanese)
make docs-build

# Build and serve locally at http://localhost:8000
make docs-serve

# Build only English version
make docs-en

# Build only Japanese version
make docs-ja

# Clean all build artifacts
make docs-clean
```

#### Manual Build Process

If you prefer to build manually:

```shell
# Build English documentation
cd docs/en
uv run jupyter-book build .

# Build Japanese documentation
cd docs/ja
uv run jupyter-book build .

# Create integrated site
cd ../..
mkdir -p _site
cp -r docs/redirect/* _site/
cp -r docs/en/_build/html _site/en
cp -r docs/ja/_build/html _site/ja

# Serve locally
cd _site
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Editing Documentation

The documentation is organized into separate English and Japanese projects:

```
docs/
├── en/           # English documentation
│   ├── _config.yml
│   ├── _toc.yml
│   ├── intro.md
│   └── ...
├── ja/           # Japanese documentation
│   ├── _config.yml
│   ├── _toc.yml
│   ├── intro.md
│   └── ...
└── redirect/     # Redirect files for language selection
```

#### Adding New Pages

1. Create your content file (`.md` or `.ipynb`) in the appropriate language directory
2. Add the file to `_toc.yml` in the corresponding language directory
3. Rebuild the documentation using the commands above

#### Language Switching

Each documentation version includes a language switcher in the sidebar. The implementation is in:
- `docs/en/language.md` - Switch to Japanese
- `docs/ja/language.md` - Switch to English

#### Common Issues

- **Build errors**: Make sure you have all dependencies installed with `uv sync --extra docs`
- **Mermaid diagrams not rendering**: Already installed via `sphinxcontrib-mermaid`
- **API documentation**: Generated automatically from source code using `autoapi`


## How to Contribute

See [CONTRIBUITING.md](CONTRIBUTING.md)

---

Copyright (c) 2023 Jij Inc.
