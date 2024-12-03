## How to Contribute

> *Development Environment Policy*:  
> Our policy is to establish a simple development environment that allows everyone to easily contribute to the project. With this in mind, we carefully select the necessary commands for setting up the environment to be as minimal as possible. Based on this policy, we have adopted an environment using `poetry` in this project.

### Setup environment with `poetry`

1: Setup poetry
```
pip install -U pip
pip install poetry
poetry self add "poetry-dynamic-versioning[plugin]"
poetry install
```

2: Setup `pre-commit`

In this project, we use pre-commit hooks to help maintain code quality. This ensures that predefined checks and formatting are automatically executed before each commit.

`pre-commit` was installed by the above command `poetry install`.
So, next enable the pre-commit hooks by running the following command in the project's root directory:

```
pre-commit install
```

> **Notes on Using pre-commit:**  
> With pre-commit enabled, predefined checks and formatting will be automatically executed before each commit. If any errors are detected during this process, the commit will be aborted. You will not be able to commit until the errors are resolved, so please fix the errors and try committing again.

You may need run `black` and `isort` before commit.
```
python -m isort ./minto
python -m black ./minto
```

3: Check tests

```
poetry shell
python -m pytest tests
```

### When you want add a dependency

**Standard dependency**
```
poetry add ...
```

**Depencency for test**
```
poetry add ... -G tests
```

**Depencency for dev**
```
poetry add ... -G dev
```

## Documentation Update

We use mkdocs to generate the documentation for `minto`.  

### Check in local environment

```
mkdocs serve
```


