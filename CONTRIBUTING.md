## How to Contribute

> _Development Environment Policy_:  
> Our policy is to establish a simple development environment that allows everyone to easily contribute to the project. With this in mind, we carefully select the necessary commands for setting up the environment to be as minimal as possible. Based on this policy, we have adopted an environment using `uv` in this project.

### Setup environment with `uv`

1: Setup uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2: Setup development environment

```
uv sync --all-extras
```

3: Setup `pre-commit`

In this project, we use pre-commit hooks to help maintain code quality. This ensures that predefined checks and formatting are automatically executed before each commit.

`pre-commit` was installed by the above command `uv sync`.
So, next enable the pre-commit hooks by running the following command in the project's root directory:

```
uv run pre-commit install
```

> **Notes on Using pre-commit:**  
> With pre-commit enabled, predefined checks and formatting will be automatically executed before each commit. If any errors are detected during this process, the commit will be aborted. You will not be able to commit until the errors are resolved, so please fix the errors and try committing again.

You may need run `ruff` before commit.

```
uv run ruff check --fix
uv run ruff format
```

4: Check tests

```
uv run pytest tests
```

### When you want add a dependency

**Standard dependency**

```
uv add ...
```

**Dependency for test**

```
uv add ... --optional test
```

**Dependency for dev**

```
uv add ... --optional dev
```

## Documentation Update

We use mkdocs to generate the documentation for `minto`.

### Check in local environment

```
uv run mkdocs serve
```
