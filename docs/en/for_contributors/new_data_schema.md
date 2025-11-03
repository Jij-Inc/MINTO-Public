# For Contributors: Adding a New Data Schema

This tutorial explains how to add a new data schema. A data schema is used to define the structure of a dataset. The implementation example is based on the addition of `ommx.v1.SampleSet` during the upgrade from v1.0 to v1.1.

## Implementing StorageStrategy

When adding a new data schema, you need to implement `StorageStrategy` in `minto.v1.datastore.py`. `StorageStrategy` is an interface for reading and writing datasets.

```python
@dataclass
class SampleSetStorage(StorageStrategy[ommx_v1.SampleSet]):
    def save(self, data: ommx_v1.SampleSet, path: pathlib.Path):
        blob = data.to_bytes()
        with open(path, "wb") as f:
            f.write(blob)

    def load(self, path: pathlib.Path) -> ommx_v1.SampleSet:
        with open(path, "rb") as f:
            return ommx_v1.SampleSet.from_bytes(f.read())

    def add_to_artifact_builder(
        self,
        data: ommx_v1.SampleSet,
        builder: ox_art.ArtifactBuilder,
        annotations: dict[str, str],
    ):
        blob = data.to_bytes()
        builder.add_layer(
            "application/org.ommx.v1.sampleset", blob, annotations
        )

    def load_from_layer(self, artifact: ox_art.Artifact, layer: ox_art.Descriptor):
        blob = artifact.get_blob(layer)
        return ommx_v1.SampleSet.from_bytes(blob)

    @property
    def extension(self):
        return "sampleset"
```

`SampleSetStorage` inherits from `StorageStrategy`. `StorageStrategy` requires the implementation of five methods: `save`, `load`, `add_to_artifact_builder`, `load_from_layer`, and `extension`.

Next, register the added `SampleSetStorage` in `DataStore._storage_mapping` and an attribute in the `DataStore` class.

```python
@dataclasses.dataclass
class DataStore:
    problems: dict[str, jm.Problem] = field(default_factory=dict)
    instances: dict[str, ommx_v1.Instance] = field(default_factory=dict)
    solutions: dict[str, ommx_v1.Solution] = field(default_factory=dict)
    objects: dict[str, dict] = field(default_factory=dict)
    parameters: dict[str, dict[str, typ.Any]] = field(default_factory=dict)
    samplesets: dict[str, ommx_v1.SampleSet] = field(default_factory=dict) # Add this line
    meta_data: dict[str, typ.Any] = field(default_factory=dict)

    _storage_mapping: typ.ClassVar[dict[str, StorageStrategy]] = {
        "problems": ProblemStorage(),
        "instances": InstanceStorage(),
        "solutions": SolutionStorage(),
        "objects": JSONStorage(),
        "parameters": JSONStorage(),
        "samplesets": SampleSetStorage(), # Add this line
        "meta_data": JSONStorage(),
    }

```

By doing this, `DataStore` will recognize the `"sampleset"` data schema and use `SampleSetStorage` to read and write datasets.

## Adding `log_sampleset` Method to `Experiment` Class

Now that `DataStore` can recognize `ommx.v1.SampleSet`, the next step is to add `Experiment.log_sampleset`.

```python
    def log_sampleset(
        self,
        name: str | ommx_v1.SampleSet,
        value: typ.Optional[ommx_v1.SampleSet] = None
    ):
        """Log a SampleSet to the experiment or run database."""

        # If name is specified, use it as the name.
        # If name is not specified and a SampleSet is passed as the first argument,
        # use the variable name of the SampleSet as the name.
        datastore = self.get_current_datastore()
        _name, _value = self._get_name_or_default(
            name, value, datastore.samplesets
        )

        self.log_data(_name, _value, "samplesets")
```

The `_get_name_or_default` method is used to determine the name and value.
If `name` is specified, it is used as is; if not, the variable name of the `SampleSet` is used.

Using `_name` and `_value`, the `SampleSet` is logged to the database.
By specifying `"samplesets"` as the third argument, `DataStore` can recognize `ommx.v1.SampleSet`.

## Adding Conversion Method for Table Data

You can view the data of `Experiment` as a `pandas.DataFrame` using methods like `Experiment.get_run_table` or `.get_experiment_tables`. The conversion of each data type is described in `minto/table.py`. Here, we will add a method to convert `SampleSet` to `pandas.DataFrame`.

For example, the conversion method for `ommx.v1.Solution` is implemented as follows:

```python
def _extract_solution_info(solution: ommx_v1.Solution):
    info = {
        "objective": solution.objective,
        "feasible": solution.feasible,
        "optimality": solution.optimality,
        "relaxation": solution.relaxation,
        "start": solution.start,
    }
    return info
```

Similarly, add a method to convert `SampleSet`.

```python
def _extract_sampleset_info(sampleset: ommx_v1.SampleSet):
    summary = sampleset.summary
    objective = summary.objective
    return {
        "num_samples": len(summary),
        "obj_mean": objective.mean(),
        "obj_std": objective.std(),
        "obj_min": objective.min(),
        "obj_max": objective.max(),
        "feasible": summary.feasible.sum(),
        "feasible_unrelaxed": summary.feasible_unrelaxed.sum(),
    }
```

Then, implement this method to be called within `create_table_info`.

```python
def create_table_info(datastore: DataStore) -> dict:
    instance_data = {}
    for name, inst in datastore.instances.items():
        instance_data[name] = _extract_instance_info(inst)

    solution_data = {}
    for name, sol in datastore.solutions.items():
        solution_data[name] = _extract_solution_info(sol)

    # Added code -------------------------
    sampleset_data = {}
    for name, sampleset in datastore.samplesets.items():
        sampleset_data[name] = _extract_sampleset_info(sampleset)
    # ------------------------- Added code

    return {
        "instance": instance_data,
        "solution": solution_data,
        "sampleset": sampleset_data,  # Added code
        "parameter": datastore.parameters,
        "metadata": datastore.meta_data,
    }
```

This adds a method to convert `SampleSet` to `pandas.DataFrame`.

## Summary

Appropriately write test code under `tests/` in between the above implementations.
With the above implementation, you can log `SampleSet` to `Experiment` as follows:

```python

experiment = Experiment()
with experiment.run():
    experiment.log_sampleset("sampleset", sampleset)
experiment.get_run_table()
```

Contributors can add new data schemas in this way.
