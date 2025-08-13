import pathlib

import jijmodeling as jm
import ommx.artifact as ox_art
import ommx_pyscipopt_adapter as scip_ad
import pytest
from helpers import assert_instance_equal, assert_sampleset_equal, assert_solution_equal

from minto.v1.datastore import DataStore


def knapsack():
    v = jm.Placeholder("v", ndim=1)
    n = v.shape[0]
    w = jm.Placeholder("w", shape=(n,))
    C = jm.Placeholder("C")
    x = jm.BinaryVar("x", shape=(n,))

    problem = jm.Problem("knapsack", sense=jm.ProblemSense.MAXIMIZE)
    i = jm.Element("i", n)
    problem += jm.sum(i, v[i] * x[i])
    problem += jm.Constraint("capa", jm.sum(i, w[i] * x[i]) <= C)
    return problem


def knapsack_data():
    return {"v": [1, 2, 3, 4, 5], "w": [2, 3, 4, 5, 6], "C": 8}


def knapsack_instance():
    problem = knapsack()
    data = knapsack_data()
    interpreter = jm.Interpreter(data)
    instance = interpreter.eval_problem(problem)
    return instance


def knapsack_solution():
    instance = knapsack_instance()
    adapter = scip_ad.OMMXPySCIPOptAdapter(instance)
    scip_model = adapter.solver_input
    scip_model.optimize()
    solution = adapter.decode(scip_model)
    return solution


def knapsack_sampleset():
    instance = knapsack_instance()
    samples = {
        0: {0: 0, 1: 0, 2: 1, 3: 1, 4: 0},  # Sample 0
        1: {0: 1, 1: 0, 2: 0, 3: 1, 4: 0},  # Sample 1
        2: {0: 0, 1: 1, 2: 1, 3: 0, 4: 0},  # Sample 2
        3: {0: 1, 1: 1, 2: 0, 3: 0, 4: 0},  # Sample 3
        4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},  # Sample 4
    }
    sampleset = instance.evaluate_samples(samples)
    return sampleset


def test_simple(tmp_path):
    ds = DataStore({}, {}, {}, {}, {}, {})
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    ds.add("knapsack", problem, "problems", with_save=True, save_dir=tmp_path)
    ds.add("knapsack", data, "objects", with_save=True, save_dir=tmp_path)
    ds.add("knapsack", instance, "instances", with_save=True, save_dir=tmp_path)
    ds.add("value", 0.2, "parameters", with_save=True, save_dir=tmp_path)
    ds.add("meta", "meta", "meta_data", with_save=True, save_dir=tmp_path)
    ds.add("knapsack", solution, "solutions", with_save=True, save_dir=tmp_path)

    assert ds.problems["knapsack"] == problem
    assert ds.objects["knapsack"] == data
    assert_instance_equal(ds.instances["knapsack"], instance)
    assert ds.parameters == {"value": 0.2}
    assert ds.meta_data == {"meta": "meta"}
    assert_solution_equal(ds.solutions["knapsack"], solution)

    ds2 = DataStore.load(tmp_path)
    assert jm.is_same(ds2.problems["knapsack"], problem)
    assert ds2.objects["knapsack"] == data
    assert_instance_equal(ds2.instances["knapsack"], instance)
    assert ds2.parameters == {"value": 0.2}
    assert ds2.meta_data == {"meta": "meta"}
    assert_solution_equal(ds2.solutions["knapsack"], solution)

    file_name = tmp_path / "sample.ommx"
    builder = ox_art.ArtifactBuilder.new_archive_unnamed(file_name)
    annotations = {"test": "test"}
    ds2.add_to_artifact_builder(builder, annotations)
    artifact = builder.build()

    artifact = ox_art.Artifact.load_archive(file_name)
    ds3 = DataStore.load_from_layers(artifact, artifact.layers)
    assert jm.is_same(ds3.problems["knapsack"], problem)
    assert ds3.objects["knapsack"] == data

    instance.annotations.update(
        {"test": "test", "org.minto.name": "knapsack", "org.minto.storage": "instances"}
    )
    assert_instance_equal(ds3.instances["knapsack"], instance)
    assert ds3.parameters == {"value": 0.2}
    assert ds3.meta_data == {"meta": "meta"}
    solution.annotations.update(
        {"test": "test", "org.minto.name": "knapsack", "org.minto.storage": "solutions"}
    )
    assert_solution_equal(ds3.solutions["knapsack"], solution)


def test_simple_sampleset(tmp_path):
    """Test for saving and loading sample sets"""
    ds = DataStore({}, {}, {}, {}, {}, {})
    sampleset = knapsack_sampleset()

    # Add and save sample set
    ds.add("test_sampleset", sampleset, "samplesets", with_save=True, save_dir=tmp_path)

    # Verification
    assert "test_sampleset" in ds.samplesets
    assert_sampleset_equal(ds.samplesets["test_sampleset"], sampleset)

    # Test loading from file
    ds2 = DataStore.load(tmp_path)
    assert "test_sampleset" in ds2.samplesets
    assert_sampleset_equal(ds2.samplesets["test_sampleset"], sampleset)

    # Test saving to and loading from artifact
    file_name = tmp_path / "sampleset.ommx"
    builder = ox_art.ArtifactBuilder.new_archive_unnamed(file_name)
    annotations = {"test": "test"}
    ds2.add_to_artifact_builder(builder, annotations)
    artifact = builder.build()

    # Load from artifact
    artifact = ox_art.Artifact.load_archive(file_name)
    ds3 = DataStore.load_from_layers(artifact, artifact.layers)

    # Verification
    assert "test_sampleset" in ds3.samplesets
    assert_sampleset_equal(ds3.samplesets["test_sampleset"], sampleset)


def test_empty_datastore(tmp_path):
    """Test for empty DataStore"""
    ds = DataStore()
    ds.save_all(tmp_path)
    loaded_ds = DataStore.load(tmp_path)

    assert not loaded_ds.problems
    assert not loaded_ds.instances
    assert not loaded_ds.solutions
    assert not loaded_ds.objects
    assert not loaded_ds.parameters
    assert not loaded_ds.meta_data


def test_multiple_problems(tmp_path):
    """Test for saving and loading multiple problems"""
    ds = DataStore()

    # Add two knapsack problems
    problem1 = knapsack()
    problem2 = knapsack()  # Same structure but different instance

    ds.add("knapsack1", problem1, "problems", with_save=True, save_dir=tmp_path)
    ds.add("knapsack2", problem2, "problems", with_save=True, save_dir=tmp_path)

    # Test loading
    loaded_ds = DataStore.load(tmp_path)
    assert len(loaded_ds.problems) == 2
    assert jm.is_same(loaded_ds.problems["knapsack1"], problem1)
    assert jm.is_same(loaded_ds.problems["knapsack2"], problem2)


def test_update_existing_data(tmp_path):
    """Test for updating existing data"""
    ds = DataStore()

    # Add initial data
    ds.add("param1", 1.0, "parameters", with_save=True, save_dir=tmp_path)

    # Update with same key
    ds.add("param1", 2.0, "parameters", with_save=True, save_dir=tmp_path)

    # Test loading
    loaded_ds = DataStore.load(tmp_path)
    assert loaded_ds.parameters["param1"] == 2.0


def test_invalid_storage_name():
    """Test for invalid storage name"""
    ds = DataStore()

    with pytest.raises(AttributeError):
        ds.add("test", {}, "invalid_storage")


def test_nested_object_storage(tmp_path):
    """Test for saving nested objects"""
    ds = DataStore()

    nested_data = {"level1": {"level2": {"value": 42, "list": [1, 2, 3]}}}

    ds.add("nested", nested_data, "objects", with_save=True, save_dir=tmp_path)

    # Test loading
    loaded_ds = DataStore.load(tmp_path)
    assert loaded_ds.objects["nested"] == nested_data


def test_large_metadata(tmp_path):
    """Test for saving large metadata"""
    ds = DataStore()

    # Create large dictionary
    large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
    ds.add("meta", large_metadata, "meta_data", with_save=True, save_dir=tmp_path)

    # Test loading
    loaded_ds = DataStore.load(tmp_path)
    assert loaded_ds.meta_data == {"meta": large_metadata}


def test_artifact_annotations(tmp_path):
    """Test for artifact annotations"""
    ds = DataStore()
    problem = knapsack()
    ds.add("test_problem", problem, "problems")

    file_name = tmp_path / "test.ommx"
    builder = ox_art.ArtifactBuilder.new_archive_unnamed(file_name)

    # Add multiple annotations
    annotations = {
        "created_by": "test_user",
        "timestamp": "2024-01-01",
        "version": "1.0",
    }

    ds.add_to_artifact_builder(builder, annotations)
    builder.build()

    # Verify annotations
    loaded_artifact = ox_art.Artifact.load_archive(file_name)
    for layer in loaded_artifact.layers:
        for key, value in annotations.items():
            assert layer.annotations[key] == value


def test_file_path_handling(tmp_path):
    """Test for file path handling"""
    ds = DataStore()

    # Test with different path formats
    paths = [
        tmp_path,
        str(tmp_path),
        tmp_path / "subdir",
        pathlib.Path(tmp_path) / "subdir2",
    ]

    for path in paths:
        if isinstance(path, (str, pathlib.Path)):
            path = pathlib.Path(path)
            path.mkdir(exist_ok=True, parents=True)

        ds.add("test", {"value": 1}, "objects", with_save=True, save_dir=path)
        loaded_ds = DataStore.load(path)
        assert loaded_ds.objects["test"]["value"] == 1


def test_sequential_operations(tmp_path):
    """Test for sequential operations"""
    ds = DataStore()

    # Execute a series of operations
    operations = [
        ("problems", knapsack(), "prob1"),
        ("objects", {"data": 1}, "obj1"),
        ("parameters", 0.5, "param1"),
        ("meta_data", "meta1", "meta1"),
    ]

    # Add sequentially
    for storage_name, data, name in operations:
        ds.add(name, data, storage_name, with_save=True, save_dir=tmp_path)

        # Test loading after each operation
        loaded_ds = DataStore.load(tmp_path)
        if storage_name == "problems":
            assert jm.is_same(loaded_ds.problems[name], data)
        elif storage_name in ["parameters", "meta_data"]:
            assert getattr(loaded_ds, storage_name)[name] == data
        else:
            assert getattr(loaded_ds, storage_name)[name] == data
