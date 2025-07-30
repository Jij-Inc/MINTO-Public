import pathlib
import tempfile

import jijmodeling as jm
import numpy as np
import pytest

import minto
import minto.v0
from minto.migrator import migrate_to_v1_from_v0


@pytest.fixture
def temp_dir():
    """Create a temporary directory for experiment data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


@pytest.fixture
def v0_experiment(temp_dir):
    """Create a sample v0 experiment with various data types."""
    # Create a v0 experiment
    exp_v0 = minto.v0.Experiment(name="test_migration", savedir=temp_dir)

    # Run 1: simple parameters
    with exp_v0.run():
        exp_v0.log_parameter("learning_rate", 0.01)
        exp_v0.log_parameter("batch_size", 32)
        exp_v0.log_parameter("optimizer", "adam")

    # Run 2: parameters with problem
    with exp_v0.run():
        # Create a simple problem
        problem = jm.Problem("test_problem")
        x = jm.BinaryVar("x", shape=(3,))
        problem += x[:].sum()
        problem += jm.Constraint("constraint", x[:].sum() == 1)

        exp_v0.log_parameter("alpha", 0.5)
        exp_v0.log_parameter("beta", 1.5)
        exp_v0.log_result("problem", problem)

    # Run 3: parameters with list and dict
    with exp_v0.run():
        exp_v0.log_parameter("gamma", 2.0)
        exp_v0.log_result("layers", [64, 128, 64])
        exp_v0.log_result("config", {"activation": "relu", "dropout": 0.5})

    # Save the experiment
    exp_v0.save()

    return exp_v0


@pytest.fixture
def v0_experiment_with_sampleset(temp_dir):
    """Create a v0 experiment with a SampleSet."""
    exp_v0 = minto.v0.Experiment(name="test_sampleset", savedir=temp_dir)

    # Create a simple problem
    problem = jm.Problem("test_problem")
    x = jm.BinaryVar("x", shape=(3,))
    problem += x[:].sum()

    # Add variables to the sample
    var_values = jm.experimental.SparseVarValues(
        "x",
        {(0,): 1, (1,): 0, (2,): 0},
        shape=(3,),
        var_type=jm.experimental.VarType.BINARY,
    )

    # Add evaluation results
    eval_result = jm.experimental.EvaluationResult(
        objective=1.0,
    )

    # Create a mock sampleset
    sample = jm.experimental.Sample(
        num_occurrences=1,
        run_id="sample1",
        var_values={"x": var_values},
        eval=eval_result,
    )

    sampleset = jm.experimental.SampleSet(data=[sample])

    with exp_v0.run():
        exp_v0.log_parameter("temperature", 1.0)
        exp_v0.log_result("sampleset", sampleset)
        exp_v0.log_result("problem", problem)

    # Save the experiment
    exp_v0.save()

    return exp_v0


@pytest.fixture
def v0_experiment_with_complex_data(temp_dir):
    """Create a v0 experiment with complex data structures."""
    exp_v0 = minto.v0.Experiment(name="test_complex", savedir=temp_dir)

    with exp_v0.run():
        # NumPy array
        array_data = np.array([[1, 2, 3], [4, 5, 6]])
        exp_v0.log_result("array", array_data)

        # Nested dictionary
        nested_dict = {"level1": {"level2": {"value": 42, "list": [1, 2, 3]}}}
        exp_v0.log_result("nested_dict", nested_dict)

    # Save the experiment
    exp_v0.save()

    return exp_v0


def test_migration_preserves_data(temp_dir, v0_experiment):
    """Test that migration preserves all data from v0 to v1."""
    # Migrate to v1
    exp_v1 = migrate_to_v1_from_v0(v0_experiment)

    # Check experiment name
    assert exp_v1.name == v0_experiment.name

    # Get the run table for verification
    run_table = exp_v1.get_run_table()

    # Verify run 1
    assert run_table.loc[0, ("parameter", "learning_rate")] == 0.01
    assert run_table.loc[0, ("parameter", "batch_size")] == 32
    assert run_table.loc[0, ("parameter", "optimizer")] == "adam"

    # Verify run 2
    assert run_table.loc[1, ("parameter", "alpha")] == 0.5
    assert run_table.loc[1, ("parameter", "beta")] == 1.5

    # Verify that the problem is migrated correctly
    assert any("problem" in col for col in run_table.columns)

    # Verify run 3
    assert run_table.loc[2, ("parameter", "gamma")] == 2.0

    # Verify objects are migrated correctly
    assert any("layers" in col for col in run_table.columns)
    assert any("config" in col for col in run_table.columns)


def test_migration_with_sampleset(v0_experiment_with_sampleset):
    """Test that migration with SampleSet objects raises appropriate error."""
    # Migration should fail with JijModeling SampleSet
    with pytest.raises(TypeError, match="JijModeling SampleSet is no longer supported"):
        migrate_to_v1_from_v0(v0_experiment_with_sampleset)


def test_migration_save_load(temp_dir, v0_experiment):
    """Test that migrated experiments can be saved and loaded correctly."""
    # Migrate to v1
    exp_v1 = migrate_to_v1_from_v0(v0_experiment)

    # Save the migrated experiment
    migration_dir = temp_dir / "migrated"
    migration_dir.mkdir(exist_ok=True)
    exp_v1.save(migration_dir)

    # Try to load the saved experiment
    loaded_exp = minto.Experiment.load_from_dir(migration_dir)

    # Verify experiment name
    assert loaded_exp.name == v0_experiment.name

    # Verify run data
    run_table = loaded_exp.get_run_table()
    assert len(run_table) == 3  # Should have 3 runs
    assert run_table.loc[0, ("parameter", "learning_rate")] == 0.01
    assert run_table.loc[1, ("parameter", "alpha")] == 0.5
    assert run_table.loc[2, ("parameter", "gamma")] == 2.0

    # Verify specific data in the runs
    assert loaded_exp.runs[1].parameters["alpha"] == 0.5
    assert loaded_exp.runs[1].parameters["beta"] == 1.5
    assert "problem" in loaded_exp.runs[1].problems


def test_migration_with_complex_data(temp_dir, v0_experiment_with_complex_data):
    """Test migration with complex data structures like numpy arrays."""
    # Migrate to v1
    exp_v1 = migrate_to_v1_from_v0(v0_experiment_with_complex_data)

    # Verify migration of complex data
    # For arrays and complex structures, we check if they were migrated
    # to objects storage correctly
    assert "array" in exp_v1.runs[0].objects
    assert "nested_dict" in exp_v1.runs[0].objects

    # Additional verification can be done by checking the object content
    array_obj = exp_v1.runs[0].objects["array"]
    assert "array" in array_obj

    nested_dict_obj = exp_v1.runs[0].objects["nested_dict"]

    # Verify nested structures are preserved
    assert "level1" in nested_dict_obj
    assert "level2" in nested_dict_obj["level1"]
    assert nested_dict_obj["level1"]["level2"]["value"] == 42
    assert nested_dict_obj["level1"]["level2"]["list"] == [1, 2, 3]


def test_migration_empty_experiment(temp_dir):
    """Test migration of an empty experiment."""
    # Create an empty v0 experiment
    exp_v0 = minto.v0.Experiment(name="empty_test", savedir=temp_dir)

    # Migrate to v1
    exp_v1 = migrate_to_v1_from_v0(exp_v0)

    # Verify basic properties
    assert exp_v1.name == exp_v0.name
    assert len(exp_v1.runs) == 0


def test_migration_with_multiple_runs_and_objects(temp_dir):
    """Test migration with multiple runs and multiple objects per run."""
    # Create a v0 experiment with multiple runs
    exp_v0 = minto.v0.Experiment(name="multi_runs", savedir=temp_dir)

    # Create some problems and configurations to reuse
    problem1 = jm.Problem("problem1")
    x1 = jm.BinaryVar("x", shape=(5,))
    problem1 += x1[:].sum()

    problem2 = jm.Problem("problem2")
    x2 = jm.BinaryVar("x", shape=(10,))
    y2 = jm.BinaryVar("y", shape=(10,))
    problem2 += x2[:].sum() + y2[:].sum()

    configs = [
        {"solver": "SA", "sweeps": 1000, "beta": 0.1},
        {"solver": "SQA", "sweeps": 500, "trotter": 4},
        {"solver": "MCMC", "sweeps": 2000, "temp": 0.5},
    ]

    # Add runs with various combinations
    for i, config in enumerate(configs):
        with exp_v0.run():
            # Log configuration
            exp_v0.log_parameters(config)

            # Log a problem
            problem = problem1 if i % 2 == 0 else problem2
            exp_v0.log_result(f"problem_{i}", problem)

            # Log some arrays
            exp_v0.log_result(f"array_{i}", np.random.rand(5, 5))

            # Log some metrics
            exp_v0.log_parameter("accuracy", 0.85 + i * 0.05)
            exp_v0.log_parameter("time", 10.0 + i * 2.5)

    # Save the experiment
    exp_v0.save()

    # Migrate to v1
    exp_v1 = migrate_to_v1_from_v0(exp_v0)

    # Verify migration
    assert len(exp_v1.runs) == len(configs)

    # Verify each run has the correct data
    run_table = exp_v1.get_run_table()
    for i, config in enumerate(configs):
        # Check config parameters
        for k, v in config.items():
            assert run_table.loc[i, ("parameter", k)] == v

        # Check metrics
        assert run_table.loc[i, ("parameter", "accuracy")] == 0.85 + i * 0.05
        assert run_table.loc[i, ("parameter", "time")] == 10.0 + i * 2.5

        # Check problem exists
        assert any(f"problem_{i}" in col for col in run_table.columns)

        # Check array exists
        assert f"array_{i}" in exp_v1.runs[i].objects
