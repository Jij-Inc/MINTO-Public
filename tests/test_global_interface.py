#!/usr/bin/env python3
"""
Test for new log_global_* interface methods.
Tests the new clear interface for experiment-level vs run-level data.
"""

import pathlib
import tempfile
import warnings

import jijmodeling as jm
import pytest

import minto


def knapsack_problem():
    """Create a simple knapsack problem for testing."""
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


@pytest.fixture
def tmp_path_fixture():
    """Provide a temporary directory for experiments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


def test_new_global_interface_basic(tmp_path_fixture):
    """Test basic functionality of new log_global_* methods."""
    exp = minto.Experiment("test_global_interface", savedir=tmp_path_fixture)

    # Test log_global_parameter
    exp.log_global_parameter("dataset", "TSP")
    exp.log_global_parameter("problem_size", 50)
    exp.log_global_parameter("max_iterations", 1000)

    # Test log_global_config
    exp.log_global_config(
        "study_metadata",
        {"author": "researcher", "purpose": "testing", "version": "1.0"},
    )

    # Test log_global_problem
    problem = knapsack_problem()
    exp.log_global_problem("knapsack", problem)

    # Verify data is stored at experiment level
    exp_store = exp.dataspace.experiment_datastore
    assert exp_store.parameters["dataset"] == "TSP"
    assert exp_store.parameters["problem_size"] == 50
    assert exp_store.parameters["max_iterations"] == 1000
    assert exp_store.objects["study_metadata"]["author"] == "researcher"
    assert jm.is_same(exp_store.problems["knapsack"], problem)


def test_global_vs_run_data_separation(tmp_path_fixture):
    """Test clear separation between global and run-level data."""
    exp = minto.Experiment("test_separation", savedir=tmp_path_fixture)

    # Set global experiment data
    exp.log_global_parameter("algorithm_family", "metaheuristic")
    exp.log_global_parameter("dataset_name", "berlin52")
    exp.log_global_config(
        "experiment_config", {"researchers": ["Alice", "Bob"], "funding": "NSF-12345"}
    )

    # Create multiple runs with different parameters
    for i, temp in enumerate([100, 500, 1000]):
        run = exp.run()
        with run:
            run.log_parameter("temperature", temp)
            run.log_parameter("cooling_rate", 0.95)
            run.log_parameter("run_id", i)
            run.log_object("run_config", {"seed": i * 42})

    # Verify separation
    exp_store = exp.dataspace.experiment_datastore

    # Global data should be in experiment store
    assert exp_store.parameters["algorithm_family"] == "metaheuristic"
    assert exp_store.parameters["dataset_name"] == "berlin52"
    assert exp_store.objects["experiment_config"]["researchers"] == ["Alice", "Bob"]

    # Run data should be in individual run stores
    assert len(exp.runs) == 3
    for i, run_store in enumerate(exp.dataspace.run_datastores):
        expected_temp = [100, 500, 1000][i]
        assert run_store.parameters["temperature"] == expected_temp
        assert run_store.parameters["cooling_rate"] == 0.95
        assert run_store.parameters["run_id"] == i
        assert run_store.objects["run_config"]["seed"] == i * 42

    # Global data should NOT be in run stores
    for run_store in exp.dataspace.run_datastores:
        assert "algorithm_family" not in run_store.parameters
        assert "dataset_name" not in run_store.parameters
        assert "experiment_config" not in run_store.objects


def test_deprecation_warnings():
    """Test that old methods generate appropriate deprecation warnings."""
    exp = minto.Experiment("test_deprecation", auto_saving=False)

    # Test log_parameter raises error outside run context
    with pytest.raises(RuntimeError, match="can only be called within a run context"):
        exp.log_parameter("old_param", "value")

    # Test log_object raises error outside run context
    with pytest.raises(RuntimeError, match="can only be called within a run context"):
        exp.log_object("old_object", {"key": "value"})

    # Test log_problem raises error outside run context
    with pytest.raises(RuntimeError, match="can only be called within a run context"):
        problem = knapsack_problem()
        exp.log_problem("old_problem", problem)


def test_new_interface_no_warnings():
    """Test that new methods don't generate warnings."""
    exp = minto.Experiment("test_no_warnings", auto_saving=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # These should not generate warnings
        exp.log_global_parameter("param", "value")
        exp.log_global_config("config", {"key": "value"})
        problem = knapsack_problem()
        exp.log_global_problem("problem", problem)

        # Filter out non-deprecation warnings
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0


def test_backward_compatibility():
    """Test that old interface raises errors outside run context."""
    exp = minto.Experiment("test_backward_compatibility", auto_saving=False)

    # Old interface should raise errors outside run context
    with pytest.raises(RuntimeError, match="can only be called within a run context"):
        exp.log_parameter("old_style_param", 42)

    with pytest.raises(RuntimeError, match="can only be called within a run context"):
        exp.log_object("old_style_object", {"data": "test"})

    with pytest.raises(RuntimeError, match="can only be called within a run context"):
        problem = knapsack_problem()
        exp.log_problem("old_style_problem", problem)

    # Within run context, old interface should work with warnings
    run = exp.run()
    with run:
        # These should generate deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            exp.log_parameter("param_in_run", 100)
            exp.log_object("object_in_run", {"key": "value"})

            # Should have deprecation warnings
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 2
            assert "Calling log_parameter() on Experiment is deprecated" in str(
                deprecation_warnings[0].message
            )
            assert "Calling log_object() on Experiment is deprecated" in str(
                deprecation_warnings[1].message
            )


def test_save_load_with_global_interface(tmp_path_fixture):
    """Test save/load functionality with new global interface."""
    # Create experiment with global data
    exp1 = minto.Experiment("test_save_load", savedir=tmp_path_fixture)
    exp1.log_global_parameter("dataset", "test_dataset")
    exp1.log_global_config("metadata", {"version": "2.0"})
    problem = knapsack_problem()
    exp1.log_global_problem("test_problem", problem)

    # Add some run data
    run = exp1.run()
    with run:
        run.log_parameter("temperature", 500)

    # Save
    exp1.save()

    # Load
    exp2 = minto.Experiment.load_from_dir(tmp_path_fixture / exp1.experiment_name)

    # Verify global data preserved
    exp_store = exp2.dataspace.experiment_datastore
    assert exp_store.parameters["dataset"] == "test_dataset"
    assert exp_store.objects["metadata"]["version"] == "2.0"
    assert jm.is_same(exp_store.problems["test_problem"], problem)

    # Verify run data preserved
    assert len(exp2.runs) == 1
    assert exp2.dataspace.run_datastores[0].parameters["temperature"] == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
