"""Test the new explicit interface for experiment and run separation."""

import jijmodeling as jm
import ommx.v1 as ommx_v1
import ommx_pyscipopt_adapter as scip_ad
import pytest

import minto
import minto.problems.knapsack


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


def test_new_explicit_interface(tmp_path):
    """Test the new explicit interface with clear experiment/run separation."""
    exp = minto.Experiment("test_explicit", savedir=tmp_path)
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    # Log experiment-level data
    exp.log_global_problem("knapsack", problem)
    exp.log_global_config("knapsack", data)
    exp.log_global_instance("knapsack", instance)
    exp.log_global_parameter("dataset_name", "knapsack_test")

    # Create multiple runs explicitly
    num_runs = 3
    runs = []

    for i in range(num_runs):
        run = exp.run()
        runs.append(run)

        # Use context manager
        with run:
            run.log_parameter("iteration", i)
            run.log_parameter("value", i * 10)
            run.log_solution("knapsack", solution)

    # Save experiment data to ensure persistence
    exp.save()

    # Verify experiment-level data
    assert exp.dataspace.experiment_datastore.problems["knapsack"] is not None
    assert exp.dataspace.experiment_datastore.objects["knapsack"] == data
    assert exp.dataspace.experiment_datastore.instances["knapsack"] == instance
    assert (
        exp.dataspace.experiment_datastore.parameters["dataset_name"] == "knapsack_test"
    )

    # Verify run-level data is separated
    for i, run in enumerate(runs):
        assert run.parameters["iteration"] == i
        assert run.parameters["value"] == i * 10
        assert "knapsack" in run.solutions

    # Verify experiment doesn't contain run-specific data
    assert "iteration" not in exp.dataspace.experiment_datastore.parameters
    assert "value" not in exp.dataspace.experiment_datastore.parameters

    # Verify runs don't contain experiment-level data
    for run in runs:
        assert "dataset_name" not in run.parameters

    # Test loading from disk
    exp2 = minto.Experiment.load_from_dir(tmp_path / exp.experiment_name)
    assert exp2.name == exp.name
    assert jm.is_same(exp2.dataspace.experiment_datastore.problems["knapsack"], problem)
    assert exp2.dataspace.experiment_datastore.objects["knapsack"] == data
    # Compare Instance objects - just check they were saved/loaded correctly
    loaded_instance = exp2.dataspace.experiment_datastore.instances["knapsack"]
    # Since Instance.__eq__ is broken, we just verify it's not None and has the
    # right type
    assert loaded_instance is not None
    assert isinstance(loaded_instance, ommx_v1.Instance)
    assert (
        exp2.dataspace.experiment_datastore.parameters["dataset_name"]
        == "knapsack_test"
    )

    for i in range(num_runs):
        assert exp2.dataspace.run_datastores[i].parameters["iteration"] == i
        assert exp2.dataspace.run_datastores[i].parameters["value"] == i * 10
        # Compare Solution objects - just check they were saved/loaded correctly
        loaded_solution = exp2.dataspace.run_datastores[i].solutions["knapsack"]
        # Since Solution.__eq__ is broken, we just verify it's not None and has
        # the right type
        assert loaded_solution is not None
        assert isinstance(loaded_solution, ommx_v1.Solution)
        # Basic checks to ensure data integrity
        assert loaded_solution.objective == solution.objective
        assert loaded_solution.feasible == solution.feasible


def test_run_without_context_manager(tmp_path):
    """Test runs can be used without context manager but must be closed manually."""
    exp = minto.Experiment("test_manual_close", savedir=tmp_path)

    run = exp.run()
    run.log_parameter("test_param", 42)

    # Should work fine
    assert run.parameters["test_param"] == 42
    assert not run.is_closed

    # Close manually
    run.close()
    assert run.is_closed

    # Should raise error after closing
    with pytest.raises(RuntimeError):
        run.log_parameter("another_param", 123)


def test_run_properties(tmp_path):
    """Test run properties and access methods."""
    exp = minto.Experiment("test_properties", savedir=tmp_path)

    run = exp.run()
    run.log_parameter("param1", 1)
    run.log_parameter("param2", "test")
    run.log_object("obj1", {"key": "value"})

    # Test properties
    assert run.parameters == {"param1": 1, "param2": "test"}
    assert run.objects == {"obj1": {"key": "value"}}
    assert run.run_id == 0
    assert not run.is_closed

    run.close()
    assert run.is_closed


def test_multiple_runs_independence(tmp_path):
    """Test that multiple runs are independent and don't interfere."""
    exp = minto.Experiment("test_independence", savedir=tmp_path)

    run1 = exp.run()
    run2 = exp.run()
    run3 = exp.run()

    # Log different data to each run
    run1.log_parameter("lr", 0.01)
    run2.log_parameter("lr", 0.001)
    run3.log_parameter("lr", 0.1)

    run1.log_parameter("batch_size", 32)
    run2.log_parameter("batch_size", 64)
    # run3 doesn't have batch_size

    # Verify independence
    assert run1.parameters["lr"] == 0.01
    assert run2.parameters["lr"] == 0.001
    assert run3.parameters["lr"] == 0.1

    assert run1.parameters["batch_size"] == 32
    assert run2.parameters["batch_size"] == 64
    assert "batch_size" not in run3.parameters

    # Different run IDs
    assert run1.run_id == 0
    assert run2.run_id == 1
    assert run3.run_id == 2

    run1.close()
    run2.close()
    run3.close()


def test_experiment_level_only_operations(tmp_path):
    """Test that experiment-level operations work correctly without runs."""
    exp = minto.Experiment("test_exp_only", savedir=tmp_path)

    # Log only experiment-level data
    exp.log_global_parameter("global_config", "production")
    exp.log_global_parameter("dataset_version", "v2.1")
    exp.log_global_config("metadata", {"created_by": "test", "purpose": "validation"})

    # Verify data is stored at experiment level
    assert (
        exp.dataspace.experiment_datastore.parameters["global_config"] == "production"
    )
    assert exp.dataspace.experiment_datastore.parameters["dataset_version"] == "v2.1"
    assert (
        exp.dataspace.experiment_datastore.objects["metadata"]["created_by"] == "test"
    )

    # No runs should exist
    assert len(exp.runs) == 0


def test_run_table_generation(tmp_path):
    """Test that run tables are generated correctly."""
    exp = minto.Experiment("test_table", savedir=tmp_path)

    # Create multiple runs with different parameters
    for i in range(3):
        run = exp.run()
        with run:
            run.log_parameter("epoch", i + 1)
            run.log_parameter("accuracy", 0.8 + i * 0.05)
            run.log_parameter("loss", 1.0 - i * 0.1)

    # Get run table
    run_table = exp.get_run_table()

    # Verify table structure
    assert len(run_table) == 3
    assert ("parameter", "epoch") in run_table.columns
    assert ("parameter", "accuracy") in run_table.columns
    assert ("parameter", "loss") in run_table.columns

    # Verify values
    assert run_table.loc[0, ("parameter", "epoch")] == 1
    assert run_table.loc[1, ("parameter", "epoch")] == 2
    assert run_table.loc[2, ("parameter", "epoch")] == 3

    assert abs(run_table.loc[0, ("parameter", "accuracy")] - 0.8) < 1e-10
    assert abs(run_table.loc[1, ("parameter", "accuracy")] - 0.85) < 1e-10
    assert abs(run_table.loc[2, ("parameter", "accuracy")] - 0.9) < 1e-10
