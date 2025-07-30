import jijmodeling as jm
import numpy as np
import ommx.v1 as ommx_v1
import ommx_openjij_adapter as oj_ad
import ommx_pyscipopt_adapter as scip_ad
import pytest
from helpers import assert_instance_equal, assert_sampleset_equal, assert_solution_equal

import minto
import minto.problems.knapsack
import minto.problems.tsp


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


def test_experiment(tmp_path):
    exp = minto.Experiment(savedir=tmp_path)
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    exp.log_problem("knapsack", problem)
    exp.log_object("knapsack", data)
    exp.log_instance("knapsack", instance)

    num_iter = 3
    for i in range(num_iter):
        with exp.run():
            exp.log_parameter("value", i)
            exp.log_solution("knapsack", solution)

    exp2 = minto.Experiment.load_from_dir(tmp_path / exp.experiment_name)
    assert exp2.name == exp.name
    assert jm.is_same(exp2.dataspace.experiment_datastore.problems["knapsack"], problem)
    assert exp2.dataspace.experiment_datastore.objects["knapsack"] == data
    assert_instance_equal(
        exp2.dataspace.experiment_datastore.instances["knapsack"], instance
    )
    for i in range(num_iter):
        assert exp2.dataspace.run_datastores[i].parameters["value"] == i
        assert_solution_equal(
            exp2.dataspace.run_datastores[i].solutions["knapsack"], solution
        )


def test_experiment_concat(tmp_path):
    # generate test based on test_experiment
    exp = [minto.Experiment(savedir=tmp_path)] * 3
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    for i in range(3):
        exp[i].log_problem("knapsack", problem)
        exp[i].log_object("knapsack", data)
        exp[i].log_instance("knapsack", instance)

        num_iter = 3
        for i in range(num_iter):
            with exp[i].run():
                exp[i].log_parameter("value", i)
                exp[i].log_solution("knapsack", solution)

    concat_exp = minto.Experiment.concat(exp)
    assert jm.is_same(
        concat_exp.dataspace.experiment_datastore.problems["knapsack"], problem
    )
    assert concat_exp.dataspace.experiment_datastore.objects["knapsack"] == data
    assert_instance_equal(
        concat_exp.dataspace.experiment_datastore.instances["knapsack"], instance
    )
    run_id = 0
    for i in range(num_iter):
        assert concat_exp.dataspace.run_datastores[run_id].parameters["value"] == i
        assert_solution_equal(
            concat_exp.dataspace.run_datastores[run_id].solutions["knapsack"], solution
        )
        run_id += 1


def test_sampleset_for_openjij(tmp_path):
    tsp = minto.problems.tsp.QuadTSP()
    tsp_problem = tsp.problem()
    n = 8
    tsp_data = tsp.data(n)
    interpreter = jm.Interpreter(tsp_data)
    tsp_instance = interpreter.eval_problem(tsp_problem)

    parametric_qubo = tsp_instance.uniform_penalty_method()
    A = parametric_qubo.parameters[0]
    qubo = parametric_qubo.with_parameters({A.id: 1.0})
    samples = oj_ad.sample_qubo_sa(qubo, num_reads=10)
    sampleset = qubo.evaluate_samples(samples)

    experiment = minto.Experiment(savedir=tmp_path)
    experiment.log_sampleset("tsp", sampleset)

    assert experiment.dataspace.experiment_datastore.samplesets["tsp"] == sampleset

    with experiment.run():
        experiment.log_sampleset("tsp", sampleset)

    stored_sampleset = experiment.dataspace.run_datastores[0].samplesets["tsp"]
    assert_sampleset_equal(stored_sampleset, sampleset)

    loaded_exp = minto.Experiment.load_from_dir(tmp_path / experiment.experiment_name)
    loaded_sampleset = loaded_exp.dataspace.run_datastores[0].samplesets["tsp"]
    assert_sampleset_equal(loaded_sampleset, sampleset)

    loaded_exp.get_run_table()


# Define solver functions for testing
def dummy_solver(
    param1: int = 1, param2: float = 2.0, problem=None, unsupported_param=None
):
    return "a"


def dummy_instance():
    knapsack = minto.problems.knapsack.KnapsackProblem()
    problem = knapsack.problem()
    data = knapsack.random_data(3)
    instance = jm.Interpreter(data).eval_problem(problem)
    return instance


def solver_with_instance(instance: ommx_v1.Instance):
    adapter = scip_ad.OMMXPySCIPOptAdapter(instance)
    model = adapter.solver_input
    model.optimize()
    return adapter.decode(model)


@pytest.fixture
def experiment():
    return minto.Experiment("test_experiment", auto_saving=False)


def test_solver_name_logging(experiment):
    """Verify that solver name is logged correctly"""
    experiment.log_solver(dummy_solver)()
    assert (
        experiment.dataspace.experiment_datastore.parameters["solver_name"]
        == "dummy_solver"
    )


def test_parameter_logging(experiment):
    """Verify that numeric parameters are logged correctly"""
    test_param1 = 10
    test_param2 = 3.14
    experiment.log_solver(dummy_solver)(param1=test_param1, param2=test_param2)

    params = experiment.dataspace.experiment_datastore.parameters
    assert params["param1"] == test_param1
    assert params["param2"] == test_param2


def test_problem_logging(experiment):
    """Verify that Problem object is logged correctly"""
    problem = jm.Problem("test_problem")
    experiment.log_solver(dummy_solver)(problem=problem)

    problems = experiment.dataspace.experiment_datastore.problems
    assert "problem" in problems
    assert problems["problem"] == problem


def test_instance_logging(experiment):
    """Verify that Instance object is logged correctly"""
    instance = dummy_instance()

    experiment.log_solver(solver_with_instance)(instance=instance)

    instances = experiment.dataspace.experiment_datastore.instances
    assert "instance" in instances
    assert_instance_equal(instances["instance"], instance)

    solutions = experiment.dataspace.experiment_datastore.solutions
    assert "solver_with_instance_result" in solutions
    assert isinstance(solutions["solver_with_instance_result"], ommx_v1.Solution)


def test_exclude_params(experiment):
    """Verify that parameters specified in exclude_params are not logged"""
    experiment.log_solver(dummy_solver, exclude_params=["param1"])(
        param1=10, param2=3.14
    )

    params = experiment.dataspace.experiment_datastore.parameters
    assert "param1" not in params
    assert params["param2"] == 3.14


def test_with_run_context(experiment):
    """Verify logging within experiment run context"""
    with experiment:
        experiment.log_solver(dummy_solver)(param1=10)

        # Verify that parameters are logged to run datastore
        run_store = experiment.dataspace.run_datastores[0]
        assert run_store.parameters["param1"] == 10
        assert run_store.parameters["solver_name"] == "dummy_solver"


def test_invalid_parameter_types(experiment):
    """Verify that unsupported parameter types are ignored"""

    class UnsupportedType:
        pass

    experiment.log_solver(dummy_solver)(param1=10, unsupported_param=UnsupportedType())

    params = experiment.dataspace.experiment_datastore.parameters
    assert "unsupported_param" not in params
    assert params["param1"] == 10


def test_scalar_parameter_logging(experiment):
    """Test logging of scalar value parameters"""
    # Integer value
    experiment.log_parameter("int_param", 42)
    assert experiment.dataspace.experiment_datastore.parameters["int_param"] == 42

    # Floating point value
    experiment.log_parameter("float_param", 3.14)
    assert experiment.dataspace.experiment_datastore.parameters["float_param"] == 3.14

    # String value
    experiment.log_parameter("str_param", "test")
    assert experiment.dataspace.experiment_datastore.parameters["str_param"] == "test"


def test_list_parameter_logging(experiment):
    """Test logging of list parameters"""
    test_list = [1, 2, 3, "test", 4.5]
    experiment.log_parameter("list_param", test_list)

    # Verify saved as parameter
    assert (
        experiment.dataspace.experiment_datastore.parameters["list_param"] == test_list
    )

    # Verify also saved as object
    stored_object = experiment.dataspace.experiment_datastore.objects.get("list_param")
    assert stored_object["parameter_list_param"] == test_list


def test_dict_parameter_logging(experiment):
    """Test logging of dictionary parameters"""
    test_dict = {
        "key1": "value1",
        "key2": 42,
        "key3": [1, 2, 3],
        "key4": {"nested": "dict"},
    }
    experiment.log_parameter("dict_param", test_dict)

    # Verify saved as parameter
    assert (
        experiment.dataspace.experiment_datastore.parameters["dict_param"] == test_dict
    )

    # Verify also saved as object
    stored_object = experiment.dataspace.experiment_datastore.objects.get("dict_param")
    assert stored_object["parameter_dict_param"] == test_dict


def test_numpy_array_parameter_logging(experiment):
    """Test logging of NumPy array parameters"""
    test_array = np.array([1, 2, 3, 4, 5])
    experiment.log_parameter("array_param", test_array)

    # Verify saved as parameter
    assert np.array_equal(
        experiment.dataspace.experiment_datastore.parameters["array_param"], test_array
    )

    # Verify also saved as object
    stored_object = experiment.dataspace.experiment_datastore.objects.get("array_param")
    assert np.array_equal(stored_object["parameter_array_param"], test_array)


def test_nested_parameter_logging(experiment):
    """Test logging of nested data structures"""
    nested_data = {
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "array": np.array([1, 2, 3]),
        "mixed": [1, {"x": 2}, np.array([3, 4])],
    }
    experiment.log_parameter("nested_param", nested_data)

    # Verify saved as parameter
    stored_param = experiment.dataspace.experiment_datastore.parameters["nested_param"]
    assert stored_param["list"] == nested_data["list"]
    assert stored_param["dict"] == nested_data["dict"]
    assert np.array_equal(stored_param["array"], nested_data["array"])

    # Verify also saved as object
    stored_object = experiment.dataspace.experiment_datastore.objects.get(
        "nested_param"
    )
    assert stored_object["parameter_nested_param"] == nested_data


def test_non_serializable_parameter(experiment):
    """Test non-serializable parameters"""

    class NonSerializable:
        pass

    non_serializable_data = {"object": NonSerializable()}

    with pytest.raises(ValueError, match="Value is not serializable"):
        experiment.log_parameter("bad_param", non_serializable_data)


def test_parameter_logging_in_run_context(experiment):
    """Test parameter logging within experiment run context"""
    with experiment:
        # Scalar value
        experiment.log_parameter("scalar_param", 42)

        # List
        experiment.log_parameter("list_param", [1, 2, 3])

        # Dictionary
        experiment.log_parameter("dict_param", {"key": "value"})

        # NumPy array
        experiment.log_parameter("array_param", np.array([1, 2, 3]))

        # Verify saved to run datastore
        run_store = experiment.dataspace.run_datastores[0]

        assert run_store.parameters["scalar_param"] == 42
        assert run_store.parameters["list_param"] == [1, 2, 3]
        assert run_store.parameters["dict_param"] == {"key": "value"}
        assert np.array_equal(run_store.parameters["array_param"], np.array([1, 2, 3]))


def test_duplicate_parameter_names(experiment):
    """Test logging the same parameter name multiple times"""
    # First logging
    experiment.log_parameter("test_param", 1)
    assert experiment.dataspace.experiment_datastore.parameters["test_param"] == 1

    # Log different value with same name
    experiment.log_parameter("test_param", 2)
    assert experiment.dataspace.experiment_datastore.parameters["test_param"] == 2


def test_empty_containers(experiment):
    """Test logging of empty lists, dictionaries, and arrays"""
    # Empty list
    experiment.log_parameter("empty_list", [])
    assert experiment.dataspace.experiment_datastore.parameters["empty_list"] == []

    # Empty dictionary
    experiment.log_parameter("empty_dict", {})
    assert experiment.dataspace.experiment_datastore.parameters["empty_dict"] == {}

    # Empty array
    experiment.log_parameter("empty_array", np.array([]))
    assert np.array_equal(
        experiment.dataspace.experiment_datastore.parameters["empty_array"],
        np.array([]),
    )


def test_log_solver_with_name(experiment):
    """Test logging solver function with specified solver name"""
    experiment.log_solver("custom_solver", dummy_solver)()
    assert (
        experiment.dataspace.experiment_datastore.parameters["solver_name"]
        == "custom_solver"
    )
