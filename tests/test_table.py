import jijmodeling as jm
import ommx_pyscipopt_adapter as scip_ad
import pandas as pd

from minto.table import (
    _extract_instance_info,
    _extract_sampleset_info,
    _extract_solution_info,
    create_table,
    create_table_from_stores,
)
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


def test_datastore_table():
    ds = DataStore({}, {}, {}, {}, {}, {})
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()
    sampleset = knapsack_sampleset()

    ds.add("knapsack", problem, "problems", with_save=False)
    ds.add("knapsack", data, "objects", with_save=False)
    ds.add("knapsack", instance, "instances", with_save=False)
    ds.add("value", 0.2, "parameters", with_save=False)
    ds.add("meta", "meta", "meta_data", with_save=False)
    ds.add("knapsack", solution, "solutions", with_save=False)
    ds.add("knapsack", sampleset, "samplesets", with_save=False)

    create_table(datastore=ds)

    table = create_table_from_stores(datastores=[ds])
    assert isinstance(table, pd.DataFrame)
    assert isinstance(table.columns, pd.MultiIndex)

    # Check instance columns
    instance_columns = [
        ("instance_knapsack", "num_vars"),
        ("instance_knapsack", "num_binary"),
        ("instance_knapsack", "num_integer"),
        ("instance_knapsack", "num_continuous"),
    ]
    for col in instance_columns:
        assert col in table.columns

    # Check solution columns
    solution_columns = [
        ("solution_knapsack", "objective"),
        ("solution_knapsack", "feasible"),
        ("solution_knapsack", "optimality"),
        ("solution_knapsack", "relaxation"),
        ("solution_knapsack", "start"),
    ]
    for col in solution_columns:
        assert col in table.columns

    # Check sampleset columns
    sampleset_columns = [
        ("sampleset_knapsack", "num_samples"),
        ("sampleset_knapsack", "obj_mean"),
        ("sampleset_knapsack", "obj_std"),
        ("sampleset_knapsack", "obj_min"),
        ("sampleset_knapsack", "obj_max"),
        ("sampleset_knapsack", "feasible"),
    ]
    for col in sampleset_columns:
        assert col in table.columns

    # Check metadata and parameter columns
    other_columns = [
        ("metadata", "meta"),
        ("parameter", "value"),
    ]
    for col in other_columns:
        assert col in table.columns


def test_extract_instance_info():
    """Test for _extract_instance_info function"""
    instance = knapsack_instance()
    instance_data = _extract_instance_info(instance)

    assert instance_data["num_vars"] == 5
    assert instance_data["num_binary"] == 5
    assert instance_data["num_integer"] == 0
    assert instance_data["num_continuous"] == 0


def test_extract_solution_info():
    """Test for _extract_solution_info function"""
    solution = knapsack_solution()
    solution_data = _extract_solution_info(solution)

    # Expected objective function value for knapsack problem
    # (optimal solution objective value)
    expected_objective = 6.0
    assert solution_data["objective"] == expected_objective
    assert solution_data["feasible"]  # True
    assert solution_data["optimality"] == 1  # Optimal solution found
    assert solution_data["relaxation"] == 0  # No relaxation


def test_extract_sampleset_info():
    """Test for _extract_sampleset_info function"""
    sampleset = knapsack_sampleset()
    sampleset_data = _extract_sampleset_info(sampleset)

    # Expected values based on knapsack_sampleset() samples
    # Sample objective values: [7, 5, 5, 3, 5] (calculated with v=[1,2,3,4,5])
    assert sampleset_data["num_samples"] == 5
    assert sampleset_data["obj_mean"] == 5.0
    assert sampleset_data["obj_min"] == 3.0
    assert sampleset_data["obj_max"] == 7.0
    assert sampleset_data["feasible"] == 4  # 4 out of 5 samples are feasible
