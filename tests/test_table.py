import pathlib

import jijmodeling as jm
import ommx.artifact as ox_art
import ommx_pyscipopt_adapter as scip_ad
import pandas as pd
import pytest

from minto.table import create_table, create_table_from_stores
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


def test_datastore_table():
    ds = DataStore({}, {}, {}, {}, {}, {})
    problem = knapsack()
    data = knapsack_data()
    instance = knapsack_instance()
    solution = knapsack_solution()

    ds.add("knapsack", problem, "problems", with_save=False)
    ds.add("knapsack", data, "objects", with_save=False)
    ds.add("knapsack", instance, "instances", with_save=False)
    ds.add("value", 0.2, "parameters", with_save=False)
    ds.add("meta", "meta", "meta_data", with_save=False)
    ds.add("knapsack", solution, "solutions", with_save=False)

    tables = create_table(datastore=ds)

    table = create_table_from_stores(datastores=[ds])
    assert isinstance(table, pd.DataFrame)
    assert isinstance(table.columns, pd.MultiIndex)
    columns = [
        ("instance_knapsack", "num_vars"),
        ("instance_knapsack", "num_binary"),
        ("instance_knapsack", "num_integer"),
        ("instance_knapsack", "num_continuous"),
        ("metadata", "meta"),
        ("parameter", "value"),
        ("solution_knapsack", "objective"),
    ]
    for col in columns:
        assert col in table.columns
