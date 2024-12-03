from __future__ import annotations

import inspect
import shutil

import jijmodeling as jm
import pytest

import minto
from minto.consts.default import DEFAULT_RESULT_DIR


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    if DEFAULT_RESULT_DIR.exists():
        shutil.rmtree(DEFAULT_RESULT_DIR)


def square(i: int) -> int:
    return i**2


def test_get_table():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    i = 2
    result = square(i)

    with exp.run():
        exp.log_solver("function", square)
        exp.log_parameter("i", i)
        exp.log_result("result", result)

    table = exp.table()

    columns = ["experiment_name", "run_id", "function", "i", "result"]
    expected_values = ["example", 0, inspect.getfile(square), 2, 4]

    assert set(columns) == set(table.columns)
    for column, expected_value in zip(columns, expected_values):
        assert table.loc[0, column] == expected_value


def test_get_table_containing_sampleset(jm_sampleset: jm.SampleSet):
    sampleset = jm.experimental.from_old_sampleset(jm_sampleset)

    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)
    with exp.run():
        exp.log_result("sampleset", sampleset)

    table = exp.table()

    columns = [
        "experiment_name",
        "run_id",
        "sample_run_id",
        "num_occurrences",
        "objective",
        "is_feasible",
        "sample_id",
        "deci_var_value",
        "eval_result",
        "onehot1_total_violation",
        "onehot2_total_violation",
    ]
    assert set(columns) == set(table.columns)


def test_get_solver_table():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    with exp.run():
        exp.log_solver("function", square)

    table = exp.table(key="solver")

    assert table["content"][0] == square


def test_get_parameter_table():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    i = 2

    with exp.run():
        exp.log_parameter("i", i)

    table = exp.table(key="parameter")

    assert table["content"][0] == i


def test_get_result_table():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    i = 2
    result = square(i)

    with exp.run():
        exp.log_result("result", result)

    table = exp.table(key="result")

    assert table["content"][0] == result
