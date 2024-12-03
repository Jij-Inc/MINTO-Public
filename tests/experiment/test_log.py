from __future__ import annotations

import shutil
from typing import Any, Literal

import jijmodeling as jm
import pytest

import minto
from minto.consts.default import DEFAULT_RESULT_DIR
from minto.records.sampleset_expansion import SampleSetRecord


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    if DEFAULT_RESULT_DIR.exists():
        shutil.rmtree(DEFAULT_RESULT_DIR)


def f():
    return "Hello, World!"


class F:
    def __call__(self):
        return "Hello, World!"


class Model:
    def f(self):
        return "Hello, World!"


def assert_log(
    exp: minto.Experiment,
    key: Literal["solver", "parameter", "result"],
    run_id: int,
    name: str,
    value: Any,
) -> None:
    key_df = exp.table(key=key)
    key_df = key_df[key_df[f"{key}_name"] == name].reset_index(drop=True)

    assert name == key_df[key_df["run_id"] == run_id].loc[0, f"{key}_name"]
    assert value == key_df[key_df["run_id"] == run_id].loc[0, "content"]

    df = exp.table()

    assert name in df.columns
    if key == "solver":
        value = key_df[key_df["run_id"] == run_id].loc[0, "source"]
    assert value == df[df["run_id"] == run_id].reset_index().loc[0, name]


def assert_log_for_sampleset(
    exp: minto.Experiment,
    key: Literal["solver", "parameter", "result"],
    run_id: int,
    name: str,
    value: Any,
) -> None:
    if key == "parameter":
        assert_log(exp=exp, key=key, run_id=run_id, name=name, value=value)
    elif key == "result":
        df = exp.table()
        for k in SampleSetRecord.dtypes:
            assert k in df


def test_log_solver():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    with exp.run():
        exp.log_solver("f1", f)

    assert_log(exp=exp, key="solver", run_id=0, name="f1", value=f)


def test_log_parameter():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    with exp.run():
        exp.log_parameter("x", 1)

    assert_log(exp=exp, key="parameter", run_id=0, name="x", value=1)


def test_log_result():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    result = f()

    with exp.run():
        exp.log_result("f1_result", result)

    assert_log(exp=exp, key="result", run_id=0, name="f1_result", value=result)


def test_log_solver_for_callable_object():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    f = F()

    with exp.run():
        exp.log_solver("f1", f)

    assert_log(exp=exp, key="solver", run_id=0, name="f1", value=f)


def test_log_parameter_for_sampleset(jm_sampleset: jm.SampleSet):
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    with exp.run():
        exp.log_parameter("x", 1)
        exp.log_parameter("sampleset", jm_sampleset)

    df = exp.table()
    assert len(df) == 1

    assert_log(exp=exp, key="parameter", run_id=0, name="x", value=1)
    assert_log_for_sampleset(
        exp=exp, key="parameter", run_id=0, name="sampleset", value=jm_sampleset
    )


def test_log_result_for_sampleset(jm_sampleset: jm.SampleSet):
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    with exp.run():
        exp.log_result("result", 1)
        exp.log_result("sampleset", jm_sampleset)

    n_samples = len(jm_sampleset.evaluation.objective)
    df = exp.table()
    assert len(df) == n_samples

    assert_log(exp=exp, key="result", run_id=0, name="result", value=1)
    assert_log_for_sampleset(
        exp=exp, key="result", run_id=0, name="sampleset", value=jm_sampleset
    )


def test_log_for_different_contexts():
    exp = minto.Experiment("example", savedir=DEFAULT_RESULT_DIR)

    with exp.run():
        exp.log_solver("f1", f)

    with exp.run():
        exp.log_parameter("x", 1)

    with exp.run():
        exp.log_result("f1_result", f())

    df = exp.table()
    assert len(df) == 3

    assert_log(exp=exp, key="solver", run_id=0, name="f1", value=f)
    assert_log(exp=exp, key="parameter", run_id=1, name="x", value=1)
    assert_log(exp=exp, key="result", run_id=2, name="f1_result", value=f())
