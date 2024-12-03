from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass
from typing import Any, Callable

import jijmodeling as jm
import jijzept as jz
import pytest

import minto

SAVEDIR = "./.minto_experiments"


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    p = pathlib.Path(SAVEDIR)
    p.mkdir(exist_ok=True)
    yield
    # postprocess
    p = pathlib.Path(SAVEDIR)
    if p.exists():
        shutil.rmtree(p)


class F:
    def __call__(self):
        return "Hello, World!"


class Model:
    def f(self):
        return "Hello, World!"


def f() -> str:
    return "Hello, World!"


@dataclass
class A:
    x: int
    y: float


@pytest.mark.parametrize("solver", [None, f, F(), Model().f])
@pytest.mark.parametrize("parameter", [None, 1, 2.0, "a", A(1, 2.0)])
@pytest.mark.parametrize("result", [None, 1, 2.0, "a", A(1, 2.0)])
def test_load_experiment(
    solver: Callable[..., Any] | None, parameter: Any, result: Any
):
    exp = minto.Experiment("my_experiment", savedir=SAVEDIR)
    with exp.run():
        if solver is not None:
            exp.log_solver("solver", solver)

        if parameter is not None:
            exp.log_parameter("parameter", parameter)

        if result is not None:
            exp.log_result("result", result)
    exp.save()

    loaded_exp = minto.load("my_experiment", savedir=SAVEDIR)

    assert set(exp.table().columns) == set(loaded_exp.table().columns)


def test_load_experiment_for_jijzept(jm_sampleset: jm.SampleSet):
    problem = jm.Problem("example_problem")
    x = jm.BinaryVar("x", shape=(5,))
    i = jm.Element("i", 5)
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)
    sampleset = jm.experimental.from_old_sampleset(jm_sampleset)
    sampler = jz.JijSASampler(config="tests/io/config.toml")

    exp = minto.Experiment("example", savedir=SAVEDIR)
    with exp.run():
        exp.log_solver("sampler", sampler.sample_model)
        exp.log_parameter("problem", problem)
        exp.log_parameter("num_reads", 10)
        exp.log_result("sampleset", sampleset)

    exp.save()
    loaded_exp = minto.load("example", savedir=SAVEDIR)
    assert set(exp.table().columns) == set(loaded_exp.table().columns)
