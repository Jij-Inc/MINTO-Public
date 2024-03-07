from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass
from typing import Any, Callable

import jijmodeling as jm
import jijzept as jz
import pytest

import minto

SAVEDIR = "./jb_results"


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


def assert_save_path_exists(exp: minto.Experiment) -> None:
    base_dir = exp.savedir / exp.name

    assert (base_dir / "dtypes.json").exists()
    assert (base_dir / "index.json").exists()

    for key in ["solver", "parameter", "result"]:
        key_dir = base_dir / key

        assert key_dir.exists()
        assert (key_dir / "info.json").exists()
        assert (key_dir / "content.csv").exists()

        if key in ["parameter", "result"]:
            problem_dir = key_dir / "problems"
            assert problem_dir.exists()

            sampleset_dir = key_dir / "samplesets"
            assert sampleset_dir.exists()

            dataclass_dir = key_dir / "dataclasses"
            assert dataclass_dir.exists()


@pytest.mark.parametrize("solver", [None, f, F(), Model().f])
@pytest.mark.parametrize("parameter", [None, 1, 2.0, "a", A(1, 2.0)])
@pytest.mark.parametrize("result", [None, 1, 2.0, "a", A(1, 2.0)])
def test_save_experiment(
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

    assert_save_path_exists(exp)


def test_save_experiment_for_jijzept(jm_sampleset: jm.SampleSet):
    problem = jm.Problem("example_problem")
    x = jm.BinaryVar("x", shape=(5,))
    i = jm.Element("i", 5)
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)
    sampleset = jm.experimental.from_old_sampleset(jm_sampleset)
    sampler = jz.JijSASampler(config="tests/io/config.toml")

    exp = minto.Experiment("sa_sampler_test", savedir=SAVEDIR)
    with exp.run():
        exp.log_solver("sampler", sampler.sample_model)
        exp.log_parameter("problem", problem)
        exp.log_parameter("num_reads", 10)
        exp.log_result("sampleset", sampleset)

    exp.save()

    assert_save_path_exists(exp)
