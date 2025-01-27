from __future__ import annotations

import inspect
import pathlib
import types
import uuid
from typing import Any, Callable, Literal, Optional, TypedDict

import jijmodeling as jm
import pandas as pd
from jijzept.response import JijModelingResponse
from pandas import DataFrame

from minto.v0.consts.default import DEFAULT_RESULT_DIR
from minto.v0.records.records import (
    Index,
    ParameterContent,
    ParameterInfo,
    ResultContent,
    ResultInfo,
    SolverContent,
    SolverInfo,
)
from minto.v0.records.sampleset_expansion import expand_sampleset
from minto.v0.table.table import SchemaBasedTable


class DatabaseComponentSchema(TypedDict):
    info: SchemaBasedTable
    content: SchemaBasedTable


class DatabaseSchema(TypedDict):
    index: SchemaBasedTable
    solver: DatabaseComponentSchema
    parameter: DatabaseComponentSchema
    result: DatabaseComponentSchema


class Experiment:
    """
    Manage and track mathematical optimization experiments efficiently.

    This class is designed to simplify the process of managing and analyzing
    data from mathematical optimization experiments. It abstracts away the
    complexities associated with data logging, storage, and retrieval, making
    it easier to focus on experiment design and result interpretation. Users
    are encouraged to utilize the logging functions provided to capture
    comprehensive details about their experiments, thereby enhancing the
    reproducibility and accessibility of their experimental work.

    Args
    ----------
    name : str, optional
        The unique name of the experiment, automatically generated if not
        provided, to identify and differentiate it from others.
    savedir : str or pathlib.Path, optional
        The directory path for saving experiment data, including logs, solver
        configurations, parameters, and results. If not specified, a default
        directory is used.

    Examples
    --------
    Basic usage with manual parameter and result logging:
    >>> import minto
    >>> exp = minto.v0.Experiment(name="trial_01")
    >>> x = 2
    >>> y = x ** 2
    >>> with exp.run():
    ...     exp.log_parameter("x", x)
    ...     exp.log_result("y", y)
    >>> exp.table()
        experiment_name  run_id  x  y
    0          trial_01       0  2  4

    Logging in iterative processes:
    >>> exp = minto.v0.Experiment(name="trial_02")
    >>> for x in range(3):
    ...     y = x ** 2
    ...     with exp.run():
    ...         exp.log_parameter("x", x)
    ...         exp.log_result("y", y)
    >>> exp.table()
        experiment_name  run_id  x  y
    0          trial_02       0  0  0
    1          trial_02       1  1  1
    2          trial_02       2  2  4


    Integrating with optimization solvers and logging complex objects such as
    problems and samplesets:
    >>> import jijzept as jz
    >>> import jijmodeling as jm
    >>> problem = jm.Problem("test")
    >>> x = jm.BinaryVar("x", shape=(3,))
    >>> problem += x[:].sum()
    >>> problem += jm.Constraint("onehot", x[:].sum() == 1)
    >>> sampler = jz.JijSASampler(config="config.toml")
    >>> sampler_args = {"search": True, "num_search": 10}
    >>> sampleset = sampler.sample_model(problem, {}, **sampler_args)
    >>> exp = minto.v0.Experiment("trial_03")
    >>> with exp.run():
    ...     exp.log_parameter("problem", problem)
    ...     exp.log_parameters(sampler_args)
    ...     exp.log_solver("solver", sampler.sample_model)
    ...     exp.log_result("sampleset", sampleset)
    >>> exp.table()
    Output is a DataFrame with experiment results, including solver and sampleset details.

    Saving experiment data for future reference and reproducibility:
    >>> exp.save()
    """

    def __init__(
        self,
        name: Optional[str] = None,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ):
        self.name = name or str(uuid.uuid4())
        self.savedir = pathlib.Path(savedir)

        database: DatabaseSchema = {
            "index": SchemaBasedTable(Index.dtypes),
            "solver": {
                "info": SchemaBasedTable(SolverInfo.dtypes),
                "content": SchemaBasedTable(SolverContent.dtypes),
            },
            "parameter": {
                "info": SchemaBasedTable(ParameterInfo.dtypes),
                "content": SchemaBasedTable(ParameterContent.dtypes),
            },
            "result": {
                "info": SchemaBasedTable(ResultInfo.dtypes),
                "content": SchemaBasedTable(ResultContent.dtypes),
            },
        }
        object.__setattr__(self, "database", database)

    def __enter__(self) -> Experiment:
        self._mkdir()

        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        pass

    def run(self) -> Experiment:
        """
        Start the experiment and create a unique ID.
        This method updates the internal database of the experiment, creating
        a new record that records the experiment data. It creates an
        environment for recording parameters, solvers, and results.

        Returns
        -------
        Experiment
            Returns the Experiment instance.
        """
        database: DatabaseSchema = getattr(self, "database")

        if database["index"].empty():
            run_id = 0
        else:
            run_id = database["index"][-1].series()["run_id"] + 1
        database["index"].insert(
            Index(
                experiment_name=self.name,
                run_id=run_id,
                # TODO: New attribute will be added.
                # date=datetime.datetime.now()
            )
        )
        return self

    def table(
        self,
        key: Literal["solver", "parameter", "result"] | None = None,
        enable_sampleset_expansion: bool = True,
    ) -> pd.DataFrame:
        """
        Compiled the logged data and returned as a pandas DataFrame.

        Args
        ----------
        key : {'solver', 'parameter', 'result', None}, optional
            Specifies which part of the experiment data to return. If None,
            merges all available data into a single DataFrame.
        enable_sampleset_expansion : bool, default True
            Enables the expansion of SampleSet objects into tabular form, if
            present within the results data.

        Returns
        ----------
        DataFrame
            If no key is specified, a merged DataFrame of the entire
            experiment, or a partial DataFrame specified by the key.
        """
        database: DatabaseSchema = getattr(self, "database")

        solver_df = _get_component_dataframe(self, "solver")
        if key == "solver":
            return solver_df
        parameter_df = _get_component_dataframe(self, "parameter")
        if key == "parameter":
            return parameter_df
        result_df = _get_component_dataframe(self, "result")
        if key == "result":
            return result_df

        df = database["index"].dataframe()
        # Merge solver
        if not solver_df.empty:
            df = df.merge(
                _pivot(solver_df, columns="solver_name", values="source"),
                on=["experiment_name", "run_id"],
                how="outer",
            )

        # Merge parameter
        if not parameter_df.empty:
            df = df.merge(
                _pivot(parameter_df, columns="parameter_name", values="content"),
                on=["experiment_name", "run_id"],
                how="outer",
            )

        # Merge result
        if not result_df.empty:
            df = df.merge(
                _pivot(result_df, columns="result_name", values="content"),
                on=["experiment_name", "run_id"],
                how="outer",
            )

        # Expand sampleset
        if enable_sampleset_expansion:
            sampleset_df = expand_sampleset(database["result"]["content"].dataframe())
            if not sampleset_df.empty:
                sampleset_df = pd.merge(
                    database["result"]["info"].dataframe()[
                        ["experiment_name", "run_id", "result_id"]
                    ],
                    sampleset_df,
                    on="result_id",
                    how="inner",
                ).drop(columns="result_id")

                result_names = [
                    name
                    for name in result_df["result_name"].unique()
                    if isinstance(
                        result_df[result_df["result_name"] == name]["content"].iloc[0],
                        (jm.experimental.SampleSet, jm.SampleSet, JijModelingResponse),
                    )
                ]

                df = df.merge(sampleset_df, on=["experiment_name", "run_id"]).drop(
                    columns=result_names
                )
        return df

    def log_solver(self, name: str, solver: Callable[..., Any]) -> None:
        """
        Log data about the solver used in the experiment.

        Parameters
        ----------
        name : str
            The name assigned to the solver for identification.
        solver : Callable[..., Any]
            The solver object to be logged.
        """
        database: DatabaseSchema = getattr(self, "database")

        run_id = int(database["index"][-1].series()["run_id"])
        solver_id = len(database["solver"]["info"])

        if isinstance(solver, types.FunctionType):
            source = inspect.getfile(solver)
        else:
            if _is_running_in_notebook():
                source = "Dynamically generated in Jupyter Notebook"
            else:
                if isinstance(solver, types.MethodType):
                    source = inspect.getfile(solver)
                else:
                    source = inspect.getfile(solver.__class__)

        info = SolverInfo(
            experiment_name=self.name,
            run_id=run_id,
            solver_name=name,
            source=source,
            solver_id=solver_id,
        )
        content = SolverContent(solver_id=solver_id, content=solver)

        database["solver"]["info"].insert(info)
        database["solver"]["content"].insert(content)

    def log_solvers(self, solvers: dict[str, Callable[..., Any]]) -> None:
        """
        Logs multiple solvers at once.

        Parameters
        ----------
        solvers : dict[str, Callable[..., Any]]
            A dictionary where keys are solver names and values are the solver
            objects.
        """
        for name, solver in solvers.items():
            self.log_solver(name, solver)

    def log_parameter(self, name: str, parameter: Any) -> None:
        """
        Log a single parameter used in the experiment.

        Parameters
        ----------
        name : str
            The name assigned to the parameter for identification.
        parameter : Any
            The value of the parameter to be logged.
        """
        database: DatabaseSchema = getattr(self, "database")

        run_id = int(database["index"][-1].series()["run_id"])
        parameter_id = len(database["parameter"]["info"])

        info = ParameterInfo(
            experiment_name=self.name,
            run_id=run_id,
            parameter_name=name,
            parameter_id=parameter_id,
        )
        content = ParameterContent(parameter_id=parameter_id, content=parameter)

        database["parameter"]["info"].insert(info)
        database["parameter"]["content"].insert(content)

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Logs multiple parameters at once.

        Parameters
        ----------
        parameters : dict[str, Any]
            A dictionary where keys are parameter names and values are the
            parameter values to be logged.
        """
        for name, parameter in parameters.items():
            self.log_parameter(name, parameter)

    def log_result(self, name: str, result: Any) -> None:
        """
        Log a single result from the experiment.

        Parameters
        ----------
        name : str
            The name assigned to the result for identification.
        result : Any
            The data or outcome to be logged as a result.
        """
        database: DatabaseSchema = getattr(self, "database")

        run_id = int(database["index"][-1].series()["run_id"])
        result_id = len(database["result"]["info"])

        info = ResultInfo(
            experiment_name=self.name,
            run_id=run_id,
            result_name=name,
            result_id=result_id,
        )
        content = ResultContent(result_id=result_id, content=result)

        database["result"]["info"].insert(info)
        database["result"]["content"].insert(content)

    def log_results(self, results: dict[str, Any]) -> None:
        """
        Logs multiple results at once.

        Parameters
        ----------
        results : dict[str, Any]
            A dictionary where keys are result names and values are the
            data or outcomes to be logged.
        """
        for name, result in results.items():
            self.log_result(name, result)

    def save(self) -> None:
        """
        Writes out all log data for parameters, solvers, and results. The data
        is saved under "savedir / experiment.name" directory.
        """
        from minto.v0.io.save import save

        save(self)

    def _mkdir(self) -> None:
        for key in ["solver", "parameter", "result"]:
            d = self.savedir / self.name / key
            d.mkdir(parents=True, exist_ok=True)

            if key in ["parameter", "result"]:
                problem_dir = d / "problems"
                problem_dir.mkdir(parents=True, exist_ok=True)

                sampleset_dir = d / "samplesets"
                sampleset_dir.mkdir(parents=True, exist_ok=True)

                dataclass_dir = d / "dataclasses"
                dataclass_dir.mkdir(parents=True, exist_ok=True)


def _get_component_dataframe(
    experiment: Experiment, key: Literal["solver", "parameter", "result"]
) -> DataFrame:
    database: DatabaseSchema = getattr(experiment, "database")

    return pd.merge(
        database[key]["info"].dataframe(),
        database[key]["content"].dataframe(),
        on=f"{key}_id",
    )


def _pivot(df: DataFrame, columns: str | list[str], values: str) -> DataFrame:
    return df.pivot_table(
        index=["experiment_name", "run_id"],
        columns=columns,
        values=values,
        aggfunc=lambda x: x,
        dropna=False,
    ).reset_index()


def _is_running_in_notebook():
    try:
        ipython = get_ipython()
        # Jupyter Notebook or JupyterLab
        if "IPKernelApp" in ipython.config:
            return True
    except NameError:
        return False
