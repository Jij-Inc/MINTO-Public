"""Module for managing experiment runs.

This module provides the `Run` class for managing individual experiment runs.
The `Run` class allows users to log parameters, solutions, samplesets, and other
run-specific data to a database.
"""

import typing as typ
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

import jijmodeling as jm
import numpy as np
import ommx.v1 as ommx_v1

from .utils import type_check
from .v1.datastore import DataStore

P = typ.ParamSpec("P")
R = typ.TypeVar("R")


@dataclass
class Run:
    """Class to manage individual experiment runs.

    This class provides a structured way to manage run-specific data within
    an optimization experiment, including logging of parameters, solutions,
    samplesets, and other run-specific data.

    Attributes:
        _datastore (DataStore): The underlying data storage for this run.
        _experiment: Reference to the parent experiment.
        _run_id (int): Unique identifier for this run.
        _start_time (datetime): Start time of the run.
        _closed (bool): Flag indicating if the run has been closed.
    """

    _datastore: DataStore = field(default_factory=DataStore, init=False)
    _experiment: typ.Optional[typ.Any] = field(
        default=None, init=False
    )  # Forward reference to avoid circular import
    _run_id: int = field(default=0, init=False)
    _start_time: datetime = field(default_factory=datetime.now, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self):
        """Post-initialization method for Run class."""
        self._start_time = datetime.now()

    def _check_not_closed(self):
        """Check if the run is still active (not closed)."""
        if self._closed:
            raise RuntimeError(
                "This run has been closed. Create a new run to log data."
            )

    def log_parameter(
        self,
        name: str,
        value: float | int | str | list | dict | np.ndarray,
    ):
        """Log a parameter to this run.

        Args:
            name (str): Name of the parameter.
            value: Value of the parameter. Can be scalar or complex data structure.

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()

        # Log to console if experiment has verbose logging enabled
        if self._experiment and hasattr(self._experiment, "_logger"):
            self._experiment._logger.log_parameter(name, value)

        # Handle complex data structures
        if isinstance(value, (list, dict, np.ndarray)):
            # Check value is serializable
            try:
                import json

                from minto.v1.json_encoder import NumpyEncoder

                json.dumps(value, cls=NumpyEncoder)
            except TypeError:
                raise ValueError("Value is not serializable.")

            # Store both as parameter and object
            self.log_object(name, {"parameter_" + name: value})

        # Store the parameter
        self._datastore.add(name, value, "parameters")

    def log_params(self, params: dict[str, float | int | str]):
        """Log multiple parameters to this run.

        Args:
            params: Dictionary of parameter names and values.

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()
        for name, value in params.items():
            self.log_parameter(name, value)

    def log_object(self, name: str, value: dict[str, typ.Any]):
        """Log a custom object to this run.

        Args:
            name: Name of the object
            value: Dictionary containing the object data

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()
        type_check([("name", name, str), ("value", value, dict)])

        # Log to console if experiment has verbose logging enabled
        if self._experiment and hasattr(self._experiment, "_logger"):
            size_info = f"{len(value)} keys" if isinstance(value, dict) else ""
            self._experiment._logger.log_object(name, "dict", size_info)

        self._datastore.add(name, value, "objects")

    def log_problem(
        self, problem_name: str | jm.Problem, problem: typ.Optional[jm.Problem] = None
    ):
        """Log an optimization problem to this run.

        Args:
            problem_name: Name of the problem, or the problem object itself.
            problem: Problem object (if problem_name is a string).

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()
        _name, _problem = self._get_name_or_default(
            problem_name, problem, self._datastore.problems
        )

        # Log to console if experiment has verbose logging enabled
        if self._experiment and hasattr(self._experiment, "_logger"):
            self._experiment._logger.log_object(_name, "jm.Problem", "problem logged")

        self._datastore.add(_name, _problem, "problems")

    def log_instance(
        self,
        instance_name: str | ommx_v1.Instance,
        instance: typ.Optional[ommx_v1.Instance] = None,
    ):
        """Log an optimization problem instance to this run.

        Args:
            instance_name: Name of the instance, or the instance object itself.
            instance: Instance object (if instance_name is a string).

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()
        _name, _instance = self._get_name_or_default(
            instance_name, instance, self._datastore.instances
        )

        # Log to console if experiment has verbose logging enabled
        if self._experiment and hasattr(self._experiment, "_logger"):
            self._experiment._logger.log_object(
                _name, "ommx.Instance", "instance logged"
            )

        self._datastore.add(_name, _instance, "instances")

    def log_solution(
        self,
        solution_name: str | ommx_v1.Solution,
        solution: typ.Optional[ommx_v1.Solution] = None,
    ):
        """Log an optimization solution to this run.

        Args:
            solution_name: Name of the solution, or the solution object itself.
            solution: Solution object (if solution_name is a string).

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()
        _name, _solution = self._get_name_or_default(
            solution_name, solution, self._datastore.solutions
        )

        # Log to console if experiment has verbose logging enabled
        if self._experiment and hasattr(self._experiment, "_logger"):
            # Extract basic info from solution for logging
            if hasattr(_solution, "objective") and hasattr(_solution, "feasible"):
                info = (
                    f"objective: {_solution.objective:.3f}, "
                    f"feasible: {_solution.feasible}"
                )
            else:
                info = "solution logged"
            self._experiment._logger.log_solution(_name, info)

        self._datastore.add(_name, _solution, "solutions")

    def log_sampleset(
        self,
        name: str | ommx_v1.SampleSet,
        value: typ.Optional[ommx_v1.SampleSet] = None,
    ):
        """Log a SampleSet to this run.

        Args:
            name: Name of the sampleset, or the sampleset object itself.
            value: SampleSet object (if name is a string).

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()
        _name, _value = self._get_name_or_default(
            name, value, self._datastore.samplesets
        )

        if isinstance(_value, (jm.SampleSet, jm.experimental.SampleSet)):
            raise ValueError(
                "JijModeling SampleSet is no longer supported. "
                "Please use OMMX SampleSet instead."
            )

        # Log to console if experiment has verbose logging enabled
        if self._experiment and hasattr(self._experiment, "_logger"):
            num_samples = len(_value.samples) if hasattr(_value, "samples") else 0
            min_energy = None

            # Try to extract min energy
            try:
                if hasattr(_value, "samples") and _value.samples:
                    energies = [
                        sample.objective
                        for sample in _value.samples
                        if hasattr(sample, "objective")
                    ]
                    if energies:
                        min_energy = min(energies)
            except Exception:
                pass  # If extraction fails, just log without min_energy

            self._experiment._logger.log_sampleset(_name, num_samples, min_energy)

        self._datastore.add(_name, _value, "samplesets")

    def log_solver(
        self,
        solver_name: str | typ.Callable[P, R],
        solver: typ.Optional[typ.Callable[P, R]] = None,
        *,
        exclude_params: typ.Optional[list[str]] = None,
    ) -> typ.Callable[P, R]:
        """Log solver name and parameters to this run.

        When the wrapped solver function is called, the following actions are performed:
        - The solver name is logged as a parameter
        - Each keyword argument is logged as appropriate (parameter, problem, instance)
        - The result is logged as a sampleset or solution if applicable

        Args:
            solver_name: Name of the solver or the solver function itself.
            solver: Solver function (if solver_name is a string).
            exclude_params: List of parameter names to exclude from logging.

        Returns:
            Wrapped solver function.

        Raises:
            RuntimeError: If the run has been closed.
        """
        self._check_not_closed()

        _solver: typ.Callable[P, R]
        if solver is not None:
            if not isinstance(solver_name, str):
                raise ValueError("solver_name must be a string.")
            if not isinstance(solver, typ.Callable):
                raise ValueError("solver must be a callable.")
            _solver = solver
        else:
            if not isinstance(solver_name, typ.Callable):
                raise ValueError("solver_name must be a callable.")
            _solver = solver_name
            solver_name = solver_name.__name__

        self.log_parameter("solver_name", solver_name)
        exclude_params = exclude_params or []

        @wraps(_solver)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import time

            start_time = time.time()

            for name, value in kwargs.items():
                if name in exclude_params:
                    continue
                if isinstance(value, (int, float, str)):
                    self.log_parameter(name, value)
                elif isinstance(value, jm.Problem):
                    # Log to experiment level through parent
                    if self._experiment:
                        self._experiment.log_problem(name, value)
                elif isinstance(value, ommx_v1.Instance):
                    # Log to experiment level through parent
                    if self._experiment:
                        self._experiment.log_instance(name, value)

            result = _solver(*args, **kwargs)

            execution_time = time.time() - start_time

            # Log solver execution
            if self._experiment and hasattr(self._experiment, "_logger"):
                self._experiment._logger.log_solver(solver_name, execution_time)

            if isinstance(
                result, (ommx_v1.SampleSet, jm.SampleSet, jm.experimental.SampleSet)
            ):
                self.log_sampleset(solver_name + "_result", result)
            elif isinstance(result, ommx_v1.Solution):
                self.log_solution(solver_name + "_result", result)

            return result

        return typ.cast(typ.Callable[P, R], wrapper)

    def _get_name_or_default(
        self, name: str | typ.Any, obj: typ.Optional[typ.Any], storage: dict
    ) -> tuple[str, typ.Any]:
        """Get the name or generate a default name if the object is provided.

        Args:
            name: Name of the object or the object itself.
            obj: The object itself (optional).
            storage: Storage dictionary to generate default names.

        Returns:
            Tuple of (name, object).
        """
        if not isinstance(name, str) and obj is None:
            return str(len(storage)), name
        elif isinstance(name, str) and (obj is not None):
            return name, obj
        else:
            raise ValueError(
                "Invalid arguments: name must be a string or obj must be provided."
            )

    def close(self):
        """Close this run and log elapsed time.

        Once closed, the run cannot be used for further logging.
        """
        if self._closed:
            return

        run_time = datetime.now() - self._start_time
        elapsed_time = run_time.total_seconds()
        self._datastore.add("elapsed_time", elapsed_time, "meta_data")

        # Log run completion
        if self._experiment and hasattr(self._experiment, "_logger"):
            self._experiment._logger.log_run_end(self._run_id, elapsed_time)

        self._closed = True

    def __enter__(self) -> "Run":
        """Enter method for context manager."""
        if self._experiment:
            self._experiment._current_run = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method for context manager."""
        if self._experiment:
            self._experiment._current_run = None
        self.close()

    @property
    def parameters(self) -> dict[str, typ.Any]:
        """Get all parameters logged to this run."""
        return self._datastore.parameters

    @property
    def objects(self) -> dict[str, typ.Any]:
        """Get all objects logged to this run."""
        return self._datastore.objects

    @property
    def solutions(self) -> dict[str, ommx_v1.Solution]:
        """Get all solutions logged to this run."""
        return self._datastore.solutions

    @property
    def samplesets(self) -> dict[str, ommx_v1.SampleSet]:
        """Get all samplesets logged to this run."""
        return self._datastore.samplesets

    @property
    def problems(self) -> dict[str, jm.Problem]:
        """Get all problems logged to this run."""
        return self._datastore.problems

    @property
    def instances(self) -> dict[str, ommx_v1.Instance]:
        """Get all instances logged to this run."""
        return self._datastore.instances

    @property
    def run_id(self) -> int:
        """Get the run ID."""
        return self._run_id

    @property
    def is_closed(self) -> bool:
        """Check if the run is closed."""
        return self._closed
