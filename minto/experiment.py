"""Module for managing optimization experiments.

This module provides the `Experiment` class for managing optimization experiments.
The `Experiment` class handles experiment-level data (problems, instances,
global parameters)
while the `Run` class handles run-specific data (solutions, run parameters).
This explicit separation replaces the previous implicit with-clause behavior.

"""

import json
import pathlib
import typing as typ
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import jijmodeling as jm
import numpy as np
import ommx.artifact as ox_artifact
import ommx.v1 as ommx_v1
import pandas as pd

from .environment import collect_environment_info
from .logger import MintoLogger
from .logging_config import LogConfig
from .table import create_table, create_table_from_stores
from .utils import type_check
from .v1.datastore import DataStore
from .v1.exp_dataspace import ExperimentDataSpace

DEFAULT_RESULT_DIR = pathlib.Path(".minto_experiments")
"""Default directory `.minto_experiments` for saving experiment data.

You can change this by setting `savedir` in the `Experiment` class."""


def _get_default_verbose_logging() -> bool:
    """Get default value for verbose_logging based on environment."""
    import os

    # In test environment, default to False unless explicitly set
    if os.environ.get("MINTO_TESTING") == "true":
        return False
    return True


@dataclass
class Experiment:
    """Class to manage optimization experiments.

    This class provides a structured way to manage optimization experiments,
    handling experiment-level data (problems, instances, global parameters).
    Run-specific data is handled by the Run class through explicit run creation.

    This design eliminates the previous implicit with-clause behavior where
    the storage location depended on context. Now, experiment-level and
    run-level operations are explicitly separated.

    Properties:
        experiment_name (str): Full name of the experiment with timestamp.
        dataspace (ExperimentDataSpace): Data storage space for the experiment.
    """

    name: str = field(default_factory=lambda: str(uuid.uuid4().hex[:8]))
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR
    auto_saving: bool = True
    collect_environment: bool = True
    verbose_logging: bool = field(
        default_factory=lambda: _get_default_verbose_logging()
    )
    log_config: LogConfig = field(default_factory=LogConfig)

    timestamp: datetime = field(default_factory=datetime.now, init=False)
    _logger: MintoLogger = field(init=False)
    _start_time: datetime = field(default_factory=datetime.now, init=False)
    _run_count: int = field(default=0, init=False)
    _current_run: typ.Optional[typ.Any] = field(default=None, init=False)

    def __post_init__(self):
        """Post-initialization method for Experiment class."""
        # Initialize logger if verbose logging is enabled
        if self.verbose_logging:
            self._logger = MintoLogger(self.log_config)
        else:
            # Create a logger with disabled config
            disabled_config = LogConfig(enabled=False)
            self._logger = MintoLogger(disabled_config)

        # Record start time
        self._start_time = datetime.now()

        # Add timestamp to experiment name
        self.experiment_name = f"{self.name}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"
        # Set save directory path
        self.savedir = pathlib.Path(self.savedir) / self.experiment_name
        # Create directory if auto_saving is enabled
        if self.auto_saving:
            self.savedir.mkdir(exist_ok=True, parents=True)
        # Initialize dataspace
        self.dataspace = ExperimentDataSpace(self.name)

        # Log experiment start
        self._logger.log_experiment_start(self.name)

        # Collect environment metadata at experiment level (once at start)
        if self.collect_environment:
            self._collect_and_log_environment_info()

    def run(self):
        """Create a new run for this experiment.

        This method explicitly creates a new Run object for logging run-specific data.
        Unlike the previous implicit with-clause behavior, this makes it clear that
        data logged to the returned Run object will be stored at the run level.

        Returns:
            Run: A new Run instance for logging run-specific data.
        """
        # Import Run here to avoid circular imports
        from .run import Run

        run = Run()
        run._experiment = self  # Set parent reference
        run._run_id = self.dataspace.add_run_datastore(
            run._datastore, with_save=self.auto_saving, save_dir=self.savedir
        )

        # Log run creation
        self._logger.log_run_start(run._run_id)
        self._run_count += 1

        return run

    def finish_experiment(self):
        """Finish the experiment and log completion information.

        This method should be called when the experiment is complete
        to log the final statistics and elapsed time.
        """
        end_time = datetime.now()
        total_duration = (end_time - self._start_time).total_seconds()

        # Log experiment completion
        self._logger.log_experiment_end(self.name, total_duration, self._run_count)

    def save(self, path: typ.Optional[str | pathlib.Path] = None):
        """Save the experiment data to disk.

        Args:
            path: Optional path to save to. If None, uses default savedir.
        """
        if path is not None:
            self.dataspace.save_dir(path)
        else:
            self.dataspace.save_dir(self.savedir)

        # Automatically log experiment completion when saving
        self.finish_experiment()

    # NEW INTERFACE: log_global_* methods for experiment-level data
    # These methods make it clear that data is stored at experiment level

    def log_global_problem(
        self, problem_name: str | jm.Problem, problem: typ.Optional[jm.Problem] = None
    ):
        """Log an optimization problem to the experiment (experiment-level data).

        This is the preferred method for logging problems at the experiment level.
        Problems logged here are shared across all runs in the experiment.

        Args:
            problem_name: Name of the problem, or the problem object itself.
            problem: Problem object (if problem_name is a string).
        """
        _name, _problem = self._get_name_or_default(
            problem_name, problem, self.dataspace.experiment_datastore.problems
        )
        self.dataspace.add_exp_data(
            _name,
            _problem,
            "problems",
            with_save=self.auto_saving,
            save_dir=self.savedir,
        )

    def log_global_instance(
        self,
        instance_name: str | ommx_v1.Instance,
        instance: typ.Optional[ommx_v1.Instance] = None,
    ):
        """Log an optimization problem instance to the experiment.

        This logs to experiment-level data.

        This is the preferred method for logging instances at the experiment level.
        Instances logged here are shared across all runs in the experiment.

        Args:
            instance_name: Name of the instance, or the instance object itself.
            instance: Instance object (if instance_name is a string).
        """
        _name, _instance = self._get_name_or_default(
            instance_name,
            instance,
            self.dataspace.experiment_datastore.instances,
        )
        self.dataspace.add_exp_data(
            _name,
            _instance,
            "instances",
            with_save=self.auto_saving,
            save_dir=self.savedir,
        )

    def log_global_parameter(
        self,
        name: str,
        value: float | int | str | list | dict | np.ndarray,
    ):
        """Log a parameter to the experiment (experiment-level data).

        This is the preferred method for logging parameters at the experiment level.
        These parameters apply to the entire experiment, such as global configuration,
        dataset properties, or experiment setup. For run-specific parameters,
        use Run.log_parameter() method.

        Args:
            name (str): Name of the parameter.
            value: Value of the parameter. Can be scalar or complex data structure.
        """
        if isinstance(value, (list, dict, np.ndarray)):
            # Check value is serializable
            try:
                from minto.v1.json_encoder import NumpyEncoder

                json.dumps(value, cls=NumpyEncoder)
            except TypeError:
                raise ValueError("Value is not serializable.")

            self.log_global_config(name, {"parameter_" + name: value})

        self.dataspace.add_exp_data(
            name,
            value,
            "parameters",
            with_save=self.auto_saving,
            save_dir=self.savedir,
        )

    def log_global_params(self, params: dict[str, float | int | str]):
        """Log multiple parameters to the experiment (experiment-level data).

        This is the preferred method for logging multiple parameters at the
        experiment level.

        Args:
            params: Dictionary of parameter names and values.
        """
        for name, value in params.items():
            self.log_global_parameter(name, value)

    def log_global_config(self, name: str, value: dict[str, typ.Any]):
        """Log a configuration object to the experiment (experiment-level data).

        This is the preferred method for logging configuration objects at the
        experiment level.
        Use this for experiment-wide settings, metadata, or complex configuration
        objects.

        Args:
            name: Name of the configuration object
            value: Dictionary containing the configuration data
        """
        type_check([("name", name, str), ("value", value, dict)])
        self.dataspace.add_exp_data(
            name,
            value,
            "objects",
            with_save=self.auto_saving,
            save_dir=self.savedir,
        )

    def log_global_object(self, name: str, value: dict[str, typ.Any]):
        """Log a global object to the experiment (experiment-level data).

        This is an alias for log_global_config() for consistency with the notebook API.
        Use this for experiment-wide objects that should be accessible across all runs.

        Args:
            name: Name of the object
            value: Dictionary containing the object data
        """
        return self.log_global_config(name, value)

    # DEPRECATED INTERFACE: Keep for backward compatibility
    # These methods are deprecated in favor of log_global_* methods

    def log_problem(
        self, problem_name: str | jm.Problem, problem: typ.Optional[jm.Problem] = None
    ):
        """Log an optimization problem.

        This method delegates to the current run's log_problem method.
        Must be called within a run context (i.e., inside 'with experiment.run():').

        Args:
            problem_name: Name of the problem, or the problem object itself.
            problem: Problem object (if problem_name is a string).

        Raises:
            RuntimeError: If called outside of a run context.

        .. deprecated::
            Calling log_problem() on Experiment is deprecated.
            Use run.log_problem() directly for clearer semantics.
        """
        if self._current_run is None:
            raise RuntimeError(
                "log_problem() can only be called within a run context. "
                "Use 'with experiment.run() as run:' and then call "
                "run.log_problem() directly."
            )

        # Inside run context - show deprecation warning
        warnings.warn(
            "Calling log_problem() on Experiment is deprecated. "
            "Use 'with experiment.run() as run:' and then call "
            "run.log_problem() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._current_run.log_problem(problem_name, problem)

    def log_instance(
        self,
        instance_name: str | ommx_v1.Instance,
        instance: typ.Optional[ommx_v1.Instance] = None,
    ):
        """Log an optimization problem instance.

        This method delegates to the current run's log_instance method.
        Must be called within a run context (i.e., inside 'with experiment.run():').

        Args:
            instance_name: Name of the instance, or the instance object itself.
            instance: Instance object (if instance_name is a string).

        Raises:
            RuntimeError: If called outside of a run context.

        .. deprecated::
            Calling log_instance() on Experiment is deprecated.
            Use run.log_instance() directly for clearer semantics.
        """
        if self._current_run is None:
            raise RuntimeError(
                "log_instance() can only be called within a run context. "
                "Use 'with experiment.run() as run:' and then call "
                "run.log_instance() directly."
            )

        # Inside run context - show deprecation warning
        warnings.warn(
            "Calling log_instance() on Experiment is deprecated. "
            "Use 'with experiment.run() as run:' and then call "
            "run.log_instance() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._current_run.log_instance(instance_name, instance)

    def log_parameter(
        self,
        name: str,
        value: float | int | str | list | dict | np.ndarray,
    ):
        """Log a parameter.

        This method delegates to the current run's log_parameter method.
        Must be called within a run context (i.e., inside 'with experiment.run():').

        Args:
            name (str): Name of the parameter.
            value: Value of the parameter. Can be scalar or complex data structure.

        Raises:
            RuntimeError: If called outside of a run context.

        .. deprecated::
            Calling log_parameter() on Experiment is deprecated.
            Use run.log_parameter() directly for clearer semantics.
        """
        if self._current_run is None:
            raise RuntimeError(
                "log_parameter() can only be called within a run context. "
                "Use 'with experiment.run() as run:' and then call "
                "run.log_parameter() directly."
            )

        # Inside run context - show deprecation warning
        warnings.warn(
            "Calling log_parameter() on Experiment is deprecated. "
            "Use 'with experiment.run() as run:' and then call "
            "run.log_parameter() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._current_run.log_parameter(name, value)

    def log_params(self, params: dict[str, float | int | str]):
        """Log multiple parameters to the experiment (experiment-level data).

        .. deprecated::
            Use log_global_params() instead for clearer experiment-level data handling.

        Args:
            params: Dictionary of parameter names and values.
        """
        import warnings

        warnings.warn(
            "log_params() is deprecated. Use log_global_params() instead for clearer "
            "experiment-level data handling.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log_global_params(params)

    def log_object(self, name: str, value: dict[str, typ.Any]):
        """Log a custom object.

        This method delegates to the current run's log_object method.
        Must be called within a run context (i.e., inside 'with experiment.run():').

        Args:
            name: Name of the object
            value: Dictionary containing the object data

        Raises:
            RuntimeError: If called outside of a run context.

        .. deprecated::
            Calling log_object() on Experiment is deprecated.
            Use run.log_object() directly for clearer semantics.
        """
        if self._current_run is None:
            raise RuntimeError(
                "log_object() can only be called within a run context. "
                "Use 'with experiment.run() as run:' and then call "
                "run.log_object() directly."
            )

        # Inside run context - show deprecation warning
        warnings.warn(
            "Calling log_object() on Experiment is deprecated. "
            "Use 'with experiment.run() as run:' and then call "
            "run.log_object() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._current_run.log_object(name, value)

    def log_solver(
        self,
        solver_name: str | typ.Callable,
        solver: typ.Optional[typ.Callable] = None,
        *,
        exclude_params: typ.Optional[list[str]] = None,
    ) -> typ.Callable:
        """Log solver name and parameters to the experiment.

        This method creates a wrapped version of a solver function that automatically
        logs solver parameters and results. This is maintained for backward
        compatibility
        with the old interface, but the new explicit interface using Run.log_solver()
        is recommended for run-specific operations.

        Args:
            solver_name: Name of the solver or the solver function itself.
            solver: Solver function (if solver_name is a string).
            exclude_params: List of parameter names to exclude from logging.

        Returns:
            Wrapped solver function.
        """
        from functools import wraps

        _solver: typ.Callable
        if solver is not None:
            if not isinstance(solver_name, str):
                raise ValueError("solver_name must be a string.")
            if not callable(solver):
                raise ValueError("solver must be a callable.")
            _solver = solver
        else:
            if not callable(solver_name):
                raise ValueError("solver_name must be a callable.")
            _solver = solver_name
            solver_name = solver_name.__name__

        # Log solver name at experiment level
        self.log_global_parameter("solver_name", solver_name)
        exclude_params = exclude_params or []

        @wraps(_solver)
        def wrapper(*args, **kwargs):
            for name, value in kwargs.items():
                if name in exclude_params:
                    continue
                if isinstance(value, (int, float, str)):
                    self.log_global_parameter(name, value)
                elif isinstance(value, jm.Problem):
                    self.log_global_problem(name, value)
                elif isinstance(value, ommx_v1.Instance):
                    self.log_global_instance(name, value)

            result = _solver(*args, **kwargs)

            if isinstance(result, ommx_v1.SampleSet):
                self.log_sampleset(solver_name + "_result", result)
            elif isinstance(result, ommx_v1.Solution):
                self.dataspace.add_exp_data(
                    solver_name + "_result",
                    result,
                    "solutions",
                    with_save=self.auto_saving,
                    save_dir=self.savedir,
                )
            elif isinstance(result, (jm.SampleSet, jm.experimental.SampleSet)):
                raise TypeError(
                    "JijModeling SampleSet is no longer supported. "
                    "Please use OMMX SampleSet instead."
                )

            return result

        return wrapper

    def __enter__(self) -> "Experiment":
        """Enter method for context manager.

        Note: This is maintained for backward compatibility.
        The new explicit interface using create_run() is recommended.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method for context manager.

        Note: This is maintained for backward compatibility.
        """
        if self.auto_saving:
            self.save()

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

    @classmethod
    def load_from_dir(cls, savedir: str | pathlib.Path) -> "Experiment":
        """Load an experiment from a directory containing saved data.

        Args:
            savedir: Directory path containing the saved experiment data.

        Returns:
            Experiment: Instance of the loaded experiment.
        """
        savedir = pathlib.Path(savedir)

        # check directory exists
        if not savedir.exists():
            raise FileNotFoundError(f"Directory not found: {savedir}")

        dataspace = ExperimentDataSpace.load_from_dir(savedir)
        exp_name = dataspace.experiment_datastore.meta_data["experiment_name"]
        experiment = cls(exp_name, auto_saving=False)
        experiment.dataspace = dataspace
        return experiment

    def save_as_ommx_archive(
        self,
        savefile: typ.Optional[str | pathlib.Path] = None,
    ) -> ox_artifact.Artifact:
        """Save the experiment data as an OMMX artifact.

        Args:
            savefile: Path to save the OMMX artifact. If None, a default name is
                generated.
        """
        if savefile is None:
            # Generate default filename with timestamp
            savefile = (
                pathlib.Path(self.savedir)
                / f"{self.name}_{self.timestamp.strftime('%Y%m%d%H%M%S')}.ommx"
            )
        builder = ox_artifact.ArtifactBuilder.new_archive_unnamed(savefile)
        self.dataspace.add_to_artifact_builder(builder)
        return builder.build()

    @classmethod
    def load_from_ommx_archive(cls, savefile: str | pathlib.Path) -> "Experiment":
        """Load an experiment from an OMMX artifact file.

        Args:
            savefile: Path to the OMMX artifact file.

        Returns:
            Experiment: Instance of the loaded experiment.
        """
        dataspace = ExperimentDataSpace.load_from_ommx_archive(savefile)
        exp_name = dataspace.experiment_datastore.meta_data["experiment_name"]
        experiment = cls(exp_name, auto_saving=False)
        experiment.dataspace = dataspace
        return experiment

    def get_run_table(self) -> pd.DataFrame:
        """Get the run data as a table.

        Returns:
            pd.DataFrame: DataFrame containing the run data.
        """
        run_table = create_table_from_stores(self.dataspace.run_datastores)
        run_table.index.name = "run_id"
        return run_table

    def get_experiment_tables(self) -> dict[str, pd.DataFrame]:
        """Get the experiment data as a table.

        Returns:
            dict[str, pd.DataFrame]: Dictionary containing the experiment data tables.
        """
        exp_table = create_table(self.dataspace.experiment_datastore)
        return exp_table

    def push_github(
        self,
        org: str,
        repo: str,
        name: typ.Optional[str] = None,
        tag: typ.Optional[str] = None,
    ) -> ox_artifact.Artifact:
        """Push the experiment data to a GitHub repository.

        Returns:
            ox_artifact.Artifact: OMMX artifact containing the experiment data.
        """
        builder = ox_artifact.ArtifactBuilder.for_github(
            org=org,
            repo=repo,
            name=name if name else self.name,
            tag=tag if tag else self.timestamp.strftime("%Y%m%d%H%M%S"),
        )
        self.dataspace.add_to_artifact_builder(builder)
        artifact = builder.build()
        artifact.push()
        return artifact

    @property
    def runs(self) -> list[DataStore]:
        """Get the list of run datastores in the experiment.

        This property provides access to all run datastores in the experiment.
        Returns the same result as `self.dataspace.run_datastores`.

        Returns:
            list[DataStore]: List of run datastores in the experiment.
        """
        return self.dataspace.run_datastores

    @classmethod
    def load_from_registry(cls, imagename: str) -> "Experiment":
        """Load an experiment from a Docker registry.

        Args:
            imagename: Name of the Docker image containing the experiment data.

        Returns:
            Experiment: Instance of the loaded experiment.
        """
        artifact = ox_artifact.Artifact.load(imagename)
        dataspace = ExperimentDataSpace.load_from_ommx_artifact(artifact)
        exp_name = dataspace.experiment_datastore.meta_data["experiment_name"]
        experiment = cls(exp_name, auto_saving=False)
        experiment.dataspace = dataspace
        return experiment

    @classmethod
    def concat(
        cls,
        experiments: list["Experiment"],
        name: typ.Optional[str] = None,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        auto_saving: bool = True,
    ) -> "Experiment":
        """Concatenate multiple experiments into a single experiment.

        Args:
            experiments: List of Experiment instances to concatenate.
            name: Name of the concatenated experiment.
            savedir: Directory path for saving the concatenated experiment data.
            auto_saving: Flag to enable automatic saving of the concatenated experiment

        Example:
            ```python
            import minto
            exp1 = minto.Experiment("exp1")
            exp2 = minto.Experiment("exp2")
            exp3 = minto.Experiment("exp3")
            new_exp = minto.Experiment.concat([exp1, exp2, exp3])
            ```

        Returns:
            Experiment: Instance of the concatenated experiment.
        """

        name = name or uuid.uuid4().hex[:8]

        if len(experiments) == 0:
            raise ValueError("No experiments provided.")

        # check if dataspaces have the same experiment-wide data
        first_datastore = experiments[0].dataspace.experiment_datastore
        for experiment in experiments[1:]:
            datastore = experiment.dataspace.experiment_datastore
            if datastore.problems.keys() != first_datastore.problems.keys():
                raise ValueError("Experiments have different problems.")
            if datastore.instances.keys() != first_datastore.instances.keys():
                raise ValueError("Experiments have different instances.")
            if datastore.solutions.keys() != first_datastore.solutions.keys():
                raise ValueError("Experiments have different solutions.")
            if datastore.objects.keys() != first_datastore.objects.keys():
                raise ValueError("Experiments have different objects.")
            if datastore.parameters.keys() != first_datastore.parameters.keys():
                raise ValueError("Experiments have different parameters.")
            if datastore.meta_data.keys() != first_datastore.meta_data.keys():
                raise ValueError("Experiments have different meta data.")

        new_experiment = cls(name, savedir, auto_saving)
        new_experiment.dataspace.experiment_datastore = first_datastore

        for experiment in experiments:
            for datastore in experiment.dataspace.run_datastores:
                new_experiment.dataspace.add_run_datastore(
                    datastore,
                    with_save=new_experiment.auto_saving,
                    save_dir=new_experiment.savedir,
                )

        return new_experiment

    def _collect_and_log_environment_info(self):
        """Collect and log environment metadata."""
        try:
            env_info = collect_environment_info()

            # Save environment info as experiment-level metadata
            self.dataspace.experiment_datastore.add(
                "environment_info",
                env_info.to_dict(),
                "meta_data",
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )

            # Also save as object for reliable persistence
            self.dataspace.experiment_datastore.add(
                "environment_info",
                env_info.to_dict(),
                "objects",
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )

            # Save key environment info as parameters (for table display)
            self.dataspace.experiment_datastore.add(
                "python_version",
                env_info.python_version,
                "parameters",
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )
            self.dataspace.experiment_datastore.add(
                "os_name",
                env_info.os_name,
                "parameters",
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )
            self.dataspace.experiment_datastore.add(
                "platform_info",
                env_info.platform_info,
                "parameters",
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )

            # Log environment information to console
            self._logger.log_environment_info(env_info.to_dict())

        except Exception as e:
            # Show warning if environment info collection fails but continue experiment
            self._logger.log_warning(f"Failed to collect environment information: {e}")

    def get_environment_info(self) -> typ.Optional[dict]:
        """Get experiment environment metadata.

        Returns:
            Environment metadata dictionary, or None if not collected.
        """
        # First try to get from metadata
        env_info = self.dataspace.experiment_datastore.meta_data.get("environment_info")
        if env_info is not None:
            return env_info

        # If not in metadata, try to get from objects
        return self.dataspace.experiment_datastore.objects.get("environment_info")

    def print_environment_summary(self):
        """Print a summary of environment information."""
        env_info = self.get_environment_info()
        if env_info is None:
            print("Environment information not available.")
            print(
                "Set collect_environment=True when creating the experiment "
                "to collect environment metadata."
            )
            return

        print("=== Experiment Environment Information ===")
        print(f"OS: {env_info['os_name']} {env_info['os_version']}")
        print(f"Platform: {env_info['platform_info']}")
        print(f"Python: {env_info['python_version']}")
        print(f"CPU: {env_info['cpu_info']} ({env_info['cpu_count']} cores)")
        print(f"Memory: {env_info['memory_total'] // (1024**3)} GB")
        print(f"Architecture: {env_info['architecture']}")

        if env_info.get("virtual_env"):
            print(f"Virtual Environment: {env_info['virtual_env']}")

        print(f"Timestamp: {env_info['timestamp']}")

        print("\nKey Package Versions:")
        for pkg, version in env_info["package_versions"].items():
            print(f"  {pkg}: {version}")

    # Run context delegation methods
    def log_solution(
        self,
        solution_name: str | ommx_v1.Solution,
        solution: typ.Optional[ommx_v1.Solution] = None,
    ):
        """Log an optimization solution to the current run.

        This method delegates to the current run's log_solution method.
        Must be called within a run context (i.e., inside 'with experiment.run():').

        Args:
            solution_name: Name of the solution, or the solution object itself.
            solution: Solution object (if solution_name is a string).

        Raises:
            RuntimeError: If called outside of a run context.

        .. deprecated::
            Calling log_solution() on Experiment is deprecated.
            Use run.log_solution() directly for clearer semantics.
        """
        if self._current_run is None:
            raise RuntimeError(
                "log_solution() can only be called within a run context. "
                "Use 'with experiment.run() as run:' and then call "
                "run.log_solution() directly."
            )

        # Inside run context - show deprecation warning
        warnings.warn(
            "Calling log_solution() on Experiment is deprecated. "
            "Use 'with experiment.run() as run:' and then call "
            "run.log_solution() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._current_run.log_solution(solution_name, solution)

    def log_sampleset(
        self,
        name: str | ommx_v1.SampleSet | jm.SampleSet | jm.experimental.SampleSet,
        value: typ.Optional[ommx_v1.SampleSet] = None,
    ):
        """Log a SampleSet to the current run.

        This method delegates to the current run's log_sampleset method.
        Must be called within a run context (i.e., inside 'with experiment.run():').

        Args:
            name: Name of the sampleset, or the sampleset object itself.
            value: SampleSet object (if name is a string).

        Raises:
            RuntimeError: If called outside of a run context.

        .. deprecated::
            Calling log_sampleset() on Experiment is deprecated.
            Use run.log_sampleset() directly for clearer semantics.
        """
        if self._current_run is None:
            raise RuntimeError(
                "log_sampleset() can only be called within a run context. "
                "Use 'with experiment.run() as run:' and then call "
                "run.log_sampleset() directly."
            )

        # Inside run context - show deprecation warning
        warnings.warn(
            "Calling log_sampleset() on Experiment is deprecated. "
            "Use 'with experiment.run() as run:' and then call "
            "run.log_sampleset() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._current_run.log_sampleset(name, value)
