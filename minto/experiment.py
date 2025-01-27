import json
import pathlib
import typing as typ
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

import jijmodeling as jm
import numpy as np
import ommx.artifact as ox_artifact
import ommx.v1 as ommx_v1
import pandas as pd

from .sampleset_converter import convert_sampleset_jijmodeling_to_ommx
from .table import create_table, create_table_from_stores
from .utils import type_check
from .v1.datastore import DataStore
from .v1.exp_dataspace import ExperimentDataSpace

DEFAULT_RESULT_DIR = pathlib.Path(".minto_experiments")


T = typ.TypeVar("T")
P = typ.ParamSpec("P")
R = typ.TypeVar("R")


@dataclass
class Experiment:
    """Class to manage optimization experiments.

    This class provides a structured way to manage optimization experiments,
    including logging of problems, instances, solutions, and parameters.
    It supports both experiment-wide data and run-specific data, with
    automatic saving capabilities and artifact creation.

    Attributes:
        name (str): Name of the experiment.
        savedir (pathlib.Path): Directory path for saving experiment data.
        auto_saving (bool): Flag to enable automatic saving of experiment data.
        timestamp (datetime): Timestamp of the experiment creation.
        _running (bool): Flag to track the current run status.
        _run_id (int): ID of the current run.

    Properties:
        experiment_name (str): Full name of the experiment with timestamp.
        database (Database): Database instance for storing experiment data.
    """

    name: str = field(default_factory=lambda: str(uuid.uuid4().hex[:8]))
    savedir: pathlib.Path = DEFAULT_RESULT_DIR
    auto_saving: bool = True

    timestamp: datetime = field(default_factory=datetime.now, init=False)
    _running: bool = field(default=False, init=False)
    _run_id: int = field(default=0, init=False)

    def __post_init__(self):
        """Post-initialization method for Experiment class.

        - Add timestamp to the experiment name.
        - Initialize the database instance.
        - Create the OMMX artifact builder.
        """
        # 実験名にタイムスタンプを追加
        self.experiment_name = f"{self.name}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"
        # 保存ディレクトリのパスを設定
        self.savedir = pathlib.Path(self.savedir) / self.experiment_name
        # 自動保存が有効な場合、ディレクトリを作成
        if self.auto_saving:
            self.savedir.mkdir(exist_ok=True, parents=True)
        # データベースインスタンスの初期化
        self.dataspace = ExperimentDataSpace(self.name)

    def run(self) -> "Experiment":
        """Start a new experiment run.

        Returns:
            Experiment: Instance of the current experiment run.
        """
        self._running = True
        self._run_id = self.dataspace.add_run_datastore(
            DataStore(), with_save=self.auto_saving, save_dir=self.savedir
        )
        self._run_start_time = datetime.now()
        return self

    def close_run(self):
        """Close the current experiment run.

        This method should be called to properly end the current run.
        """
        run_time = datetime.now() - self._run_start_time
        elapsed_time = run_time.total_seconds()
        self.log_data(
            "elapsed_time",
            elapsed_time,
            "meta_data",
        )

        self._running = False
        self._run_id += 1

    def __enter__(self) -> "Experiment":
        """Enter method for the context manager.

        Returns:
            Experiment: Instance of the current experiment run.
        """
        if self._running:
            return self
        else:
            return self.run()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method for the context manager.

        Args:
            exc_type: exception type
            exc_value: exception value
            traceback: traceback information
        """
        self.close_run()

    def save(self, path: typ.Optional[str | pathlib.Path] = None):
        """Save the experiment database to the disk."""
        if path is not None:
            self.dataspace.save_dir(path)
        else:
            self.dataspace.save_dir(self.savedir)

    def get_current_datastore(self) -> "DataStore":
        """Get the current DataStore object."""
        if self._running:
            return self.dataspace.run_datastores[self._run_id]
        else:
            return self.dataspace.experiment_datastore

    def log_data(
        self,
        name: str,
        data: typ.Any,
        storage_name: str,
    ):
        """Log data to the experiment database.

        This method is not intended to be called directly by the user.
        Instead, users should use other `log_*` methods.
        Other `log_*` methods wrap this method to save data to the dataspace.
        If the current experiment is not running, the data is saved to `.dataspace.experiment_datastore`.
        If the current experiment is running, the data is saved to `.dataspace.run_datastores[self._run_id]`.

        Args:
            name: Name of the data
            data: Data object to be saved
            storage_name: Name of the storage object
        """
        if not self._running:
            self.dataspace.add_exp_data(
                name,
                data,
                storage_name,
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )
        else:
            self.dataspace.add_run_data(
                self._run_id,
                name,
                data,
                storage_name,
                with_save=self.auto_saving,
                save_dir=self.savedir,
            )

    def _get_name_or_default(
        self, name: str | T, obj: typ.Optional[T], storage: dict
    ) -> tuple[str, T]:
        """Get the name or generate a default name if the object is provided.

        Args:
            name: Name of the object or the type of the object.
            obj: The object itself (optional).

        Returns:
            str: The name or a generated default name.
        """
        if not isinstance(name, str) and obj is None:
            return str(len(storage)), name
        elif isinstance(name, str) and (obj is not None):
            return name, obj
        else:
            raise ValueError(
                "Invalid arguments: name must be a string or obj must be provided."
            )

    def log_problem(
        self, problem_name: str | jm.Problem, problem: typ.Optional[jm.Problem] = None
    ):
        """Log an optimization problem to the experiment database.

        Args:
            problem_name: Name of the problem, or the problem object itself.
            problem: Problem object (if problem_name is a string).

        If a run is active, the problem is saved with the current run ID.
        """
        datastore = self.get_current_datastore()
        _name, _problem = self._get_name_or_default(
            problem_name, problem, datastore.problems
        )
        self.log_data(
            _name,
            _problem,
            "problems",
        )

    def log_instance(
        self,
        instance_name: str | ommx_v1.Instance,
        instance: typ.Optional[ommx_v1.Instance] = None,
    ):
        """Log an optimization problem instance to the experiment database.

        Logs an ommx.v1.Instance to the experiment database.
        If instance_name is not specified, it will be named sequentially as "0", "1", "2", etc.

        Args:
            instance_name: Name of the instance, or the instance object itself.
            instance: Instance object (if instance_name is a string).

        Example:
            ```python
            import minto
            exp = minto.Experiment("exp1")
            instance = ommx_v1.Instance()
            exp.log_instance("instance1", instance)
            ```
        """
        datastore = self.get_current_datastore()
        _name, _instance = self._get_name_or_default(
            instance_name, instance, datastore.instances
        )
        self.log_data(
            _name,
            _instance,
            "instances",
        )

    def log_solution(
        self,
        solution_name: str | ommx_v1.Solution,
        solution: typ.Optional[ommx_v1.Solution] = None,
    ):
        """Log an optimization solution to the experiment database.

        Args:
            solution_name: Name of the solution, or the solution object itself.
            solution: Solution object (if solution_name is a string).

        If a run is active, the solution is saved with the current run ID.
        """
        datastore = self.get_current_datastore()
        _name, _solution = self._get_name_or_default(
            solution_name, solution, datastore.solutions
        )
        self.log_data(_name, _solution, "solutions")

    def log_parameter(
        self,
        name: str,
        value: float | int | str | list | dict | np.ndarray,
    ):
        """Log a parameter to the experiment database.

        This method allows logging of both simple scalar values (float, int, str) and
        complex data structures (list, dict, numpy.ndarray) as parameters. Complex data
        structures are first checked for JSON serializability and then stored as objects
        with a "parameter_" prefix to distinguish them from regular parameters.

        Args:
            name (str): Name of the parameter. Must be a string that uniquely identifies
                the parameter within the experiment.
            value (Union[float, int, str, list, dict, numpy.ndarray]): Value of the parameter.
                Can be one of the following types:
                - float: Floating point numbers
                - int: Integer numbers
                - str: String values
                - list: Python lists (must be JSON serializable)
                - dict: Python dictionaries (must be JSON serializable)
                - numpy.ndarray: NumPy arrays (must be JSON serializable)

        Raises:
            ValueError: If a complex data structure (list, dict, numpy.ndarray) is not
                JSON serializable. This can happen if the structure contains objects that
                cannot be converted to JSON (e.g., custom objects, functions).
            TypeError: If the name is not a string or if the value is not one of the
                supported types.

        Note:
            - For complex data structures (list, dict, numpy.ndarray), the value is stored
              both as a parameter and as an object with a "parameter_" prefix.
            - If the experiment is running (i.e., within a run context), the parameter
              is saved with the current run ID. Otherwise, it is saved as experiment-wide data.
            - NumPy arrays are converted to nested lists when serialized to JSON.

        Example:
            ```python
            exp = Experiment("example")
            # Logging simple scalar values
            exp.log_parameter("learning_rate", 0.001)
            exp.log_parameter("batch_size", 32)

            # Logging complex data structures
            exp.log_parameter("layer_sizes", [64, 128, 64])
            exp.log_parameter("model_config", {"activation": "relu", "dropout": 0.5})
            exp.log_parameter("weights", np.array([1.0, 2.0, 3.0]))
            ```
        """
        if isinstance(value, (list, dict, np.ndarray)):
            # Check value is serializable
            try:
                from minto.v1.json_encoder import NumpyEncoder

                json.dumps(value, cls=NumpyEncoder)
            except TypeError:
                raise ValueError(f"Value is not serializable.")

            self.log_object(name, {"paramter_" + name: value})

        self.log_data(name, value, "parameters")

    def log_params(self, params: dict[str, float | int]):
        """Log multiple parameters to the experiment database.

        Args:
            params: Dictionary of parameter names and values.

        If a run is active, the parameters are saved with the current run ID.
        Else, they are saved as experiment-wide data.
        """
        for name, value in params.items():
            self.log_parameter(name, value)

    def log_object(self, name: str, value: dict[str, typ.Any]):
        """Log a custom object to the experiment database.

        Args:
            name: Name of the object
            value: Dictionary containing the object data

        If a run is active, the object is saved with the current run ID.
        """
        type_check([("name", name, str), ("value", value, dict)])
        self.log_data(name, value, "objects")

    def log_sampleset(
        self,
        name: str | ommx_v1.SampleSet | jm.SampleSet | jm.experimental.SampleSet,
        value: typ.Optional[ommx_v1.SampleSet] = None,
    ):
        """Log a SampleSet to the experiment or run database."""

        # If name is specified, use it as the name.
        # If name is not specified and a SampleSet is passed as the first argument,
        # use the variable name of the SampleSet as the name.
        datastore = self.get_current_datastore()
        _name, _value = self._get_name_or_default(name, value, datastore.samplesets)

        if isinstance(_value, (jm.SampleSet, jm.experimental.SampleSet)):
            _value = convert_sampleset_jijmodeling_to_ommx(_value)

        self.log_data(_name, _value, "samplesets")

    def log_solver(
        self,
        solver: typ.Callable[P, R],
        *,
        exclude_params: typ.Optional[list[str]] = None,
    ) -> typ.Callable[P, R]:
        """Log solver name and parameters to the dataspace.

        When the wrapped solver function is called, the following actions are performed:

        - The solver name is logged as a parameter with `.log_parameter("solver_name", solver_name)`.
        - Each keyword argument passed to the solver function is logged as a parameter if it is of type int, float, or str.
        - If a keyword argument is of type `jm.Problem`, it is logged as a problem.
        - If a keyword argument is of type `ommx_v1.Instance`, it is logged as an instance.
        - After the solver function executes, the result is logged as a sampleset if it is of type `ommx_v1.SampleSet`, `jm.SampleSet`, or `jm.experimental.SampleSet`.
        - If the result is of type `ommx_v1.Solution`, it is logged as a solution.

        Args:
            solver: Solver function to be logged.

        Returns:
            typ.Callable[P, R]: Wrapped solver function.

        Example:
            ```python
            import minto
            exp = minto.Experiment("exp1")
            result = exp.log_solver(solver)(parameter=1)
            exp.dataspace.experiment_datastore.parameters
            # {'solver_name': 'solver', 'parameter': 1}
            ```
        """
        solver_name = solver.__name__
        self.log_parameter("solver_name", solver_name)

        exclude_params = exclude_params or []

        @wraps(solver)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for name, value in kwargs.items():
                if name in exclude_params:
                    continue
                if isinstance(value, (int, float, str)):
                    self.log_parameter(name, value)
                elif isinstance(value, jm.Problem):
                    self.log_problem(name, value)
                elif isinstance(value, ommx_v1.Instance):
                    self.log_instance(name, value)

            result = solver(*args, **kwargs)

            if isinstance(
                result, (ommx_v1.SampleSet, jm.SampleSet, jm.experimental.SampleSet)
            ):
                self.log_sampleset(solver_name + "_result", result)
            elif isinstance(result, ommx_v1.Solution):
                self.log_solution(solver_name + "_result", result)
            return result

        return typ.cast(typ.Callable[P, R], wrapper)

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
        self, savefile: typ.Optional[str | pathlib.Path] = None
    ) -> ox_artifact.Artifact:
        """Save the experiment data as an OMMX artifact.

        Args:
            savefile: Path to save the OMMX artifact. If None, a default name is generated.
        """
        if savefile is None:
            # デフォルトのファイル名を生成（タイムスタンプ付き）
            savefile = (
                self.savedir
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
            pd.DataFrame: DataFrame containing the experiment data.
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
                new_experiment._run_id = new_experiment.dataspace.add_run_datastore(
                    datastore,
                    with_save=new_experiment.auto_saving,
                    save_dir=new_experiment.savedir,
                )

        return new_experiment
