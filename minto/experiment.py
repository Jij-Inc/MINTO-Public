import pathlib
import typing as typ
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import jijmodeling as jm
import ommx.artifact as ox_artifact
import ommx.v1 as ommx_v1
import pandas as pd

from .table import create_table, create_table_from_stores
from .utils import type_check
from .v1.datastore import DataStore
from .v1.exp_dataspace import ExperimentDataSpace

DEFAULT_RESULT_DIR = pathlib.Path(".minto_experiments")


T = typ.TypeVar("T")


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

        Args:
            instance_name: Name of the instance, or the instance object itself.
            instance: Instance object (if instance_name is a string).

        If a run is active, the instance is saved with the current run ID.
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

    def log_parameter(self, name: str, value: float | int | str):
        """Log a parameter to the experiment database.

        Args:
            name: Name of the parameter
            value: Value of the parameter

        If a run is active, the parameter is saved with the current run ID.
        Else, it is saved as experiment-wide data.
        """
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

    @classmethod
    def load_from_dir(cls, savedir: str | pathlib.Path) -> "Experiment":
        """Load an experiment from a directory containing saved data.

        Args:
            savedir: Directory path containing the saved experiment data.

        Returns:
            Experiment: Instance of the loaded experiment.
        """
        savedir = pathlib.Path(savedir)

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
