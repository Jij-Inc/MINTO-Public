import datetime
import enum
import pathlib
import typing as typ
from dataclasses import dataclass, field

import ommx.artifact as ox_art

from .datastore import DataStore


class Space(enum.Enum):
    """Enumeration defining the types of spaces in the experiment data structure.

    Attributes:
        EXPERIMENT: Space for experiment-wide data
        RUN: Space for individual run data
    """

    EXPERIMENT = "experiment"
    RUN = "run"


@dataclass
class ExperimentDataSpace:
    """A container class for managing experimental data with multiple runs.

    This class provides a structured way to organize and manage data from optimization
    experiments, including both experiment-wide data and data from individual runs.

    Directory structure when saved:
    ```
    base_dir/
    ├── experiment/          # Experiment-wide data
    │   ├── problems_*.problem
    │   ├── instances_*.instance
    │   └── ...
    └── runs/               # Individual run data
        ├── 0/             # First run
        │   ├── problems_*.problem
        │   ├── solutions_*.solution
        │   └── ...
        ├── 1/             # Second run
        └── ...
    ```

    Attributes:
        experiment_name (str): Name identifier for the experiment
        experiment_datastore (DataStore): Storage for experiment-wide data
        run_datastores (list[DataStore]): List of storage for individual runs
        version (ClassVar[str]): Version of the data space format
        exp_dirname (ClassVar[str]): Name of the experiment directory
        runs_dirname (ClassVar[str]): Name of the runs directory
    """

    experiment_name: str
    experiment_datastore: DataStore = field(default_factory=DataStore)
    run_datastores: list[DataStore] = field(default_factory=list)

    version: typ.ClassVar[str] = "1.0"
    exp_dirname: typ.ClassVar[str] = "experiment"
    runs_dirname: typ.ClassVar[str] = "runs"

    def __post_init__(self):
        """Initialize metadata after instance creation.

        Sets up basic metadata including version, experiment name, and timestamp.
        """
        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        meta_data = {
            "version": self.version,
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
        }

        self.experiment_datastore.meta_data = meta_data
        self._first_saving = False

    def add_run_datastore(
        self,
        run_datastore: DataStore,
        with_save=False,
        save_dir: str | pathlib.Path = ".",
    ) -> int:
        """Add a new run datastore to the experiment.

        Args:
            run_datastore (DataStore): The datastore for the new run

        Returns:
            int: ID assigned to the new run

        Example:
            >>> exp_space = ExperimentDataSpace("optimization_exp")
            >>> run_store = DataStore()
            >>> run_id = exp_space.add_run_datastore(run_store)
        """
        run_id = len(self.run_datastores)
        self.run_datastores.append(run_datastore)
        self.add_run_data(run_id, "run_id", run_id, "meta_data", with_save, save_dir)
        return run_id

    def add_exp_data(
        self,
        name: str,
        obj,
        storage_name: str,
        with_save=False,
        save_dir: str | pathlib.Path = ".",
    ):
        """Add data to the experiment-wide storage.

        Args:
            name (str): Identifier for the data
            obj (Any): The data to store
            storage_name (str): Type of storage to use
            with_save (bool, optional): Whether to save immediately. Defaults to False.
            save_dir (str | Path, optional): Directory for saving. Defaults to "."

        Example:
            >>> exp_space.add_exp_data("config", {"param": 1}, "objects")
        """

        exp_dir = pathlib.Path(save_dir) / self.exp_dirname
        if with_save and not self._first_saving:
            self.experiment_datastore.save_all(exp_dir)
            self._first_saving = True

        self.experiment_datastore.add(
            name, obj, storage_name, with_save, save_dir=exp_dir
        )

    def add_run_data(
        self,
        run_id: int,
        name: str,
        obj,
        storage_name: str,
        with_save=False,
        save_dir: str | pathlib.Path = ".",
    ):
        """Add data to a specific run's storage.

        Args:
            run_id (int): ID of the run to add data to
            name (str): Identifier for the data
            obj (Any): The data to store
            storage_name (str): Type of storage to use
            with_save (bool, optional): Whether to save immediately. Defaults to False.
            save_dir (str | Path, optional): Directory for saving. Defaults to "."

        Raises:
            ValueError: If run_id is not found in run_datastores
        """
        exp_dir = pathlib.Path(save_dir) / self.exp_dirname
        if with_save and not self._first_saving:
            self.experiment_datastore.save_all(exp_dir)
            self._first_saving = True

        if run_id >= len(self.run_datastores):
            raise ValueError(f"Run ID {run_id} not found in run_datastores")
        runs_dir = pathlib.Path(save_dir) / self.runs_dirname / str(run_id)
        self.run_datastores[run_id].add(name, obj, storage_name, with_save, runs_dir)

    def save_dir(self, base_dir: str | pathlib.Path):
        """Save all experiment data to a directory structure.

        Creates a directory structure containing all experiment and run data,
        organized according to the standard layout.

        Args:
            base_dir (str | Path): Base directory for saving data
        """
        base_dir = pathlib.Path(base_dir)
        # save experiment data
        exp_dir = base_dir / self.exp_dirname
        self.experiment_datastore.save_all(exp_dir)

        # save run data
        runs_dir = base_dir / self.runs_dirname
        for run_id, datastore in enumerate(self.run_datastores):
            run_dir = runs_dir / str(run_id)
            datastore.save_all(run_dir)

    @classmethod
    def load_from_dir(cls, base_dir: str | pathlib.Path) -> "ExperimentDataSpace":
        """Load an experiment data space from a directory structure.

        Args:
            base_dir (str | Path): Base directory containing the experiment data

        Returns:
            ExperimentDataSpace: New instance containing the loaded data

        Example:
            >>> exp_space = ExperimentDataSpace.load_from_dir("./experiments/exp1")
        """
        base_dir = pathlib.Path(base_dir)
        exp_dir = base_dir / cls.exp_dirname
        exp_datastore = DataStore.load(exp_dir)

        runs_dir = base_dir / cls.runs_dirname
        run_datastores = []
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                run_datastore = DataStore.load(run_dir)
                run_id = run_datastore.meta_data["run_id"]
                len_datastores = len(run_datastores)
                if len_datastores > run_id:
                    run_datastores[run_id] = run_datastore
                else:
                    for _ in range(run_id - len_datastores + 1):
                        run_datastores.append(DataStore())
                    run_datastores[run_id] = run_datastore

            exp_name = exp_datastore.meta_data["experiment_name"]

        return cls(
            experiment_name=exp_name,
            experiment_datastore=exp_datastore,
            run_datastores=run_datastores,
        )

    def add_to_artifact_builder(self, builder: ox_art.ArtifactBuilder):
        """Add all experiment data to an OMMX artifact builder.

        Adds both experiment-wide data and run data to the artifact builder,
        with appropriate annotations to maintain the hierarchical structure.

        Args:
            builder (ox_art.ArtifactBuilder): The artifact builder to add data to
        """
        minto_namespace = self.experiment_datastore.minto_namespace
        annotation_key = minto_namespace + ".space"
        annotations = {annotation_key: Space.EXPERIMENT.value}
        self.experiment_datastore.add_to_artifact_builder(
            builder=builder, annotations=annotations
        )

        annotations[annotation_key] = Space.RUN.value
        for run_id, datastore in enumerate(self.run_datastores):
            annotations[minto_namespace + ".run_id"] = str(run_id)
            datastore.add_to_artifact_builder(builder, annotations)

    @classmethod
    def load_from_ommx_archive(cls, path: str | pathlib.Path) -> "ExperimentDataSpace":
        """Create an experiment data space from an OMMX artifact.

        Loads and organizes data from an OMMX artifact, reconstructing the
        experiment and run structure based on layer annotations.

        Args:
            path (str | Path): Path to the OMMX artifact file

        Returns:
            ExperimentDataSpace: New instance containing the loaded data

        Notes:
            - Uses layer annotations to distinguish between experiment and run data
            - Creates empty datastores for missing runs to maintain run ID sequence
            - Sets experiment_name to "unknown" if not found in metadata
        """
        artifact = ox_art.Artifact.load_archive(path)
        return cls.load_from_ommx_artifact(artifact)

    @classmethod
    def load_from_ommx_artifact(
        cls, artifact: ox_art.Artifact
    ) -> "ExperimentDataSpace":
        minto_namespace = DataStore.minto_namespace
        exp_layers: list[ox_art.Descriptor] = []
        runs_layers: dict[int, list[ox_art.Descriptor]] = {}
        for layer in artifact.layers:
            storage_name = layer.annotations.get(minto_namespace + ".storage")
            space_name = layer.annotations.get(minto_namespace + ".space")
            if storage_name is None or space_name is None:
                continue
            if space_name == Space.EXPERIMENT.value:
                exp_layers.append(layer)
            elif space_name == Space.RUN.value:
                run_id_str = layer.annotations.get(minto_namespace + ".run_id")
                if run_id_str is None:
                    continue
                run_id: int = int(run_id_str)
                if run_id not in runs_layers:
                    runs_layers[run_id] = []
                runs_layers[run_id].append(layer)

        exp_datastore = DataStore.load_from_layers(artifact, exp_layers)

        if len(runs_layers) == 0:
            max_run_id = 0
        else:
            max_run_id = max(runs_layers.keys())
        run_datastores = []
        for run_id in range(max_run_id + 1):
            if run_id not in runs_layers:
                run_datastores.append(DataStore)
            else:
                layers = runs_layers[run_id]
                run_datastores.append(DataStore.load_from_layers(artifact, layers))

        if "experiment_name" not in exp_datastore.meta_data:
            exp_datastore.meta_data["experiment_name"] = "unknown"
        exp_name = exp_datastore.meta_data["experiment_name"]

        return cls(
            experiment_name=exp_name,
            experiment_datastore=exp_datastore,
            run_datastores=run_datastores,
        )
