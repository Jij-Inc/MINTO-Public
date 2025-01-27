import abc
import copy
import json
import pathlib
import typing as typ
from dataclasses import dataclass, field

import jijmodeling as jm
import ommx.artifact as ox_art
import ommx.v1 as ommx_v1

from .json_encoder import NumpyEncoder

T = typ.TypeVar("T")


class StorageStrategy(abc.ABC, typ.Generic[T]):
    """Abstract base class defining the interface for storage strategies.

    This class provides a generic interface for storing and loading different types of data,
    supporting both file system storage and OMMX artifact storage.

    Type Parameters:
        T: The type of data this strategy handles
    """

    @abc.abstractmethod
    def save(self, data: T, path: pathlib.Path):
        """Save data to the specified path.

        Args:
            data (T): The data to save
            path (pathlib.Path): The path where the data should be saved
        """
        pass

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> T:
        """Load data from the specified path.

        Args:
            path (pathlib.Path): The path from which to load the data

        Returns:
            T: The loaded data
        """
        pass

    @abc.abstractmethod
    def add_to_artifact_builder(
        self, data, builder: ox_art.ArtifactBuilder, annotations: dict[str, str]
    ) -> None:
        pass

    @abc.abstractmethod
    def load_from_layer(self, artifact: ox_art.Artifact, layer: ox_art.Descriptor) -> T:
        pass

    @property
    @abc.abstractmethod
    def extension(self) -> str:
        pass


@dataclass
class JSONStorage(StorageStrategy[dict]):
    def save(self, data, path: pathlib.Path):
        with path.open("w") as f:
            json.dump(data, f, cls=NumpyEncoder)

    def load(self, path: pathlib.Path) -> typ.Any:
        with path.open("r") as f:
            return json.load(f)

    def add_to_artifact_builder(
        self, data, builder: ox_art.ArtifactBuilder, annotations: dict[str, str]
    ):
        blob = json.dumps(data, cls=NumpyEncoder).encode("utf-8")
        builder.add_layer("application/json", blob, annotations)

    def load_from_layer(self, artifact: ox_art.Artifact, layer: ox_art.Descriptor):
        json_bytes = artifact.get_blob(layer)
        data = json.loads(json_bytes.decode("utf-8"))
        return data

    @property
    def extension(self):
        return "json"


@dataclass
class ProblemStorage(StorageStrategy[jm.Problem]):
    def save(self, data: jm.Problem, path: pathlib.Path):
        blob = jm.to_protobuf(data)
        with open(path, "wb") as f:
            f.write(blob)

    def load(self, path: pathlib.Path) -> jm.Problem:
        with open(path, "rb") as f:
            return jm.from_protobuf(f.read())

    def add_to_artifact_builder(
        self,
        data: jm.Problem,
        builder: ox_art.ArtifactBuilder,
        annotations: dict[str, str],
    ):
        blob = jm.to_protobuf(data)
        builder.add_layer(
            "application/vnd.jij.jijmodeling.v1.problem", blob, annotations
        )

    def load_from_layer(self, artifact: ox_art.Artifact, layer: ox_art.Descriptor):
        blob = artifact.get_blob(layer)
        return jm.from_protobuf(blob)

    @property
    def extension(self):
        return "problem"


@dataclass
class InstanceStorage(StorageStrategy[ommx_v1.Instance]):
    def save(self, data: ommx_v1.Instance, path: pathlib.Path):
        blob = data.to_bytes()
        with open(path, "wb") as f:
            f.write(blob)

    def load(self, path: pathlib.Path) -> ommx_v1.Instance:
        with open(path, "rb") as f:
            return ommx_v1.Instance.from_bytes(f.read())

    def add_to_artifact_builder(
        self,
        data: ommx_v1.Instance,
        builder: ox_art.ArtifactBuilder,
        annotations: dict[str, str],
    ):
        data.annotations.update(annotations)
        builder.add_instance(data)

    def load_from_layer(
        self, artifact: ox_art.Artifact, layer: ox_art.Descriptor
    ) -> ommx_v1.Instance:
        return artifact.get_instance(layer)

    @property
    def extension(self):
        return "instance"


@dataclass
class SolutionStorage(StorageStrategy[ommx_v1.Solution]):
    def save(self, data: ommx_v1.Solution, path: pathlib.Path):
        blob = data.to_bytes()
        with open(path, "wb") as f:
            f.write(blob)

    def load(self, path: pathlib.Path) -> ommx_v1.Solution:
        with open(path, "rb") as f:
            return ommx_v1.Solution.from_bytes(f.read())

    def add_to_artifact_builder(
        self,
        data: ommx_v1.Solution,
        builder: ox_art.ArtifactBuilder,
        annotations: dict[str, str],
    ):
        data.annotations.update(annotations)
        builder.add_solution(data)

    def load_from_layer(self, artifact: ox_art.Artifact, layer: ox_art.Descriptor):
        sol = artifact.get_solution(layer)
        # ommx artifact has a bug.
        # The annotations are not copied to the solution.
        sol.annotations.update(layer.annotations)
        return sol

    @property
    def extension(self):
        return "solution"


@dataclass
class SampleSetStorage(StorageStrategy[ommx_v1.SampleSet]):
    def save(self, data: ommx_v1.SampleSet, path: pathlib.Path):
        blob = data.to_bytes()
        with open(path, "wb") as f:
            f.write(blob)

    def load(self, path: pathlib.Path) -> ommx_v1.SampleSet:
        with open(path, "rb") as f:
            return ommx_v1.SampleSet.from_bytes(f.read())

    def add_to_artifact_builder(
        self,
        data: ommx_v1.SampleSet,
        builder: ox_art.ArtifactBuilder,
        annotations: dict[str, str],
    ):
        blob = data.to_bytes()
        builder.add_layer("application/org.ommx.v1.sampleset", blob, annotations)

    def load_from_layer(self, artifact: ox_art.Artifact, layer: ox_art.Descriptor):
        blob = artifact.get_blob(layer)
        return ommx_v1.SampleSet.from_bytes(blob)

    @property
    def extension(self):
        return "sampleset"


@dataclass
class DataStore:
    """A data store for managing optimization-related data with multiple storage types.

    This class provides a unified interface for storing and managing different types of
    optimization-related data, including problems, instances, solutions, and various
    metadata. It supports both file system storage and OMMX artifact storage.

    The data store maintains separate storage for:
    - Problems: JijModeling Problem instances
    - Instances: OMMX Instance objects
    - Solutions: OMMX Solution objects
    - Objects: Generic JSON-serializable objects
    - Parameters: Configuration parameters
    - Samplesets: OMMX SampleSet objects
    - Meta-data: Additional metadata

    Directory structure:
    ```
    dir
     ├── problem_*.problem    # Individual problem files
     ├── instance_*.instance  # Individual instance files
     ├── solution_*.solution  # Individual solution files
     ├── objects_*.json      # Individual object files
     ├── parameters_.json    # Single parameters file
     ├── samplesets_*.sampleset  # Individual sampleset files
     └── meta_data_.json     # Single metadata file
    ```

    Attributes:
        problems (dict[str, jm.Problem]): Storage for optimization problems
        instances (dict[str, ommx_v1.Instance]): Storage for problem instances
        solutions (dict[str, ommx_v1.Solution]): Storage for problem solutions
        objects (dict[str, dict]): Storage for generic JSON-serializable objects
        parameters (dict[str, dict[str, Any]]): Storage for parameters
        meta_data (dict[str, Any]): Storage for metadata
    """

    problems: dict[str, jm.Problem] = field(default_factory=dict)
    instances: dict[str, ommx_v1.Instance] = field(default_factory=dict)
    solutions: dict[str, ommx_v1.Solution] = field(default_factory=dict)
    objects: dict[str, dict] = field(default_factory=dict)
    parameters: dict[str, float | int] = field(default_factory=dict)
    samplesets: dict[str, ommx_v1.SampleSet] = field(default_factory=dict)
    meta_data: dict[str, typ.Any] = field(default_factory=dict)

    _storage_mapping: typ.ClassVar[dict[str, StorageStrategy]] = {
        "problems": ProblemStorage(),
        "instances": InstanceStorage(),
        "solutions": SolutionStorage(),
        "objects": JSONStorage(),
        "parameters": JSONStorage(),
        "samplesets": SampleSetStorage(),
        "meta_data": JSONStorage(),
    }

    minto_namespace: typ.ClassVar[str] = "org.minto"

    def add(
        self,
        name: str,
        obj,
        storage_name: str,
        with_save=False,
        save_dir: str | pathlib.Path = ".",
    ):
        """Add an object to the specified storage.

        Args:
            name (str): Identifier for the object
            obj (Any): Object to store
            storage_name (str): Type of storage ('problems', 'instances', etc.)
            with_save (bool, optional): Whether to save to disk. Defaults to False.
            save_dir (str, optional): Directory for saving files. Defaults to ".".

        Examples:
            >>> ds = DataStore()
            >>> ds.add("problem1", problem, "problems", with_save=True)
            >>> ds.add("param1", {"value": 42}, "parameters")
        """
        # Add the object to the storage
        getattr(self, storage_name)[name] = obj

        # Save the object to the file
        if with_save:
            # check existance of save_dir
            save_dir = pathlib.Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            match storage_name:
                case "parameters" | "meta_data":
                    # Save the whole dictionary to the file
                    # Example:
                    # (original) .parameters = {"a": 1, "b": 2}
                    # user input: add("new", 3, "parameters")
                    # (new) .parameters = {"a": 1, "b": 2, "new": 3}
                    # save to the file: {"a": 1, "b": 2, "new": 3}
                    storage = self._storage_mapping[storage_name]
                    file_name = self._file_name(
                        storage_name, "", storage.extension
                    )  # noqa
                    attr_value = getattr(self, storage_name)
                    storage.save(attr_value, save_dir / pathlib.Path(file_name))  # noqa
                case _:
                    # save each the object to the file with the name
                    # Example:
                    # (original) .objects = {"a": {"x": 1}}
                    # user input: add("new", {"z": 3}, "objects")
                    # (new) .objects = {"a": {"x": 1}, "new": {"z": 3}}
                    # save to the file (objects_new.json): {"z": 3}
                    storage = self._storage_mapping[storage_name]
                    file_name = self._file_name(
                        storage_name, name, storage.extension
                    )  # noqa
                    storage.save(obj, save_dir / pathlib.Path(file_name))

    @classmethod
    def _file_name(cls, name: str, obj_name: str, extionsion: str):
        return f"{name}_{obj_name}.{extionsion}"

    def save_all(self, path: pathlib.Path):
        """Save all stored data to the specified directory.

        Saves all objects in all storage types to their respective files in the
        specified directory, maintaining the standard directory structure.

        Args:
            path (pathlib.Path): Directory where files should be saved
        """

        # Check path is exist or not
        if not path.exists():
            path.mkdir(parents=True)
        for storage_name, storage in self._storage_mapping.items():
            storage: StorageStrategy
            attr_value: dict = getattr(self, storage_name)
            if storage_name in ("parameters", "meta_data"):
                # Save the whole dictionary to the file
                file_name = self._file_name(storage_name, "", storage.extension)
                storage.save(attr_value, path / file_name)
            else:
                # save each the object to the file with the name
                for obj_name, obj in attr_value.items():
                    file_name = self._file_name(
                        storage_name, obj_name, storage.extension
                    )
                    storage.save(obj, path / file_name)

    @classmethod
    def load(cls, path: pathlib.Path):
        """Load a DataStore from a directory.

        Creates a new DataStore instance and populates it with data from files
        in the specified directory.

        Args:
            path (pathlib.Path): Directory containing the stored files

        Returns:
            DataStore: New DataStore instance containing the loaded data
        """
        data = {}
        for storage_name, storage in cls._storage_mapping.items():
            storage: StorageStrategy
            if storage_name in ("parameters", "meta_data"):
                file_path = path / cls._file_name(
                    storage_name, "", storage.extension
                )  # noqa
                if file_path.exists():
                    data[storage_name] = storage.load(file_path)
            else:
                data[storage_name] = {}
                file_pattern = cls._file_name(storage_name, "*", storage.extension)
                for file_path in path.glob(str(file_pattern)):  # noqa
                    obj_name = "_".join(
                        file_path.stem.split(".")[0].split("_")[1:]
                    )  # noqa
                    data[storage_name][obj_name] = storage.load(file_path)
        return cls(**data)

    def add_to_artifact_builder(
        self, builder: ox_art.ArtifactBuilder, annotations: dict
    ):
        """Add all stored data to an OMMX artifact builder.

        Adds all objects from all storage types to the artifact builder, including
        appropriate annotations for each layer.

        Args:
            builder (ox_art.ArtifactBuilder): The artifact builder
            annotations (dict[str, str]): Base annotations for all layers
        """
        _annotations = copy.copy(annotations)
        for storage_name, storage in self._storage_mapping.items():
            storage: StorageStrategy
            _annotations[self.minto_namespace + ".storage"] = storage_name
            if storage_name in ("parameters", "meta_data"):
                obj = getattr(self, storage_name)
                storage.add_to_artifact_builder(obj, builder, _annotations)
            else:
                for obj_name, obj in getattr(self, storage_name).items():
                    _annotations[self.minto_namespace + ".name"] = obj_name
                    storage.add_to_artifact_builder(obj, builder, _annotations)

    @classmethod
    def load_from_layers(
        cls, artifact: ox_art.Artifact, layers: typ.Iterable[ox_art.Descriptor]
    ):
        """Create a DataStore from OMMX artifact layers.

        Args:
            artifact (ox_art.Artifact): The OMMX artifact containing the data
            layers (Iterable[ox_art.Descriptor]): Layer descriptors to process

        Returns:
            DataStore: New DataStore instance containing the data from the layers
        """
        datastore = cls()
        for layer in layers:
            storage_name = layer.annotations.get(
                cls.minto_namespace + ".storage"
            )  # noqa
            if storage_name is None:
                continue
            storage = cls._storage_mapping[storage_name]
            obj = storage.load_from_layer(artifact, layer)
            if storage_name == "parameters":
                datastore.parameters = obj
            elif storage_name == "meta_data":
                datastore.meta_data = obj
            else:
                obj_name = layer.annotations.get(cls.minto_namespace + ".name")
                if obj_name is None:
                    continue
                getattr(datastore, storage_name)[obj_name] = obj
        return datastore
