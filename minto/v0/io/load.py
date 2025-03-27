from __future__ import annotations

import json
import pathlib
from dataclasses import make_dataclass
from typing import TYPE_CHECKING, Literal

import jijmodeling as jm
import numpy as np
import pandas as pd

from minto.v0.consts.default import DEFAULT_RESULT_DIR
from minto.v0.experiment.experiment import Experiment
from minto.v0.table.table import SchemaBasedTable

if TYPE_CHECKING:
    from minto.v0.experiment.experiment import DatabaseSchema


def load(
    experiment_name: str,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
) -> Experiment:
    savedir = pathlib.Path(savedir)
    """
    Loads an experiment from the experiment name and save directory. This
    function rebuilds the Experiment instance.

    Args
    ----------
    experiment_name : str
        The name of the experiment to load. This name is used to locate the
        corresponding directory within `savedir`.
    savedir : str, pathlib.Path, optional
        The base directory where experiments are saved. Defaults to
        ./.minto_experiments.

    Returns
    ----------
    Experiment
        Experiment instance loaded from the save directory.

    Examples
    ----------
    >>> import minto
    >>> exp = minto.v0.load("experiment_name")
    """
    if not (savedir / experiment_name).exists():
        raise FileNotFoundError(f"{(savedir / experiment_name)} is not found.")

    exp = Experiment(experiment_name, savedir=savedir)

    database: DatabaseSchema = getattr(exp, "database")

    base_dir = savedir / experiment_name
    with open(base_dir / "dtypes.json", "r") as f:
        dtypes = json.load(f)

    keys: list[Literal["index", "solver", "parameter", "result"]] = [
        "index",
        "solver",
        "parameter",
        "result",
    ]
    for key in keys:
        if key == "index":
            with open(base_dir / "index.json", "r") as f:
                obj = json.load(f)
                run_ids = range(*obj["run_id_range"])
                index = pd.DataFrame(
                    {
                        "experiment_name": [obj["experiment_name"]] * len(run_ids),
                        "run_id": run_ids,
                    }
                )
            database["index"] = SchemaBasedTable.from_dataframe(index)
        else:
            with open(base_dir / f"{key}" / "info.json", "r") as f:
                obj = json.load(f)
                info = pd.DataFrame(
                    [
                        {
                            "experiment_name": obj["experiment_name"],
                            f"{key}_name": name,
                            "run_id": rid,
                            f"{key}_id": kid,
                        }
                        for name, runs in obj["runs"].items()
                        for rid, kid in zip(runs["run_id"], runs[f"{key}_id"])
                    ],
                    columns=["experiment_name", "run_id", f"{key}_id", f"{key}_name"],
                )
                if key == "solver":
                    info["source"] = pd.read_csv(
                        base_dir / f"{key}" / "content.csv", usecols=["content"]
                    )["content"]
            database[key]["info"] = SchemaBasedTable.from_dataframe(info)

            content = pd.read_csv(base_dir / f"{key}" / "content.csv")

            if key in ("parameter", "result"):
                problems = {}
                for i, file in enumerate(
                    (exp.savedir / exp.name / f"{key}" / "problems").glob("*")
                ):
                    with open(file, "rb") as f:
                        content_id = int(file.name.split(".")[0])
                        problem = jm.from_protobuf(f.read())

                        problems[i] = {f"{key}_id": content_id, "content": problem}
                content = pd.concat([content, pd.DataFrame(problems).T]).reset_index(
                    drop=True
                )

                samplesets = {}
                for i, file in enumerate(
                    (exp.savedir / exp.name / f"{key}" / "samplesets").glob("*")
                ):
                    with open(file, "r") as f:
                        content_id = int(file.name.split(".")[0])
                        sampleset = jm.experimental.SampleSet.from_dict(json.load(f))
                        samplesets[i] = {f"{key}_id": content_id, "content": sampleset}
                content = pd.concat([content, pd.DataFrame(samplesets).T]).reset_index(
                    drop=True
                )

                dc_objs = {}
                for i, file in enumerate(
                    (exp.savedir / exp.name / f"{key}" / "dataclasses").glob("*")
                ):
                    with open(file, "r") as f:
                        content_id = int(file.name.split(".")[0])
                        json_obj = json.load(f)
                        dc_obj = make_dataclass(json_obj["name"], json_obj["type"])(
                            **json_obj["data"]
                        )

                        dc_objs[i] = {f"{key}_id": content_id, "content": dc_obj}
                content = pd.concat([content, pd.DataFrame(dc_objs).T]).reset_index(
                    drop=True
                )

            if content.empty:
                content = pd.DataFrame(columns=dtypes[key]["content"])
            content = content.astype(dtypes[key]["content"])
            content = content[content[f"{key}_id"].isin(info[f"{key}_id"])].sort_values(
                f"{key}_id"
            )
            database[key]["content"] = SchemaBasedTable.from_dataframe(content)
    return exp
