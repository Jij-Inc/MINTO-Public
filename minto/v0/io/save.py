from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Literal

import jijmodeling as jm
import numpy as np
import pandas as pd
from jijzept.response import JijModelingResponse

from minto.v0.experiment.experiment import Experiment


def save(experiment: Experiment) -> None:
    """Save the experiment to a file.

    Args
    ----------
    experiment : Experiment
        The experiment to save.

    Raises
    ----------
    ValueError
        If the experiment does not have a database attribute.

    Examples
    ----------
    save [minto.v0.Experiment][minto.v0.experiment.experiment.Experiment] object to a file.
    This function is called as a method of [minto.v0.Experiment][minto.v0.experiment.experiment.Experiment] object.
    >>> import minto
    >>> experiment = minto.v0.Experiment()
    >>> experiment.save()
    """
    database: DatabaseSchema = getattr(experiment, "database")

    base_dir = experiment.savedir / experiment.name

    keys: list[Literal["index", "solver", "parameter", "result"]] = [
        "index",
        "solver",
        "parameter",
        "result",
    ]
    dtypes = {}
    for key in keys:
        dtypes[key] = {"info": {}, "content": {}}

    for key in keys:
        if key == "index":
            with open(base_dir / "index.json", "w") as f:
                df = database[key].dataframe()
                index = {
                    "experiment_name": experiment.name,
                    "run_id_range": [
                        int(df["run_id"].min()),
                        int(df["run_id"].max()) + 1,
                    ],
                }
                json.dump(index, f)
            dtypes[key] = database[key].pandas_dtypes
        else:
            data_dir = base_dir / key

            # get dtypes
            dtypes[key]["info"] = database[key]["info"].pandas_dtypes
            dtypes[key]["content"] = database[key]["content"].pandas_dtypes

            # save info
            with open(data_dir / "info.json", "w") as f:
                df = database[key]["info"].dataframe()
                if df.empty:
                    info = {
                        "experiment_name": experiment.name,
                        "runs": {},
                    }
                else:
                    info = {
                        "experiment_name": experiment.name,
                        "runs": df.groupby(f"{key}_name")[["run_id", f"{key}_id"]]
                        .apply(lambda x: x.to_dict(orient="list"))
                        .to_dict(),
                    }
                json.dump(info, f)

            # save content
            _save_content(database, key, base_dir)

    with open(base_dir / "dtypes.json", "w") as f:
        json.dump(dtypes, f)


class _NumpyEncoder(json.JSONEncoder):
    def default(experiment, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _save_content(
    database: DatabaseSchema,
    key: Literal["solver", "parameter", "result"],
    base_dir: pathlib.Path,
) -> None:
    others = {}
    for index, record in database[key]["content"].dict().items():
        content_id = record[f"{key}_id"]
        content = record["content"]

        # save problem
        if isinstance(content, jm.Problem):
            with open(base_dir / f"{key}" / "problems" / f"{content_id}.pb", "wb") as f:
                f.write(jm.to_protobuf(content))

        # save jijzept JijModelingResponse
        elif isinstance(content, JijModelingResponse):
            sampleset = content.get_sampleset()
            with open(
                base_dir / f"{key}" / "samplesets" / f"{content_id}.json", "w"
            ) as f:
                json.dump(sampleset.to_dict(), f, cls=_NumpyEncoder)

        # save jijmodeling SampleSet
        elif isinstance(content, jm.SampleSet):
            sampleset = jm.experimental.from_old_sampleset(content)
            with open(
                base_dir / f"{key}" / "samplesets" / f"{content_id}.json", "w"
            ) as f:
                json.dump(sampleset.to_dict(), f, cls=_NumpyEncoder)

        # save jijmodeling experimental SampleSet
        elif isinstance(content, jm.experimental.SampleSet):
            with open(
                base_dir / f"{key}" / "samplesets" / f"{content_id}.json", "w"
            ) as f:
                json.dump(content.to_dict(), f, cls=_NumpyEncoder)

        elif is_dataclass(content):
            with open(
                base_dir / f"{key}" / "dataclasses" / f"{content_id}.json", "w"
            ) as f:
                content = {
                    "name": content.__class__.__name__,
                    "type": {field.name: repr(field.type) for field in fields(content)},
                    "data": asdict(content),
                }
                json.dump(content, f, cls=_NumpyEncoder)

        # save other
        else:
            for name, value in record.items():
                if key == "solver" and name == "content":
                    value = database[key]["info"].dict()[index]["source"]
                others.setdefault(index, {})[name] = value
    if len(others):
        others = pd.DataFrame(others).T
    else:
        others = pd.DataFrame(columns=[f"{key}_id", "content"])
    others.to_csv(base_dir / f"{key}" / "content.csv", index=False)
