from __future__ import annotations

import re
from typing import Literal

import jijmodeling as jm
import pandas as pd
from jijzept.response import JijModelingResponse

from minto.v0.records.records import Record
from minto.v0.table.table import SchemaBasedTable


class SampleSetRecord(Record):
    sample_run_id: str
    num_occurrences: int
    objective: float
    is_feasible: bool
    sample_id: int
    deci_var_value: dict[str, jm.experimental.SparseVarValues]
    eval_result: jm.experimental.EvaluationResult


def expand_sampleset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    sampleset_df = dataframe[
        dataframe["content"].apply(
            lambda x: isinstance(
                x, (jm.experimental.SampleSet, jm.SampleSet, JijModelingResponse)
            )
        )
    ]
    df_list = []
    for _, record in sampleset_df.iterrows():
        table = convert_sampleset_to_table(
            record["content"], content_id=record["result_id"], key="result"
        )
        df_list.append(table.dataframe())

    if len(df_list) == 0:
        return pd.DataFrame()
    else:
        return pd.concat(df_list)


def to_valid_name(name: str) -> str:
    return re.sub(r"\W|^(?=\d)", "_", name)


def convert_sampleset_to_table(
    sampleset: jm.experimental.SampleSet | JijModelingResponse,
    content_id: int,
    key: Literal["parameter", "result"],
) -> SchemaBasedTable:
    if isinstance(sampleset, JijModelingResponse):
        sampleset = jm.experimental.from_old_sampleset(sampleset.sample_set)
    elif isinstance(sampleset, jm.SampleSet):
        sampleset = jm.experimental.from_old_sampleset(sampleset)

    schema = SampleSetRecord.dtypes
    schema[f"{key}_id"] = int
    # add constraint violation columns to schema
    if len(sampleset) > 0:
        for constraint_name in sampleset[0].eval.constraints.keys():
            constraint_name = to_valid_name(constraint_name)
            schema[constraint_name + "_total_violation"] = float

    sampleset_table = SchemaBasedTable(schema=schema)
    sampleset_table._validator.model_rebuild(
        _types_namespace={
            "VarType": jm.experimental.VarType,
            "Violation": jm.experimental.Violation,
        }
    )

    for sample_id, sample in enumerate(sampleset):
        record = {
            f"{key}_id": content_id,
            "sample_run_id": sample.run_id,
            "num_occurrences": int(sample.num_occurrences),
            "objective": float(sample.eval.objective),
            "is_feasible": bool(sample.is_feasible()),
            "sample_id": sample_id,
            "deci_var_value": sample.var_values,
            "eval_result": sample.eval,
        }
        # extract constraint total violation
        total_violations: dict[str, float] = {}
        for constraint_name, constraint in sample.eval.constraints.items():
            constraint_name = to_valid_name(constraint_name)
            total_violations[constraint_name + "_total_violation"] = (
                constraint.total_violation
            )
        record.update(total_violations)
        sampleset_table.insert(record)
    return sampleset_table
