from __future__ import annotations

import datetime
from typing import Any, Callable, Type, get_type_hints

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)


class Record(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    def dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.model_fields}

    def series(self) -> pd.Series[Any]:
        return pd.Series(self.dict())

    @classmethod
    @property
    def dtypes(cls) -> dict[str, Type[Any]]:
        return get_type_hints(cls)


class Index(Record):
    experiment_name: StrictStr
    run_id: StrictInt
    # TODO: New attribute will be added.
    # date: datetime.datetime


class SolverInfo(Record):
    experiment_name: StrictStr
    run_id: StrictInt
    solver_name: StrictStr
    source: StrictStr
    solver_id: StrictInt


class SolverContent(Record):
    solver_id: StrictInt
    content: Callable[..., Any]


class ParameterInfo(Record):
    experiment_name: StrictStr
    run_id: StrictInt
    parameter_name: StrictStr
    parameter_id: StrictInt


class ParameterContent(Record):
    parameter_id: StrictInt
    content: Any


class ResultInfo(Record):
    experiment_name: StrictStr
    run_id: StrictInt
    result_name: StrictStr
    result_id: StrictInt


class ResultContent(Record):
    """ResultValue

    Attributes:
        value_id (int): value id
        value (Any): value
    """

    result_id: StrictInt
    content: Any


def get_pandas_dtypes(dtypes: dict[str, Type[Any]]) -> dict[str, str]:
    """isinstance

    Args:
        dtypes (dict[str, Type[Any]]): dtypes

    Returns:
        T: value
    """

    uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
    int_types = [int, np.int8, np.int16, np.int32, np.int64]
    float_types = [float, np.float16, np.float32, np.float64]
    complex_types = [complex, np.complex64, np.complex128]

    pandas_dtypes: dict[str, str] = {}
    for k, v in dtypes.items():
        if v is StrictInt:
            pandas_dtypes[k] = "int"
        elif v is StrictFloat:
            pandas_dtypes[k] = "float"
        elif v in (bool, StrictBool):
            pandas_dtypes[k] = "boolean"
        elif v in (str, StrictStr):
            pandas_dtypes[k] = "string"
        elif v in (datetime.date, datetime.datetime):
            pandas_dtypes[k] = "datetime64[ns]"
        elif v is datetime.timedelta:
            pandas_dtypes[k] = "timedelta64[ns]"
        elif v in uint_types + int_types + float_types + complex_types:
            pandas_dtypes[k] = v.__name__
        else:
            pandas_dtypes[k] = "object"
    return pandas_dtypes


def get_simple_dypes(dtypes: dict[str, str]) -> dict[str, Type[Any]]:
    uint_types = ["uint8", "uint16", "uint32", "uint64"]
    int_types = ["int8", "int16", "int32", "int64"]
    float_types = ["float16", "float32", "float64"]
    complex_types = ["complex64", "complex128"]
    bool_types = ["bool", "boolean"]

    simple_dtypes: dict[str, Type[Any]] = {}
    for k, v in dtypes.items():
        if v is uint_types:
            simple_dtypes[k] = np.uint64
        elif v in int_types:
            simple_dtypes[k] = int
        elif v in float_types:
            simple_dtypes[k] = float
        elif v in complex_types:
            simple_dtypes[k] = complex
        elif v in bool_types:
            simple_dtypes[k] = bool
        elif v == "string":
            simple_dtypes[k] = str
        elif v == "datetime64[ns]":
            simple_dtypes[k] = datetime.datetime
        elif v == "timedelta64[ns]":
            simple_dtypes[k] = datetime.timedelta
        else:
            simple_dtypes[k] = Any
    return simple_dtypes
