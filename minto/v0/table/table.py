from __future__ import annotations

from typing import Any, Type

import pandas as pd
from pandas import DataFrame
from pydantic import create_model

from minto.v0.records.records import Record, get_pandas_dtypes, get_simple_dypes
from minto.v0.typing import ArtifactDataType


class SchemaBasedTable:
    def __init__(self, schema: dict[str, Type[Any]]) -> None:
        self._schema = schema
        self._pandas_dtypes = get_pandas_dtypes(schema)

        field_definitions: dict[str, Any] = {k: (v, ...) for k, v in schema.items()}
        self._validator = create_model(
            "Validator", __base__=Record, **field_definitions
        )
        self._records: list[Record] = []

    def __len__(self) -> int:
        return self.records.__len__()

    def __str__(self) -> str:
        return self.records.__str__()

    def __getitem__(self, i: int) -> Record:
        return self.records[i]

    @property
    def records(self) -> list[Record]:
        return self._records

    @property
    def schema(self) -> dict[str, Type[Any]]:
        return self._schema

    @property
    def pandas_dtypes(self) -> dict[str, Any]:
        return self._pandas_dtypes

    @classmethod
    def from_dataframe(cls, dataframe: DataFrame) -> SchemaBasedTable:
        schema = get_simple_dypes({k: v for k, v in dataframe.dtypes.items()})
        table = cls(schema)
        for record in dataframe.itertuples(index=False):
            table.insert(record)
        return table

    @classmethod
    def from_dict(cls, dct: ArtifactDataType) -> SchemaBasedTable:
        if dct:
            schema = get_simple_dypes(
                {k: type(v).__name__ for k, v in list(dct.values())[0].items()}
            )
            table = cls(schema)
            for record in dct.values():
                table.insert(pd.Series(record))
            return table
        else:
            return cls({})

    def insert(
        self,
        record: dict[str, Any] | Record | pd.Series[Any] | list[Any] | tuple[Any, ...],
    ) -> None:
        """Insert a new record.

        Args
        ----------
        record : dict[str, Any] | Record | Series[Any] | list[Any] | tuple[Any, ...]
            The record to be appended.
        """

        record = self._validate_record(record)
        self.records.append(record)

    def _validate_record(
        self,
        record: dict[str, Any] | Record | pd.Series[Any] | list[Any] | tuple[Any, ...],
    ) -> Record:
        if isinstance(record, Record):
            record = record.dict()
        elif isinstance(record, pd.Series):
            record = record.to_dict()
        elif isinstance(record, (list, tuple)):
            record = {k: v for k, v in zip(self._schema, record)}
        return self._validator(**record)

    def empty(self) -> bool:
        """Returns True if the Container is empty."""
        return len(self) == 0

    def dataframe(self) -> DataFrame:
        data: dict[str, list[Any]] = {}
        for record in self.records:
            for k, v in record.dict().items():
                data.setdefault(k, []).append(v)

        df = pd.DataFrame(data, columns=self.pandas_dtypes)
        return df.astype(self.pandas_dtypes)

    def dict(self) -> ArtifactDataType:
        return self.dataframe().to_dict(orient="index")
