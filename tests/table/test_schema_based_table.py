from __future__ import annotations

from typing import Any

import pandas as pd

import minto

TEST_SCHEMA = {
    "a": int,
    "b": float,
    "c": str,
    "d": list[int],
    "e": dict[str, int],
    "f": Any,
}


def test_generate_table():
    table = minto.SchemaBasedTable(TEST_SCHEMA)

    assert table.empty()


def test_pandas_dtypes():
    table = minto.SchemaBasedTable(TEST_SCHEMA)

    expected_dtypes = {
        "a": "int",
        "b": "float",
        "c": "string",
        "d": "object",
        "e": "object",
        "f": "object",
    }

    assert table.pandas_dtypes == expected_dtypes


def test_generate_table_from_pandas_dataframe():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": ["1", "2", "3"],
            "d": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "e": [{"a": 1}, {"b": 2}, {"c": 3}],
            "f": [None, None, None],
        }
    )

    table = minto.SchemaBasedTable.from_dataframe(df)

    assert not table.empty()
    assert len(table.records) == 3


def test_generate_table_from_dict():
    obj = {
        0: {"a": 1, "b": 1.0, "c": "1", "d": [1, 2, 3], "e": {"a": 1}, "f": None},
        1: {"a": 2, "b": 2.0, "c": "2", "d": [4, 5, 6], "e": {"b": 2}, "f": None},
        2: {"a": 3, "b": 3.0, "c": "3", "d": [7, 8, 9], "e": {"c": 3}, "f": None},
    }

    table = minto.SchemaBasedTable.from_dict(obj)

    assert not table.empty()
    assert len(table.records) == 3


def test_insert_record_into_table():
    table = minto.SchemaBasedTable(TEST_SCHEMA)

    record = pd.Series(
        {"a": 1, "b": 2.0, "c": "3", "d": [4, 5, 6], "e": {"a": 1}, "f": None}
    )
    table.insert(record)

    assert not table.empty()

    for k, v in table.records[0].dict().items():
        assert k in record
        assert record[k] == v
