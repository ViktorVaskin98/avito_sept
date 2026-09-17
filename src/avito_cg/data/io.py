from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from avito_cg.config import PATHS

QUERY_FEATURE_COLUMNS: tuple[str, ...] = (
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
)

QUERY_KEY_COLUMNS: tuple[str, ...] = QUERY_FEATURE_COLUMNS

ITEM_FEATURE_COLUMNS: tuple[str, ...] = (
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
)

TEXT_COLUMNS: frozenset[str] = frozenset(
    {
        "search_query",
        "search_infm_params_text",
        "item_title_raw",
        "item_description_raw",
        "item_infm_params_text",
        "query_id",
        "item_id",
    }
)

_HEAVY_COLUMNS: frozenset[str] = frozenset({"item_description_raw"})


def _types_mapper(dtype: pa.DataType) -> pd.api.extensions.ExtensionDtype | None:
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return pd.ArrowDtype(pa.string())
    return None


def _decimals_to_float(table: pa.Table) -> pa.Table:
    for index, field in enumerate(table.schema):
        if pa.types.is_decimal(field.type):
            table = table.set_column(
                index,
                field.name,
                pc.cast(table.column(index), pa.float64()),
            )
    return table


def available_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Прочитать parquet, взяв только перечисленные колонки

    columns=None читает всё, включая описания, так что делать это стоит осознанно
    """
    if columns is not None:
        present = set(available_columns(path))
        missing = [name for name in columns if name not in present]
        if missing:
            raise KeyError(f"нет таких колонок в {path.name}: {missing}")
    table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    table = _decimals_to_float(table)
    return table.to_pandas(types_mapper=_types_mapper)


def load_train(
    columns: Sequence[str] | None = None, *, with_description: bool = False
) -> pd.DataFrame:
    """Обучающие пары «запрос - выбранное объявление»

    Описание объявления по умолчанию не читаю: целиком оно почти никогда не нужно,
    а стоит половину объёма файла
    """
    if columns is None:
        columns = [
            name
            for name in (*QUERY_FEATURE_COLUMNS, *ITEM_FEATURE_COLUMNS)
            if with_description or name not in _HEAVY_COLUMNS
        ]
    return read_parquet(PATHS.train, columns)


def load_benchmark_items(
    columns: Sequence[str] | None = None, *, with_description: bool = True
) -> pd.DataFrame:
    """Корпус, среди которого ищем

    Тут описание как раз нужно, это часть индекса
    """
    if columns is None:
        columns = [
            name for name in ITEM_FEATURE_COLUMNS if with_description or name not in _HEAVY_COLUMNS
        ]
    return read_parquet(PATHS.benchmark_items, columns)


def load_benchmark_queries(columns: Sequence[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = ("query_id", *QUERY_FEATURE_COLUMNS)
    return read_parquet(PATHS.benchmark_queries, columns)


def query_key(frame: pd.DataFrame, columns: Sequence[str] = QUERY_KEY_COLUMNS) -> pd.Series:
    """Суррогатный идентификатор запроса для train, где своего query_id нет

    Склеиваю признаки через \\x1f, этот символ в текстах не встречается
    """
    parts = [frame[name].astype("string").fillna("") for name in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key.str.cat(part, sep="\x1f")
    return key.rename("query_key")
