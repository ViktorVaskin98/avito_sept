"""Разбор промахов: не «сколько потеряли», а «на чём именно»

Агрегат говорит, что метрика 0.90, и на этом останавливается. Чтобы понять, куда идти
дальше, нужно другое: чем промахи систематически отличаются от попаданий. Если окажется,
что теряются в основном далёкие объявления, значит перекручен вес гео. Если короткие
родовые запросы, значит нужен признак, различающий внутри клона. Если длинные редкие -
наоборот, лексика не дотягивается.

Поэтому для каждого релевантного объявления собираю его признаки и сравниваю распределения
у попавших в топ-50 и у промахнувшихся. Размер эффекта, а не p-value: наблюдений тут
три тысячи, значимым будет что угодно.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from avito_cg.analysis import cliffs_delta, haversine
from avito_cg.data.text import normalize, tokenize


def positive_ranks(
    relevant: Mapping[str, set[str]],
    order: np.ndarray,
    item_ids: np.ndarray,
    query_ids: Sequence[str],
) -> pd.DataFrame:
    """Для каждого релевантного объявления его место в выдаче, -1 если не нашлось"""
    position = {item: index for index, item in enumerate(item_ids)}
    rows = []
    for row, query_id in enumerate(query_ids):
        ranking = {
            int(candidate): rank for rank, candidate in enumerate(order[row]) if candidate >= 0
        }
        for item in relevant.get(query_id, ()):
            index = position.get(item)
            rows.append(
                {
                    "query_id": query_id,
                    "query_row": row,
                    "item_id": item,
                    "item_row": -1 if index is None else index,
                    "rank": -1 if index is None else ranking.get(index, -1),
                }
            )
    return pd.DataFrame(rows)


def describe_positives(
    ranks: pd.DataFrame,
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    query_latitude: np.ndarray,
    query_longitude: np.ndarray,
    top_k: int = 50,
) -> pd.DataFrame:
    """Признаки каждого релевантного объявления и того запроса, по которому его искали"""
    frame = ranks.copy()
    frame["попал"] = (frame["rank"] >= 0) & (frame["rank"] < top_k)
    frame["нашёлся"] = frame["rank"] >= 0

    query_text = queries["search_query"].fillna("").astype(str).to_numpy()
    empty_filter = (
        queries["search_infm_params_text"].fillna("").astype(str).str.len() == 0
    ).to_numpy()
    frame["слов в запросе"] = [len(query_text[row].split()) for row in frame["query_row"]]
    frame["пустой фильтр"] = [bool(empty_filter[row]) for row in frame["query_row"]]

    titles = corpus["item_title_raw"].fillna("").astype(str).to_numpy()
    twins = pd.Series([normalize(text) for text in titles]).value_counts()
    latitude = corpus["item_latitude"].to_numpy(dtype=float)
    longitude = corpus["item_longitude"].to_numpy(dtype=float)

    distance, duplicates, coverage = [], [], []
    for _, row in frame.iterrows():
        item_row, query_row = int(row["item_row"]), int(row["query_row"])
        if item_row < 0:
            distance.append(np.nan)
            duplicates.append(np.nan)
            coverage.append(np.nan)
            continue
        distance.append(
            float(
                haversine(
                    np.array([query_latitude[query_row]]),
                    np.array([query_longitude[query_row]]),
                    np.array([latitude[item_row]]),
                    np.array([longitude[item_row]]),
                )[0]
            )
        )
        duplicates.append(float(twins.get(normalize(titles[item_row]), 1)))
        query_tokens = set(tokenize(query_text[query_row]))
        coverage.append(
            len(query_tokens & set(tokenize(titles[item_row]))) / len(query_tokens)
            if query_tokens
            else np.nan
        )

    frame["расстояние, км"] = distance
    frame["близнецов по заголовку"] = duplicates
    frame["покрытие заголовка"] = coverage
    return frame


def compare_hits_and_misses(described: pd.DataFrame) -> pd.DataFrame:
    """Чем промахи отличаются от попаданий, по каждому признаку

    Дельта Клиффа, а не критерий: на трёх тысячах наблюдений значимым окажется что угодно,
    а решения принимаются по величине расхождения
    """
    hits = described[described["попал"]]
    misses = described[~described["попал"]]
    records = []
    for column in (
        "слов в запросе",
        "расстояние, км",
        "близнецов по заголовку",
        "покрытие заголовка",
    ):
        left = hits[column].to_numpy(dtype=float)
        right = misses[column].to_numpy(dtype=float)
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if left.size == 0 or right.size == 0:
            continue
        records.append(
            {
                "признак": column,
                "медиана у попавших": round(float(np.median(left)), 3),
                "медиана у промахов": round(float(np.median(right)), 3),
                "дельта Клиффа": round(cliffs_delta(right, left), 3),
            }
        )
    return pd.DataFrame(records)


def breakdown(described: pd.DataFrame, column: str, bins: Sequence[float]) -> pd.DataFrame:
    """Доля попаданий по корзинам признака

    Нужна, чтобы увидеть форму зависимости, а не только направление: если попадания
    падают только на хвосте, чинить надо хвост, а не сдвигать веса целиком
    """
    values = described[column].to_numpy(dtype=float)
    labels = np.digitize(values, bins)
    rows = []
    for label in range(len(bins) + 1):
        mask = (labels == label) & np.isfinite(values)
        if mask.sum() < 20:
            continue
        low = "-inf" if label == 0 else f"{bins[label - 1]:g}"
        high = "inf" if label == len(bins) else f"{bins[label]:g}"
        rows.append(
            {
                "корзина": f"[{low}, {high})",
                "релевантных": int(mask.sum()),
                "доля попаданий": round(float(described.loc[mask, "попал"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)
