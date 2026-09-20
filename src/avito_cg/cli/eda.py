"""Полный прогон разбора данных

Считает все таблицы, рисует графики в reports/figures и складывает числа
в reports/eda.json, чтобы ноутбук и docs/EDA.md брали их оттуда, а не пересчитывали
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from avito_cg import analysis, figures
from avito_cg.config import PATHS
from avito_cg.data.io import (
    QUERY_KEY_COLUMNS,
    load_benchmark_items,
    load_benchmark_queries,
    load_train,
)

RADIUS_KM = 25.0
TOP_MICROCATS = 3


def _print(title: str, payload: Any) -> None:
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    if isinstance(payload, pd.DataFrame):
        print(payload.to_string(index=False))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"  {key}: {value}")
    else:
        print(payload)


def run() -> dict[str, Any]:
    """Посчитать весь разбор данных разом: таблицы в reports/eda.json, графики в figures

    Числа из этой команды - единственный источник для docs/EDA.md. Ноутбука,
    который надо прокликать, чтобы получить число из документации, тут нет намеренно
    """
    PATHS.ensure()
    started = time.time()
    train = load_train()
    corpus = load_benchmark_items(with_description=False)
    queries = load_benchmark_queries()
    print(
        f"загружено за {time.time() - started:.1f} c: {len(train)} пар, "
        f"{len(corpus)} объявлений, {len(queries)} запросов"
    )

    results: dict[str, Any] = {}

    counts = analysis.query_item_counts(train)
    results["релевантных на запрос"] = {
        "уникальных запросов": len(counts),
        "среднее": float(counts.mean()),
        "медиана": float(counts.median()),
        "максимум": int(counts.max()),
        **{f"ровно {n}": float((counts == n).mean()) for n in (1, 2, 3)},
    }
    _print("Релевантных объявлений на запрос", results["релевантных на запрос"])
    figures.positives_per_query(counts, PATHS.figures / "positives_per_query.png")

    # описание читаю один раз и переиспользую: это самая тяжёлая колонка в файле
    with_description = load_train(
        columns=[
            "search_query",
            "item_title_raw",
            "item_infm_params_text",
            "item_description_raw",
        ]
    )
    overlap = analysis.overlap_report(with_description, sample=60_000)
    results["лексический разрыв"] = overlap.to_dict("records")
    _print("Лексический разрыв", overlap)
    figures.lexical_coverage(overlap, PATHS.figures / "lexical_coverage.png")

    truncation = analysis.description_truncation_report(with_description)
    results["обрезка описания"] = truncation.to_dict("records")
    _print("Сколько описания достаточно", truncation)
    del with_description

    results["география"] = analysis.location_match_report(train)
    _print("География, плоское совпадение", results["география"])

    centroids = analysis.search_location_centroids(train)
    distance = analysis.distance_report(train, centroids)
    results["радиус"] = distance.to_dict("records")
    _print("Полнота гео-фильтра по радиусу", distance)
    figures.distance_recall(
        distance,
        PATHS.figures / "distance_recall.png",
        exact_match=results["география"]["совпадение id"],
    )
    figures.location_sizes(
        corpus["item_location_id"].value_counts().to_numpy(), PATHS.figures / "location_sizes.png"
    )

    results["фасеты"] = analysis.facet_report(train)
    _print("Фасетный фильтр", results["фасеты"])

    results["микрокатегория по запросу"] = analysis.microcat_concentration(train)
    _print(
        "Концентрация микрокатегорий у повторяющихся запросов", results["микрокатегория по запросу"]
    )

    topk, vectorizer, model = analysis.microcat_classifier_report(train)
    results["классификатор микрокатегорий"] = topk.to_dict("records")
    _print("Предсказание микрокатегории по тексту запроса", topk)
    figures.microcat_topk(topk, PATHS.figures / "microcat_topk.png")

    bias = analysis.selection_bias_report(train, corpus)
    results["смещение выбора"] = bias.to_dict("records")
    _print("Смещение выбора по рейтингу, отзывам и цене", bias)

    shift = analysis.shift_report(train, queries)
    results["сдвиг распределений"] = shift.to_dict("records")
    _print("Обучение против бенчмарка", shift)
    figures.query_length_shift(
        train.drop_duplicates(subset=list(QUERY_KEY_COLUMNS))["search_query"]
        .astype(str)
        .str.split()
        .str.len()
        .to_numpy(dtype=float),
        queries["search_query"].astype(str).str.split().str.len().to_numpy(dtype=float),
        PATHS.figures / "query_length_shift.png",
    )

    duplicates = analysis.duplicate_report(corpus)
    results["дубли"] = duplicates
    _print("Дубли заголовков", duplicates)
    title_counts = corpus["item_title_raw"].astype(str).str.lower().value_counts()
    figures.title_duplicates(
        corpus["item_title_raw"].astype(str).str.lower().map(title_counts).to_numpy(),
        PATHS.figures / "title_duplicates.png",
    )

    ceiling, sizes = analysis.filter_ceiling_report(
        train,
        corpus,
        queries,
        centroids,
        vectorizer,
        model,
        radius_km=RADIUS_KM,
        top_microcats=TOP_MICROCATS,
    )
    results["потолок фильтров"] = ceiling.to_dict("records")
    results["кандидатов после фильтров"] = {
        f"не больше {limit}": float((sizes["всё"] <= limit).mean()) for limit in (50, 200, 1000)
    }
    _print("Потолок полноты против числа кандидатов", ceiling)
    _print("Сколько запросов помещается в ответ целиком", results["кандидатов после фильтров"])
    figures.filter_ceiling(ceiling, PATHS.figures / "filter_ceiling.png")
    figures.candidates_after_filters(sizes["всё"], PATHS.figures / "candidates_after_filters.png")

    destination = PATHS.root / "reports" / "eda.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8"
    )
    print(f"\nчисла сохранены в {destination}")
    print(f"графики в {PATHS.figures}")
    print(f"всего {time.time() - started:.0f} c")
    return results


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


if __name__ == "__main__":
    run()
