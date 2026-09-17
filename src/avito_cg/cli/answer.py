"""Сборка answer.csv по настоящему корпусу

Отдельная команда от baseline: там локальный бенчмарк, здесь настоящие 189 212 объявлений
и 2 452 запроса без разметки. Конфигурация приходит снаружи, чтобы отправляемое было ровно тем,
что я мерил локально, а не «примерно тем же».

Гео здесь учится на всех обучающих парах, а не на fit_pairs: отложенных запросов на этой
стороне нет, прятать не от кого.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from avito_cg.config import PATHS, TOP_K
from avito_cg.data.io import load_benchmark_items, load_benchmark_queries, load_train
from avito_cg.eval.submission import save_submission
from avito_cg.index.fields import PARAM_KEYS_FILE, item_fields, load_parser, query_texts
from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import DEFAULT_FIELDS, BM25FIndex, Field
from avito_cg.retrieval.fusion import DEFAULT_CONFIG, FusionConfig, retrieve
from avito_cg.retrieval.signals import FacetSignal, GeoSignal, MicrocatSignal


def run(
    *,
    fields: Sequence[Field] = DEFAULT_FIELDS,
    output: Path | None = None,
    with_filter: bool = False,
    fusion: FusionConfig | None = DEFAULT_CONFIG,
    weights: dict[str, float] | None = None,
    top_k: int = TOP_K,
) -> Path:
    PATHS.ensure()
    destination = output or PATHS.submissions / "answer.csv"
    weights = weights or {"гео": 0.10, "фасеты": 0.0, "микрокатегория": 0.0}

    started = time.time()
    items = load_benchmark_items()
    queries = load_benchmark_queries()
    print(
        f"загружено за {time.time() - started:.0f} c: "
        f"{len(items)} объявлений, {len(queries)} запросов"
    )

    parser = load_parser(
        items["item_infm_params_text"].astype(str).tolist(), PATHS.interim / PARAM_KEYS_FILE
    )

    cache = PATHS.artifacts / "lexical_benchmark"
    started = time.time()
    if BM25FIndex.exists(cache):
        index = BM25FIndex.load(cache, fields)
        print(f"индекс прочитан из кэша за {time.time() - started:.0f} c", flush=True)
    else:
        index = BM25FIndex(fields).fit(item_fields(items, parser))
        index.save(cache)
        print(f"индекс построен за {time.time() - started:.0f} c", flush=True)

    started = time.time()
    texts = query_texts(queries, with_filter=with_filter)
    if fusion is None:
        order, _ = index.search(texts, top_k=top_k)
        print(f"поиск за {time.time() - started:.0f} c, только лексика", flush=True)
    else:
        train = load_train(
            columns=[
                "search_location_id",
                "search_query",
                "search_infm_params_text",
                "item_latitude",
                "item_longitude",
                "item_microcat_id",
                "item_infm_params_text",
            ]
        )
        geo = GeoIndex.fit(train, items)
        signals = [
            (GeoSignal.build(geo, queries), weights["гео"]),
            (FacetSignal.build(queries, items, train), weights["фасеты"]),
            (MicrocatSignal.build(queries, items, train), weights["микрокатегория"]),
        ]
        del train
        latitude, longitude = geo.query_coordinates(queries["search_location_id"].to_numpy())
        order = retrieve(
            index,
            signals,
            texts,
            top_k=top_k,
            config=fusion,
            padding=geo,
            query_coordinates=(latitude, longitude),
        )
        print(
            f"поиск за {time.time() - started:.0f} c, лексика + "
            + ", ".join(f"{signal.name}={weight}" for signal, weight in signals),
            flush=True,
        )

    item_ids = items["item_id"].astype(str).to_numpy()
    query_ids = queries["query_id"].astype(str).tolist()
    predictions = {
        query_id: [item_ids[position] for position in row if position >= 0]
        for query_id, row in zip(query_ids, order, strict=True)
    }

    # пустые места в ответе это чистая потеря: метрика не штрафует за лишних кандидатов,
    # так что недобор слотов стоит видеть сразу
    filled = np.array([len(items_) for items_ in predictions.values()])
    print(
        f"кандидатов на запрос: медиана {int(np.median(filled))}, "
        f"запросов с неполным ответом {(filled < top_k).mean():.3f}, "
        f"совсем пустых {(filled == 0).sum()}"
    )

    save_submission(
        predictions,
        destination,
        expected_query_ids=query_ids,
        corpus_item_ids=set(item_ids),
        top_k=top_k,
    )
    print(f"ответ записан в {destination}")
    return destination


if __name__ == "__main__":
    run()
