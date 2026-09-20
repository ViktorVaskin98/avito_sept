"""Сборка answer.csv по настоящему корпусу

Отдельная команда от baseline: там локальный бенчмарк, здесь настоящие 189 212 объявлений
и 2 452 запроса без разметки. Конфигурация приходит снаружи, чтобы отправляемое было ровно тем,
что я мерил локально, а не «примерно тем же».

Гео здесь учится на всех обучающих парах, а не на fit_pairs: отложенных запросов на этой
стороне нет, прятать не от кого. По той же причине популярность объявления и память кликлога
для переранжировщика считаются по всему train.

Переранжировщик подключается, если он обучен и лежит в артефактах. Обучается он на локальном
бенчмарке командой rerank, сюда приезжает готовым: разметки бенчмарка мы не касаемся вообще.
"""

from __future__ import annotations

import json
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
from avito_cg.retrieval.candidates import collect
from avito_cg.retrieval.fusion import DEFAULT_CONFIG, FusionConfig, retrieve
from avito_cg.retrieval.signals import (
    DenseSignal,
    FacetSignal,
    GeoSignal,
    MicrocatSignal,
    load_embeddings,
)
from avito_cg.train.reranker import CandidateSet, ItemContext, build_features, predict

RERANKER = "reranker.cbm"
# та же глубина, на которой реранкер обучался и мерился
RERANK_DEPTH = 200


def _reranked(
    index,
    signals,
    texts,
    items,
    queries,
    pairs,
    model_path,
    *,
    coordinates,
    extra,
    config,
    top_k,
):
    """Выдача после переранжирования обученной на локальном бенчмарке моделью

    Кандидаты собираю тем же collect, что и при замере: признаки должны считаться
    ровно по той же формуле, иначе модель получит другие единицы и выдача разъедется
    с тем, что я мерил
    """
    from catboost import CatBoostClassifier, CatBoostRanker

    query_ids = queries["query_id"].astype(str).tolist()
    # сбор кандидатов по настоящему корпусу это семь минут, а смена модели секунды.
    # Кэш привязан к набору признаков: поменял FEATURES - удали файл
    cache = PATHS.artifacts / "rerank_benchmark.npz"
    if cache.exists():
        stored = np.load(cache)
        data = CandidateSet(
            query=stored["query"],
            item=stored["item"],
            features=stored["features"],
            label=stored["label"],
        )
        print(f"таблица признаков из кэша {cache.name}, строк {len(data.label)}", flush=True)
    else:
        collected = collect(
            index, signals, texts, depth=RERANK_DEPTH, config=config, extra_candidates=extra
        )
        context = ItemContext.build(items, index, pairs)
        latitude, longitude = coordinates
        data = build_features(
            collected.items,
            collected.lexical,
            collected.scores,
            queries,
            context,
            index,
            query_latitude=latitude,
            query_longitude=longitude,
            query_ids=query_ids,
        )
        np.savez_compressed(
            cache,
            features=data.features,
            label=data.label,
            query=data.query,
            item=data.item,
        )

    kind = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    model = CatBoostRanker() if kind["objective"] != "logloss" else CatBoostClassifier()
    model.load_model(str(model_path))
    print(f"переранжирование моделью «{kind['имя']}», {kind['objective']}", flush=True)
    scores = predict(model, data.features)

    order = np.full((len(query_ids), top_k), -1, dtype=np.int64)
    ranked = np.lexsort((-scores, data.query))
    filled = np.zeros(len(query_ids), dtype=np.int64)
    for position in ranked:
        query = data.query[position]
        if filled[query] < top_k:
            order[query, filled[query]] = data.item[position]
            filled[query] += 1
    return order


def run(
    *,
    fields: Sequence[Field] = DEFAULT_FIELDS,
    output: Path | None = None,
    with_filter: bool = False,
    fusion: FusionConfig | None = DEFAULT_CONFIG,
    weights: dict[str, float] | None = None,
    top_k: int = TOP_K,
    rerank: bool = True,
) -> Path:
    PATHS.ensure()
    destination = output or PATHS.submissions / "answer.csv"
    # подобраны на локальном бенчмарке покоординатным спуском, см. docs/EXPERIMENTS.md
    weights = weights or {"гео": 0.10, "фасеты": 0.10, "микрокатегория": 0.02, "плотный": 0.5}

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
                "item_id",
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
        # для переранжировщика от train нужны только эти две колонки: популярность
        # объявления и память «что выбирали под этот же текст». Остальное отпускаю,
        # полный кадр это полтора гигабайта
        pairs = train[["item_id", "search_query"]].copy()
        del train

        # плотный поиск подключается, только если эмбеддинги посчитаны, см. docs/KAGGLE.md
        extra = None
        embeddings = PATHS.artifacts / "embeddings_benchmark.npy"
        if weights.get("плотный") and embeddings.exists():
            from avito_cg.train.biencoder import QUERY_PREFIX, encode

            item_vectors = load_embeddings(embeddings, items["item_id"].astype(str).to_numpy())
            cache = PATHS.artifacts / "query_vectors_benchmark.npy"
            if cache.exists():
                query_vectors = np.load(cache)
            else:
                query_vectors = encode(
                    queries["search_query"].fillna("").astype(str).tolist(),
                    PATHS.artifacts / "encoder",
                    prefix=QUERY_PREFIX,
                    max_length=32,
                    device="cpu",
                )
                np.save(cache, query_vectors)
            dense = DenseSignal(query_vectors=query_vectors, item_vectors=item_vectors)
            signals.append((dense, weights["плотный"]))
            # плотный поиск не только переупорядочивает, но и приводит кандидатов,
            # до которых лексика не дотягивается вовсе
            extra = dense.top_candidates(top_k=200)
        elif weights.get("плотный"):
            print(f"эмбеддингов нет в {embeddings}, плотный сигнал пропускаю", flush=True)
        latitude, longitude = geo.query_coordinates(queries["search_location_id"].to_numpy())
        model_path = PATHS.artifacts / RERANKER
        # переранжировщик обучен на полном наборе сигналов, включая плотный. Без него
        # часть признаков посчитать нечем, и применять модель бессмысленно
        available = {signal.name for signal, _ in signals}
        if rerank and model_path.exists() and "плотный" in available:
            order = _reranked(
                index,
                signals,
                texts,
                items,
                queries,
                pairs,
                model_path,
                coordinates=(latitude, longitude),
                extra=extra,
                config=fusion,
                top_k=top_k,
            )
        else:
            if rerank:
                print(f"без переранжирования, порядок слияния ({model_path})", flush=True)
            order = retrieve(
                index,
                signals,
                texts,
                top_k=top_k,
                config=fusion,
                padding=geo,
                query_coordinates=(latitude, longitude),
                extra_candidates=extra,
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
