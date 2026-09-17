"""Слияние BM25F с гео, фасетами и микрокатегорией на локальном бенчмарке

Все сигналы учатся только на разрешённой части обучающих пар: центры локаций, калибровка
расстояния, калибровка фасетов и классификатор микрокатегорий считаются по fit_pairs,
отложенные запросы туда не попадают.

Веса подбираю по одному, покоординатным спуском, а не сеткой. Сетка на три измерения это
сотни прогонов по двадцать секунд, а метрика тут с ошибкой 0.01, и такая сетка гарантированно
найдёт победителя, завышенного на шум.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from avito_cg.config import PATHS, TOP_K
from avito_cg.data.io import load_train
from avito_cg.eval.benchmark import LocalBenchmark
from avito_cg.eval.metrics import (
    bootstrap_ci,
    paired_bootstrap,
    recall_at_k,
    recall_curve,
    recall_per_query,
)
from avito_cg.index.fields import query_texts
from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import BM25FIndex
from avito_cg.retrieval.fusion import FusionConfig, retrieve
from avito_cg.retrieval.signals import (
    DenseSignal,
    FacetSignal,
    GeoSignal,
    MicrocatSignal,
    load_embeddings,
)

COLUMNS = [
    "item_id",
    "item_latitude",
    "item_longitude",
    "item_microcat_id",
    "item_infm_params_text",
]
TRAIN_COLUMNS = ["search_location_id", "search_query", "search_infm_params_text", *COLUMNS]

GRIDS = {
    "фасеты": (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
    "микрокатегория": (0.0, 0.005, 0.01, 0.02, 0.05, 0.10),
    "плотный": (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8),
    "гео": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.35),
}
START = {"гео": 0.10, "фасеты": 0.10, "микрокатегория": 0.02, "плотный": 0.0}
EMBEDDINGS = "embeddings_local.npy"
ENCODER = "encoder"


def run(*, top_k: int = TOP_K) -> None:
    PATHS.ensure()
    started = time.time()
    train = load_train(columns=TRAIN_COLUMNS)
    local = LocalBenchmark.load(PATHS.interim / "local_benchmark")
    corpus = local.corpus(train, columns=COLUMNS)
    pairs = local.fit_pairs(train)

    geo = GeoIndex.fit(pairs, corpus)
    signals = {
        "гео": GeoSignal.build(geo, local.queries),
        "фасеты": FacetSignal.build(local.queries, corpus, pairs),
        "микрокатегория": MicrocatSignal.build(local.queries, corpus, pairs),
    }
    del train, pairs

    # плотный сигнал появляется только если эмбеддинги уже посчитаны на GPU,
    # см. docs/KAGGLE.md. Без них всё остальное работает как раньше
    embeddings = PATHS.artifacts / EMBEDDINGS
    if embeddings.exists():
        from avito_cg.train.biencoder import QUERY_PREFIX, encode

        item_vectors = load_embeddings(embeddings, corpus["item_id"].astype(str).to_numpy())
        query_vectors = encode(
            local.queries["search_query"].fillna("").astype(str).tolist(),
            PATHS.artifacts / ENCODER,
            prefix=QUERY_PREFIX,
            max_length=32,
            device="cpu",
        )
        signals["плотный"] = DenseSignal(query_vectors=query_vectors, item_vectors=item_vectors)
    else:
        print(f"эмбеддингов нет в {embeddings}, плотный сигнал пропускаю", flush=True)

    print(f"сигналы готовы за {time.time() - started:.0f} c", flush=True)

    index = BM25FIndex.load(PATHS.artifacts / "lexical_local")
    item_ids = corpus["item_id"].astype(str).to_numpy()
    query_ids = local.query_ids
    texts = query_texts(local.queries)

    def rank(weights: dict[str, float], depth: int) -> dict[str, list[str]]:
        order = retrieve(
            index,
            [(signals[name], weight) for name, weight in weights.items()],
            texts,
            top_k=depth,
            config=FusionConfig(mode="normalized"),
        )
        return {
            query: [item_ids[position] for position in row if position >= 0]
            for query, row in zip(query_ids, order, strict=True)
        }

    def evaluate(weights: dict[str, float]) -> tuple[float, np.ndarray]:
        predictions = rank(weights, top_k)
        return recall_at_k(local.relevant, predictions), recall_per_query(
            local.relevant, predictions
        )

    weights = {name: value for name, value in START.items() if name in signals}
    base, previous = evaluate(weights)
    start_scores = previous
    print(f"исходная точка, только гео: {base:.4f}", flush=True)

    for name, grid in GRIDS.items():
        print(f"\nвес сигнала «{name}»", flush=True)
        results = []
        for value in grid:
            score, per_query = evaluate(dict(weights, **{name: value}))
            results.append((value, score, per_query))
            print(f"  {value:5.3f} -> {score:.4f}", flush=True)
        weights[name], _, chosen = max(results, key=lambda row: row[1])

        # абсолютные интервалы тут не помогают: обе конфигурации гоняются на одних
        # и тех же запросах, и общая дисперсия сокращается только в разностях
        delta, low, high = paired_bootstrap(previous, chosen)
        verdict = "значимо" if low > 0 else "в пределах шума"
        print(f"  выбрано {weights[name]}: {delta:+.4f} [{low:+.4f}, {high:+.4f}], {verdict}")
        previous = chosen

    score, per_query = evaluate(weights)
    low, high = bootstrap_ci(per_query)
    print(f"\nИтог: {weights}")
    print(f"Recall@50 = {score:.4f}  95% ДИ [{low:.4f}, {high:.4f}]")
    delta, low, high = paired_bootstrap(start_scores, per_query)
    print(f"прирост к одному гео: {delta:+.4f} [{low:+.4f}, {high:+.4f}]")

    curve = recall_curve(local.relevant, rank(weights, 1000), ks=(1, 10, 50, 100, 200, 500, 1000))
    print("\nГлубина кандидатов")
    print(
        pd.DataFrame([{"k": k, "Recall@k": round(v, 4)} for k, v in curve.items()]).to_string(
            index=False
        )
    )
    print(f"\nвсего {time.time() - started:.0f} c")


if __name__ == "__main__":
    run()
