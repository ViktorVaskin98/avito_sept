"""Сборка локального бенчмарка и проверка, что он не врёт

Кроме самой сборки гоняю тут две тривиальные модели. Они нужны не ради качества,
а чтобы убедиться, что вся цепочка «разметка - предсказание - метрика» сходится:
случайный отбор должен давать примерно 50/189212, а ближайшие по расстоянию
заметно больше. Если эти два числа выглядят не так, сломан сплит или метрика,
и дальше идти нельзя
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from avito_cg.analysis import haversine, search_location_centroids
from avito_cg.config import PATHS, RANDOM_SEED, TOP_K
from avito_cg.data.io import load_benchmark_items, load_benchmark_queries, load_train
from avito_cg.eval import benchmark as bm
from avito_cg.eval.metrics import bootstrap_ci, recall_at_k, recall_per_query

SPLIT_DIR = "local_benchmark"


def _sanity_baselines(
    local: bm.LocalBenchmark, train: pd.DataFrame, top_k: int = TOP_K
) -> pd.DataFrame:
    corpus = local.corpus(
        train, columns=["item_id", "item_latitude", "item_longitude", "item_location_id"]
    )
    item_ids = corpus["item_id"].astype(str).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)

    random_pick = {
        query_id: item_ids[rng.choice(len(item_ids), size=top_k, replace=False)].tolist()
        for query_id in local.relevant
    }

    centroids = search_location_centroids(local.fit_pairs(train))
    located = local.queries.join(centroids, on="search_location_id")
    query_lat = located["latitude"].to_numpy(dtype=float)
    query_lon = located["longitude"].to_numpy(dtype=float)
    item_lat = corpus["item_latitude"].to_numpy(dtype=float)
    item_lon = corpus["item_longitude"].to_numpy(dtype=float)

    nearest: dict[str, list[str]] = {}
    query_ids = local.queries["query_id"].astype(str).to_numpy()
    for start in range(0, len(query_ids), 128):
        stop = min(start + 128, len(query_ids))
        distance = haversine(
            query_lat[start:stop, None],
            query_lon[start:stop, None],
            item_lat[None, :],
            item_lon[None, :],
        )
        distance = np.where(np.isnan(distance), np.inf, distance)
        order = np.argpartition(distance, top_k, axis=1)[:, :top_k]
        for offset, row in enumerate(order):
            nearest[query_ids[start + offset]] = item_ids[row].tolist()

    records = []
    for name, predictions in (("случайные 50", random_pick), ("50 ближайших по гео", nearest)):
        scores = recall_per_query(local.relevant, predictions, top_k)
        low, high = bootstrap_ci(scores)
        records.append(
            {
                "модель": name,
                "Recall@50": round(recall_at_k(local.relevant, predictions, top_k), 5),
                "95% ДИ": f"[{low:.4f}, {high:.4f}]",
            }
        )
    return pd.DataFrame(records)


def run(*, sanity: bool = True, seen_share: float = bm.SEEN_SHARE, seed: int = RANDOM_SEED) -> None:
    PATHS.ensure()
    started = time.time()
    train = load_train()
    queries = load_benchmark_queries()
    items = load_benchmark_items(with_description=False)

    local = bm.build(train, queries, seen_share=seen_share, seed=seed)
    print(f"собрано за {time.time() - started:.0f} c\n")
    for key, value in local.meta.items():
        print(f"  {key}: {value}")

    print("\nСверка с настоящим бенчмарком")
    print(bm.describe(local, train, queries, items).to_string(index=False))

    problems = bm.leakage_checks(local, train)
    print("\nПроверка на утечки:", "чисто" if not problems else "")
    for problem in problems:
        print(f"  - {problem}")

    if sanity:
        print("\nКонтрольные модели")
        print(_sanity_baselines(local, train).to_string(index=False))

    destination = PATHS.interim / SPLIT_DIR
    local.save(destination)
    print(f"\nсплит сохранён в {destination}")


if __name__ == "__main__":
    run()
