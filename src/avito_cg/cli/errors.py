"""Разбор промахов финальной конфигурации на локальном бенчмарке

Условие спрашивает прямым текстом: какие типы ошибок нашёл при анализе и что с ними
сделал. Раньше эти числа - дельта Клиффа, доля попаданий по корзинам расстояния
и покрытия - считались разово в сессии, а в репозитории лежал только модуль
`eval/errors.py`, который никто не вызывал. То есть ответ на вопрос из условия
не воспроизводился ни одной командой. Эта команда его воспроизводит.

Считается на той же таблице признаков, что и замеры переранжирования, так что
разбор описывает ровно ту конфигурацию, которая уходит на платформу:

    uv run avito-cg errors
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from avito_cg.cli.rerank import COLUMNS, DEPTH, TRAIN_COLUMNS, prepare
from avito_cg.config import PATHS, TOP_K
from avito_cg.data.io import load_train
from avito_cg.eval.benchmark import LocalBenchmark
from avito_cg.eval.errors import (
    breakdown,
    compare_hits_and_misses,
    describe_positives,
    positive_ranks,
)
from avito_cg.eval.metrics import recall_at_k
from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import BM25FIndex
from avito_cg.retrieval.signals import load_embeddings
from avito_cg.train.reranker import ItemContext, predict, ranked_items

DISTANCE_BINS = (1.0, 5.0, 25.0, 100.0, 500.0)
COVERAGE_BINS = (1e-9, 0.5, 0.999)
TWIN_BINS = (2.0, 5.0, 20.0, 100.0)


def run(*, top_k: int = TOP_K) -> pd.DataFrame:
    """Посчитать, чем промахи систематически отличаются от попаданий

    Агрегат говорит «метрика 0.94» и на этом останавливается. Чтобы понять, куда идти
    дальше, нужно другое: если теряются далёкие объявления - перекручен вес гео, если
    короткие родовые запросы - нужен признак, различающий внутри клона. Поэтому для
    каждого релевантного объявления собираются его признаки, и сравниваются распределения
    у попавших в топ-k и у промахнувшихся - по размеру эффекта, а не по p-value:
    наблюдений три тысячи, значимым окажется что угодно
    """
    PATHS.ensure()
    started = time.time()
    train = load_train(columns=TRAIN_COLUMNS)
    local = LocalBenchmark.load(PATHS.interim / "local_benchmark")
    corpus = local.corpus(train, columns=COLUMNS)
    pairs = local.fit_pairs(train)
    geo = GeoIndex.fit(pairs, corpus)
    index = BM25FIndex.load(PATHS.artifacts / "lexical_local")
    context = ItemContext.build(corpus, index, pairs)
    item_ids = context.item_ids
    item_vectors = load_embeddings(PATHS.artifacts / "embeddings_local.npy", item_ids)
    print(f"подготовка за {time.time() - started:.0f} c", flush=True)

    _, data, query_ids = prepare(
        local.queries,
        local.relevant,
        corpus,
        pairs,
        geo,
        index,
        context,
        item_vectors,
        np.load(PATHS.artifacts / "query_vectors_local.npy"),
        cache=PATHS.artifacts / "rerank_local.npz",
    )

    model_path = PATHS.artifacts / "reranker.cbm"
    if model_path.exists():
        from catboost import CatBoostClassifier, CatBoostRanker

        kind = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        model = CatBoostRanker() if kind["objective"] != "logloss" else CatBoostClassifier()
        model.load_model(str(model_path))
        scores = predict(model, data.features)
        label = f"после переранжирования моделью «{kind['имя']}»"
    else:
        # без модели разбираю порядок слияния: кандидаты в таблице уже отсортированы
        # по итоговому скору, так что убывающая шкала это просто обратная позиция
        scores = -np.arange(len(data.label), dtype=np.float64)
        label = "порядок слияния, переранжировщика нет"

    order = ranked_items(data, scores, len(query_ids), top_k=DEPTH)
    predictions = {
        query: [str(item_ids[position]) for position in row[:top_k] if position >= 0]
        for query, row in zip(query_ids, order, strict=True)
    }
    print(f"\n{label}")
    print(f"Recall@{top_k} = {recall_at_k(local.relevant, predictions, top_k):.4f}")

    latitude, longitude = geo.query_coordinates(local.queries["search_location_id"].to_numpy())
    ranks = positive_ranks(local.relevant, order, item_ids, query_ids)
    described = describe_positives(
        ranks,
        local.queries,
        corpus,
        query_latitude=latitude,
        query_longitude=longitude,
        top_k=top_k,
    )

    total = len(described)
    hit = int(described["попал"].sum())
    deeper = int((described["нашёлся"] & ~described["попал"]).sum())
    print(
        f"\nрелевантных {total}: в топ-{top_k} {hit}, "
        f"глубже в пуле {deeper}, вне пула {total - hit - deeper}"
    )

    print("\nЧем промахи отличаются от попаданий")
    print(compare_hits_and_misses(described).to_string(index=False))

    for column, bins in (
        ("расстояние, км", DISTANCE_BINS),
        ("покрытие заголовка", COVERAGE_BINS),
        ("близнецов по заголовку", TWIN_BINS),
    ):
        print(f"\nДоля попаданий по «{column}»")
        print(breakdown(described, column, bins).to_string(index=False))

    destination = PATHS.root / "reports" / "errors.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # служебные номера строк наружу не нужны, а шестнадцать знаков после запятой
    # только мешают читать глазами
    described.drop(columns=["query_row", "item_row"]).round(4).to_csv(
        destination, index=False, encoding="utf-8"
    )
    print(f"\nпострочный разбор в {destination}")
    return described


if __name__ == "__main__":
    run()
