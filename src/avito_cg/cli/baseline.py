"""Первая настоящая модель: BM25F по трём полям

Здесь же проверяю, отзывается ли локальный бенчмарк на осмысленные изменения.
Порядок замеров повторяет таблицу лексического покрытия из EDA: сначала один заголовок,
потом плюс параметры, потом плюс описание. Если метрика на этих трёх шагах не растёт,
значит сломан не BM25, а сплит или метрика.

Веса полей подбираю перевзвешиванием уже построенного индекса: токенизация корпуса
занимает минуты, а пересборка матрицы секунды, так что сетка обходится дёшево.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
import pandas as pd

from avito_cg.config import PATHS, TOP_K
from avito_cg.data.io import load_train
from avito_cg.eval.benchmark import LocalBenchmark
from avito_cg.eval.metrics import bootstrap_ci, recall_at_k, recall_curve, recall_per_query
from avito_cg.index.fields import PARAM_KEYS_FILE, item_fields, load_parser, query_texts
from avito_cg.index.lexical import BM25FIndex, Field

CORPUS_COLUMNS = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
SPLIT_DIR = "local_benchmark"
CANDIDATE_DEPTH = 1000


def _predictions(
    order: np.ndarray, item_ids: np.ndarray, query_ids: list[str], top_k: int
) -> dict[str, list[str]]:
    """Индексы строк корпуса превратить в item_id, отбросив пустые места"""
    return {
        query_id: [item_ids[position] for position in row[:top_k] if position >= 0]
        for query_id, row in zip(query_ids, order, strict=True)
    }


def _score(
    relevant: dict[str, set[str]],
    order: np.ndarray,
    item_ids: np.ndarray,
    query_ids: list[str],
) -> dict[str, float]:
    predictions = _predictions(order, item_ids, query_ids, TOP_K)
    scores = recall_per_query(relevant, predictions)
    low, high = bootstrap_ci(scores)
    return {"Recall@50": recall_at_k(relevant, predictions), "низ": low, "верх": high}


def run(*, grid: bool = True) -> None:
    PATHS.ensure()
    started = time.time()

    train = load_train(columns=CORPUS_COLUMNS)
    local = LocalBenchmark.load(PATHS.interim / SPLIT_DIR)
    corpus = local.corpus(train, columns=CORPUS_COLUMNS)
    del train
    print(f"корпус {len(corpus)} объявлений загружен за {time.time() - started:.0f} c", flush=True)

    parser = load_parser(
        corpus["item_infm_params_text"].astype(str).tolist(), PATHS.interim / PARAM_KEYS_FILE
    )

    cache = PATHS.artifacts / "lexical_local"
    started = time.time()
    if BM25FIndex.exists(cache):
        index = BM25FIndex.load(cache)
        print(f"индекс прочитан из кэша за {time.time() - started:.0f} c", flush=True)
    else:
        index = BM25FIndex().fit(item_fields(corpus, parser))
        index.save(cache)
        print(f"индекс построен за {time.time() - started:.0f} c", flush=True)
    print(f"словарь {len(index.vocabulary)} термов", flush=True)

    item_ids = corpus["item_id"].astype(str).to_numpy()
    query_ids = local.query_ids
    plain = query_texts(local.queries)
    with_filter = query_texts(local.queries, with_filter=True)

    # словарь и idf считаются один раз по всем трём полям, а поля отключаются
    # обнулением веса. Поэтому «только заголовок» это не совсем отдельный индекс
    # по заголовкам: idf там остаётся общим, и частые в описаниях слова
    # получают низкий вес даже при поиске по одному заголовку
    print("\nЧто даёт каждое поле")
    ablation = [
        ("только заголовок", [Field("title", 1.0, 0.6)]),
        ("заголовок + параметры", [Field("title", 1.0, 0.6), Field("params", 1.0, 0.75)]),
        (
            "заголовок + параметры + описание",
            [Field("title", 1.0, 0.6), Field("params", 1.0, 0.75), Field("description", 1.0, 0.75)],
        ),
    ]
    records = []
    for label, fields in ablation:
        index.reweight(fields)
        started = time.time()
        order, _ = index.search(plain, top_k=TOP_K)
        records.append(
            {"конфигурация": label, **_score(local.relevant, order, item_ids, query_ids)}
        )
        records[-1]["секунд"] = round(time.time() - started, 1)
    print(pd.DataFrame(records).round(4).to_string(index=False))

    best_fields = [
        Field("title", 1.0, 0.6),
        Field("params", 1.0, 0.75),
        Field("description", 1.0, 0.75),
    ]
    if grid:
        print("\nПодбор весов полей")
        # параметры с весом 1 метрику роняют (0.2773 -> 0.2613 на первом прогоне):
        # это 140 токенов родовых фасетных значений, общих для тысяч объявлений,
        # поэтому сетка идёт от нуля, а не вокруг единицы
        results = []
        for title_boost, params_boost, description_boost in product(
            (1.0, 2.0, 3.0, 5.0, 8.0), (0.0, 0.1, 0.25, 0.5), (0.5, 1.0, 2.0)
        ):
            fields = [
                Field("title", title_boost, 0.6),
                Field("params", params_boost, 0.75),
                Field("description", description_boost, 0.75),
            ]
            index.reweight(fields)
            order, _ = index.search(plain, top_k=TOP_K)
            predictions = _predictions(order, item_ids, query_ids, TOP_K)
            results.append(
                {
                    "заголовок": title_boost,
                    "параметры": params_boost,
                    "описание": description_boost,
                    "Recall@50": recall_at_k(local.relevant, predictions),
                }
            )
        table = pd.DataFrame(results).sort_values("Recall@50", ascending=False)
        print(table.head(10).round(4).to_string(index=False))
        # 60 конфигураций на выборке, где стандартная ошибка метрики около 0.01,
        # гарантированно дадут победителя, завышенного на шум. Поэтому смотрю
        # не на первую строку, а на то, вся ли верхушка таблицы лежит в одной области
        spread = table.head(10)["Recall@50"]
        print(f"разброс топ-10: {spread.max() - spread.min():.4f} при ошибке метрики около 0.01")
        top = table.iloc[0]
        best_fields = [
            Field("title", float(top["заголовок"]), 0.6),
            Field("params", float(top["параметры"]), 0.75),
            Field("description", float(top["описание"]), 0.75),
        ]

    print(f"\nЛучшие веса: {[(f.name, f.boost) for f in best_fields]}")
    index.reweight(best_fields)

    print("\nФильтр поиска дописан к тексту запроса")
    comparison = []
    for label, texts in (("только запрос", plain), ("запрос + фильтр", with_filter)):
        order, _ = index.search(texts, top_k=TOP_K)
        comparison.append({"запрос": label, **_score(local.relevant, order, item_ids, query_ids)})
    print(pd.DataFrame(comparison).round(4).to_string(index=False))

    print("\nГлубина кандидатов: где лежит потолок для переранжирования")
    order, _ = index.search(plain, top_k=CANDIDATE_DEPTH)
    predictions = _predictions(order, item_ids, query_ids, CANDIDATE_DEPTH)
    curve = recall_curve(local.relevant, predictions, ks=(1, 5, 10, 25, 50, 100, 200, 500, 1000))
    print(
        pd.DataFrame(
            [{"k": k, "Recall@k": round(value, 4)} for k, value in curve.items()]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    run()
