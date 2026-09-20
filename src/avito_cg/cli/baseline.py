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
from avito_cg.index.fields import item_fields, load_parser, query_texts
from avito_cg.index.lexical import DEFAULT_FIELDS, K1, BM25FIndex, Field

CORPUS_COLUMNS = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
SPLIT_DIR = "local_benchmark"
CANDIDATE_DEPTH = 1000

TITLE_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 20.0, 40.0)
PARAMS_GRID = (0.0, 0.1, 0.25, 0.5)
DESCRIPTION_GRID = (0.5, 1.0, 2.0)
# от этого значения и выше метрика по весу заголовка выходит на полку, см. docs/EXPERIMENTS.md
TITLE_PLATEAU = 8.0
K1_GRID = (0.6, 0.9, 1.2, 1.6, 2.2)


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
    """Замерить BM25F на локальном бенчмарке: вклад полей, веса, насыщение, глубина

    Первая таблица - проверка не столько модели, сколько сплита: если добавление
    описания к заголовку не поднимает метрику, сломан не BM25, а разметка или метрика.
    Последняя - глубина кандидатов, из неё видно, сколько теряется на упорядочивании,
    а сколько на поиске, и это определило всю дальнейшую конструкцию
    """
    PATHS.ensure()
    started = time.time()

    train = load_train(columns=CORPUS_COLUMNS)
    local = LocalBenchmark.load(PATHS.interim / SPLIT_DIR)
    corpus = local.corpus(train, columns=CORPUS_COLUMNS)
    del train
    print(f"корпус {len(corpus)} объявлений загружен за {time.time() - started:.0f} c", flush=True)

    parser = load_parser()

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

    best_fields = list(DEFAULT_FIELDS)
    if grid:

        def score(fields: list[Field], k1: float | None = None) -> float:
            index.reweight(fields, k1=k1)
            order, _ = index.search(plain, top_k=TOP_K)
            return recall_at_k(local.relevant, _predictions(order, item_ids, query_ids, TOP_K))

        print("\nПодбор весов полей")
        # параметры с весом 1 метрику роняют (0.2773 -> 0.2613 на первом прогоне):
        # это 140 токенов родовых фасетных значений, общих для тысяч объявлений,
        # поэтому сетка идёт от нуля, а не вокруг единицы. Верх по заголовку доведён
        # до 40, чтобы в сетку попало и выбранное решением значение 20, и то, что
        # за ним: иначе таблица упирается в край и по ней не видно, есть ли там полка
        results = []
        for title_boost, params_boost, description_boost in product(
            TITLE_GRID, PARAMS_GRID, DESCRIPTION_GRID
        ):
            fields = [
                Field("title", title_boost, 0.6),
                Field("params", params_boost, 0.75),
                Field("description", description_boost, 0.75),
            ]
            results.append(
                {
                    "заголовок": title_boost,
                    "параметры": params_boost,
                    "описание": description_boost,
                    "Recall@50": score(fields),
                }
            )
        table = pd.DataFrame(results).sort_values("Recall@50", ascending=False)
        print(table.head(10).round(4).to_string(index=False))

        # argmax по такой сетке на выборке со стандартной ошибкой метрики около 0.01
        # гарантированно завышен на шум, поэтому победитель отсюда не берётся.
        # Смотрю на другое: лежит ли верхушка таблицы в одной области и попадает ли
        # в неё то, что зашито в DEFAULT_FIELDS
        spread = table.head(10)["Recall@50"]
        print(
            f"разброс топ-10: {spread.max() - spread.min():.4f} при ошибке метрики около 0.01, "
            f"то есть это одна точка, а не рейтинг"
        )
        chosen = {field.name: field.boost for field in DEFAULT_FIELDS}
        same_area = table.head(10)[
            (table.head(10)["заголовок"] >= TITLE_PLATEAU)
            & (table.head(10)["параметры"] == chosen["params"])
        ]
        print(
            f"решение берёт {chosen}, в первой десятке сетки "
            f"{len(same_area)} конфигураций из той же области"
        )

        print("\nНасыщение k1 при весах решения")
        # k1 задаёт, насколько быстро растущая частота терма перестаёт добавлять скор.
        # При весе заголовка 20 именно k1 определяет форму насыщения, и до этого
        # он не подбирался вовсе, стоял по умолчанию
        k1_table = [{"k1": value, "Recall@50": score(best_fields, value)} for value in K1_GRID]
        print(pd.DataFrame(k1_table).round(4).to_string(index=False))

    print(f"\nВеса решения: {[(f.name, f.boost) for f in best_fields]}, k1 = {K1}")
    index.reweight(best_fields, k1=K1)

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
