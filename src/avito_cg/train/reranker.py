"""Переранжирование кандидатов градиентным бустингом

Зачем он нужен, стало понятно из разбора промахов, а не из общих соображений. Два оставшихся
класса потерь описываются одной фразой: вес сигнала должен зависеть от силы другого сигнала.

Далёкие, но текстуально точные объявления теряются, потому что гео давит их даже при идеальном
совпадении текста: за 25 км доля попаданий падает с 0.92 до 0.46. Простое ослабление штрафа
проверено и не работает, оно ухудшает метрику монотонно, потому что вместе с далёкими
правильными в ответ лезут далёкие неправильные. Нужна условная поправка «ослабь гео, если текст
совпал отлично», а взвешенная сумма такого не выражает ни при каких весах.

**На чём обучать.** Обучающих запросов взять неоткуда: их позитивы в локальном корпусе почти
не лежат, это и есть конструкция с пулом-донором. Поэтому обучаю на самих отложенных запросах
с кросс-валидацией по фолдам: модель никогда не видит фолд, на котором предсказывает, и оценка
получается честной. Для ответа по настоящему корпусу модель обучается на всех 2 452 запросах
и применяется к бенчмарку, разметки которого мы не касаемся вообще.

Цена такой схемы - маленькая обучающая выборка: 3 093 позитива на десяток признаков.
Для бустинга это немного, и результат вполне может оказаться в пределах шума.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from avito_cg.analysis import haversine
from avito_cg.config import RANDOM_SEED
from avito_cg.data.text import normalize, tokenize

FEATURES = [
    "лексика",
    "лексика_норм",
    "лексика_ранг",
    "гео",
    "расстояние",
    "фасеты",
    "микрокатегория",
    "плотный",
    "плотный_ранг",
    "итоговый_ранг",
    "слов_в_запросе",
    "пустой_фильтр",
    "покрытие_заголовка",
    "близнецов",
    "цена",
    "рейтинг",
    "отзывов",
    "кандидатов",
]


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Кандидаты одного запроса вместе с признаками и метками"""

    query: np.ndarray
    item: np.ndarray
    features: np.ndarray
    label: np.ndarray


def build_features(
    candidates: Sequence[np.ndarray],
    lexical: Sequence[np.ndarray],
    signal_scores: dict[str, Sequence[np.ndarray]],
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    query_latitude: np.ndarray,
    query_longitude: np.ndarray,
    relevant: dict[str, set[str]],
    query_ids: Sequence[str],
) -> CandidateSet:
    """Собрать таблицу признаков по кандидатам всех запросов

    Ранги добавляю рядом со скорами намеренно: дерево на скоре учит порог в абсолютных
    единицах, который между запросами не переносится, а ранг переносится
    """
    item_ids = corpus["item_id"].astype(str).to_numpy()
    titles = corpus["item_title_raw"].fillna("").astype(str).to_numpy()
    twins = pd.Series([normalize(text) for text in titles]).value_counts()
    twin_count = np.array([twins.get(normalize(text), 1) for text in titles], dtype=float)
    latitude = corpus["item_latitude"].to_numpy(dtype=float)
    longitude = corpus["item_longitude"].to_numpy(dtype=float)
    price = corpus["item_price"].to_numpy(dtype=float)
    rating = corpus["item_rating"].to_numpy(dtype=float)
    reviews = corpus["item_rating_reviews_count"].to_numpy(dtype=float)

    query_text = queries["search_query"].fillna("").astype(str).to_numpy()
    empty_filter = (
        queries["search_infm_params_text"].fillna("").astype(str).str.len() == 0
    ).to_numpy()

    rows, labels, query_index, item_index = [], [], [], []
    for position, query_id in enumerate(query_ids):
        items = candidates[position]
        if items.size == 0:
            continue
        truth = relevant.get(query_id, set())
        tokens = set(tokenize(query_text[position]))

        lexical_scores = lexical[position]
        geo_scores = signal_scores["гео"][position]
        facet_scores = signal_scores["фасеты"][position]
        microcat_scores = signal_scores["микрокатегория"][position]
        dense_scores = signal_scores["плотный"][position]
        final = signal_scores["итог"][position]

        distance = haversine(
            np.full(items.size, query_latitude[position]),
            np.full(items.size, query_longitude[position]),
            latitude[items],
            longitude[items],
        )
        coverage = np.array(
            [
                len(tokens & set(tokenize(titles[item]))) / len(tokens) if tokens else 0.0
                for item in items
            ]
        )
        block = np.column_stack(
            [
                lexical_scores,
                lexical_scores / max(lexical_scores.max(), 1e-9),
                _ranks(lexical_scores),
                geo_scores,
                np.nan_to_num(distance, nan=-1.0, posinf=-1.0),
                facet_scores,
                microcat_scores,
                dense_scores,
                _ranks(dense_scores),
                _ranks(final),
                np.full(items.size, len(query_text[position].split())),
                np.full(items.size, float(empty_filter[position])),
                coverage,
                twin_count[items],
                price[items],
                rating[items],
                reviews[items],
                np.full(items.size, float(items.size)),
            ]
        )
        rows.append(block)
        labels.append(np.array([item_ids[item] in truth for item in items], dtype=int))
        query_index.append(np.full(items.size, position))
        item_index.append(items)

    return CandidateSet(
        query=np.concatenate(query_index),
        item=np.concatenate(item_index),
        features=np.vstack(rows),
        label=np.concatenate(labels),
    )


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.empty(scores.size, dtype=np.float64)
    order[np.argsort(-scores)] = np.arange(scores.size)
    return order


def cross_validated_scores(
    data: CandidateSet,
    n_queries: int,
    *,
    folds: int = 5,
    seed: int = RANDOM_SEED,
    iterations: int = 400,
    depth: int = 6,
) -> np.ndarray:
    """Скор реранкера для каждого кандидата, полученный моделью, его не видевшей

    Разбиение по запросам, а не по строкам: иначе кандидаты одного запроса попадут
    и в обучение, и в контроль, и оценка будет завышена
    """
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, folds, size=n_queries)
    scores = np.zeros(len(data.label))

    for fold in range(folds):
        test_queries = np.flatnonzero(assignment == fold)
        test_mask = np.isin(data.query, test_queries)
        if test_mask.sum() == 0 or (~test_mask).sum() == 0:
            continue
        model = _fit(data, ~test_mask, iterations=iterations, depth=depth, seed=seed)
        scores[test_mask] = model.predict_proba(data.features[test_mask])[:, 1]
    return scores


def _fit(
    data: CandidateSet, mask: np.ndarray, *, iterations: int, depth: int, seed: int
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="Logloss",
        # позитивов около половины процента от строк, без веса модель просто
        # научится всем отвечать «нет»
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(data.features[mask], data.label[mask], feature_names=FEATURES))
    return model


def fit_full(
    data: CandidateSet, *, iterations: int = 400, depth: int = 6, seed: int = RANDOM_SEED
) -> CatBoostClassifier:
    """Обучить на всех данных, для применения к настоящему корпусу"""
    return _fit(
        data, np.ones(len(data.label), dtype=bool), iterations=iterations, depth=depth, seed=seed
    )


def importance(model: CatBoostClassifier) -> pd.DataFrame:
    values = model.get_feature_importance()
    return (
        pd.DataFrame({"признак": FEATURES, "важность": values.round(2)})
        .sort_values("важность", ascending=False)
        .reset_index(drop=True)
    )
