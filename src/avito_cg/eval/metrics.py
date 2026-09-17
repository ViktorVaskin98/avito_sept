"""Целевая метрика и диагностика вокруг неё

Recall@50 усредняется по запросам, а не по парам: запрос с двумя релевантными объявлениями
весит столько же, сколько запрос с одним. В данных у 83.6% запросов ровно один релевантный
item, так что метрика ведёт себя почти как hit-rate, и разброс на выборке в 2.5 тысячи
запросов получается порядка 0.01. Сравнивать модели без доверительного интервала
бессмысленно, поэтому тут же лежит bootstrap_ci
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np

from avito_cg.config import RANDOM_SEED, TOP_K


def recall_per_query(
    relevant: Mapping[str, Collection[str]],
    predicted: Mapping[str, Sequence[str]],
    k: int = TOP_K,
) -> np.ndarray:
    """Вклад каждого запроса: |топ-k ∩ релевантные| / |релевантные|

    Запросы беру из relevant: если модель для запроса не вернула ничего, это честный ноль,
    а не пропущенная строка
    """
    scores = np.empty(len(relevant), dtype=np.float64)
    for index, (query_id, truth) in enumerate(relevant.items()):
        truth_set = set(truth)
        if not truth_set:
            scores[index] = np.nan
            continue
        top = predicted.get(query_id, ())[:k]
        scores[index] = len(truth_set.intersection(top)) / len(truth_set)
    return scores


def recall_at_k(
    relevant: Mapping[str, Collection[str]],
    predicted: Mapping[str, Sequence[str]],
    k: int = TOP_K,
) -> float:
    return float(np.nanmean(recall_per_query(relevant, predicted, k)))


def recall_curve(
    relevant: Mapping[str, Collection[str]],
    predicted: Mapping[str, Sequence[str]],
    ks: Sequence[int] = (1, 5, 10, 20, 50, 100, 200, 500, 1000),
) -> dict[int, float]:
    """Recall на разных отсечках

    Нужен, чтобы различать две совершенно разные проблемы: кандидат вообще не нашёлся
    (плохо на всех k) или нашёлся, но не дожил до топ-50 (разрыв между k=1000 и k=50).
    Первое лечится генератором кандидатов, второе переранжированием
    """
    return {k: recall_at_k(relevant, predicted, k) for k in ks}


def bootstrap_ci(
    scores: np.ndarray,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Доверительный интервал среднего по запросам, бутстрэпом"""
    clean = scores[~np.isnan(scores)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_resamples, clean.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def candidate_coverage(
    relevant: Mapping[str, Collection[str]],
    candidates: Mapping[str, Collection[str]],
) -> float:
    """Потолок качества для данного набора кандидатов

    Сколько релевантных объявлений вообще попало в кандидаты до отсечки топ-50.
    Выше этого числа никакое переранжирование не поднимет
    """
    return recall_at_k(relevant, {q: list(c) for q, c in candidates.items()}, k=10**9)
