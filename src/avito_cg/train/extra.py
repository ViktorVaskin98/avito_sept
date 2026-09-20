"""Дополнительные обучающие запросы для переранжировщика

Реранкер обучался на тех же 2 452 отложенных запросах, на которых и мерился: 3 093 позитива
на восемнадцать признаков, и по приросту видно, что он недоучен - из доступных по потолку
0.041 он забрал 0.008.

Взять запросы есть откуда. При сборке сплита часть обучающих запросов уходит в пул-донор,
их пары не участвуют ни в калибровке сигналов, ни в обучении энкодера, а их объявления
ложатся в корпус как незнакомые дистракторы. Значит для донорского запроса известны
и признаки, и метки, и при этом ни один сигнал его не видел. Таких запросов 161 456.

Ровно так брать их нельзя. У позитивов донорских запросов доля знакомых обучению 4.2%,
у отложенных 28.7%: донор по построению состоит из объявлений, которые в fit не попали.
Если не поправить, модель выучит, что популярное объявление скорее лишнее, то есть ровно
обратное тому, что происходит на отложенных запросах. Поэтому набираю донорские запросы
так, чтобы доля знакомых среди их позитивов совпала с отложенными - тем же приёмом
согласования маргиналов, что и сам сплит.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from avito_cg.config import RANDOM_SEED
from avito_cg.data.io import QUERY_FEATURE_COLUMNS, query_key
from avito_cg.eval.benchmark import LocalBenchmark, synthetic_query_id


def build_donor_set(
    train: pd.DataFrame,
    local: LocalBenchmark,
    *,
    n_queries: int = 12_000,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, float]]:
    """Донорские запросы с позитивами в корпусе и согласованной долей знакомых

    Возвращаю запросы в том же виде, что local.queries, разметку в том же виде,
    что local.relevant, и отчёт о том, насколько согласование удалось
    """
    rng = np.random.default_rng(seed)
    keys = query_key(train).to_numpy()
    query_ids = np.array([synthetic_query_id(key) for key in keys])
    items = train["item_id"].astype(str).to_numpy()

    validation = set(local.queries["query_id"].astype(str))
    in_fit = np.zeros(len(train), dtype=bool)
    in_fit[local.fit_rows] = True
    donor = ~in_fit & ~np.isin(query_ids, list(validation))

    corpus = set(local.corpus_item_ids.tolist())
    fit_items = set(items[local.fit_rows])

    rows = np.flatnonzero(donor)
    rows = rows[[items[row] in corpus for row in rows]]

    positives: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        positives[query_ids[row]].add(items[row])

    target = _seen_share(local.relevant, fit_items)
    with_seen: list[str] = []
    without_seen: list[str] = []
    for query, chosen in positives.items():
        bucket = with_seen if chosen & fit_items else without_seen
        bucket.append(query)

    rng.shuffle(with_seen)
    rng.shuffle(without_seen)
    chosen_queries = _match_share(
        with_seen, without_seen, positives, fit_items, target, n_queries, rng
    )

    selected = {query: positives[query] for query in chosen_queries}
    first_row: dict[str, int] = {}
    for row in rows:
        first_row.setdefault(query_ids[row], int(row))
    order = [first_row[query] for query in chosen_queries]

    queries = train.iloc[order][list(QUERY_FEATURE_COLUMNS)].copy()
    queries.insert(0, "query_id", chosen_queries)
    queries = queries.reset_index(drop=True)

    report = {
        "запросов": len(queries),
        "позитивов": sum(len(chosen) for chosen in selected.values()),
        "доля знакомых у позитивов": _seen_share(selected, fit_items),
        "цель по отложенным": target,
        "донорских запросов всего": len(positives),
    }
    return queries, selected, report


def _seen_share(relevant: dict[str, set[str]], fit_items: set[str]) -> float:
    total = [item in fit_items for chosen in relevant.values() for item in chosen]
    return float(np.mean(total)) if total else 0.0


def _match_share(
    with_seen: list[str],
    without_seen: list[str],
    positives: dict[str, set[str]],
    fit_items: set[str],
    target: float,
    n_queries: int,
    rng: np.random.Generator,
) -> list[str]:
    """Добрать запросы без знакомых позитивов до нужной доли, а не сколько попало

    Запросов со знакомым позитивом мало, поэтому они задают масштаб: беру их все
    (или столько, сколько влезает в бюджет), а остальными развожу до целевой доли
    """
    head = with_seen[: max(1, n_queries // 3)]
    seen_count = sum(item in fit_items for query in head for item in positives[query])
    unseen_count = sum(item not in fit_items for query in head for item in positives[query])

    needed_unseen = seen_count * (1 - target) / target - unseen_count
    chosen = list(head)
    for query in without_seen:
        if needed_unseen <= 0 or len(chosen) >= n_queries:
            break
        chosen.append(query)
        needed_unseen -= len(positives[query])
    return chosen
