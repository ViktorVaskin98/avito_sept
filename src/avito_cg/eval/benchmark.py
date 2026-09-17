"""Локальный бенчмарк, собранный из train

Платформа даёт 7 попыток, подбирать по ним веса нельзя. Нужна локальная копия бенчмарка,
которая предсказывает платформенный скор хотя бы по знаку изменений.

Наивный сплит «отложили запросы, взяли случайные объявления в корпус» врёт в двух местах,
и оба видно по цифрам из EDA.

Первое. В настоящем корпусе только 9.6% объявлений встречаются в обучающих парах. Если набрать
корпус случайными объявлениями из train, знакомыми окажутся все 100%, и любой признак уровня
объявления (сколько раз его выбирали, его популярность) будет выглядеть сильнее, чем он есть.
На реальном корпусе такого признака просто нет у девяти объявлений из десяти. Лечу это третьей
частью сплита: часть запросов уходит в пул-донор, их пары не участвуют в обучении вообще,
а их объявления идут в корпус как незнакомые дистракторы.

Второе. Запросы бенчмарка систематически другие: 3.20 слова против 2.45 и 63.1% пустых фильтров
против 35.2%. Отложенные запросы поэтому набираю не случайно, а по совместному распределению
(пустой фильтр, число слов), снятому с настоящих запросов бенчмарка. Разметки там нет,
распределение признаков доступно легально, так что это обычное перевзвешивание по наблюдаемым
ковариатам, а не подглядывание в ответы.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_cg.config import RANDOM_SEED
from avito_cg.data.io import (
    ITEM_FEATURE_COLUMNS,
    QUERY_FEATURE_COLUMNS,
    QUERY_KEY_COLUMNS,
    query_key,
)

N_QUERIES = 2452
CORPUS_SIZE = 189_212
SEEN_SHARE = 0.096
WORD_BINS = (1, 2, 3, 4, 5)


def synthetic_query_id(key: str) -> str:
    """Стабильный идентификатор запроса той же формы, что на платформе

    В train своего query_id нет, а вся остальная машинерия (метрика, сборка ответа,
    валидатор формата) работает с идентификаторами. Беру хэш от ключа запроса:
    16 символов, воспроизводится между запусками
    """
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def _word_bin(text: str) -> int:
    """Число слов в запросе, склеенное в последнюю корзину"""
    return min(len(str(text).split()), WORD_BINS[-1] + 1)


def _stratum(empty_filter: bool, words: int) -> tuple[bool, int]:
    return empty_filter, words


def target_strata(benchmark_queries: pd.DataFrame) -> dict[tuple[bool, int], float]:
    """Совместное распределение (пустой фильтр, число слов) у настоящих запросов"""
    cells = Counter(
        _stratum(len(str(params)) == 0, _word_bin(text))
        for params, text in zip(
            benchmark_queries["search_infm_params_text"].astype(str),
            benchmark_queries["search_query"].astype(str),
            strict=True,
        )
    )
    total = sum(cells.values())
    return {cell: count / total for cell, count in cells.items()}


@dataclass(frozen=True, slots=True)
class LocalBenchmark:
    """Отложенные запросы, корпус и та часть train, которой разрешено пользоваться"""

    queries: pd.DataFrame
    relevant: dict[str, set[str]]
    corpus_item_ids: np.ndarray
    fit_rows: np.ndarray
    meta: dict[str, Any]

    @property
    def query_ids(self) -> list[str]:
        return self.queries["query_id"].astype(str).tolist()

    def corpus(self, train: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
        """Материализовать корпус по сохранённым item_id

        Признаки объявления беру из первой встреченной строки train: они там
        повторяются для одного и того же объявления
        """
        wanted = columns or [name for name in ITEM_FEATURE_COLUMNS if name in train.columns]
        # item_id уходит в индекс и возвращается обратно колонкой через reset_index,
        # поэтому в списке запрашиваемых колонок его быть не должно
        wanted = [name for name in wanted if name != "item_id"]
        unique = train.drop_duplicates(subset="item_id").set_index("item_id")
        return unique.loc[self.corpus_item_ids, wanted].reset_index()

    def fit_pairs(self, train: pd.DataFrame) -> pd.DataFrame:
        """Пары, которыми разрешено обучаться: без отложенных запросов и без донора"""
        return train.iloc[self.fit_rows]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.queries.to_parquet(directory / "queries.parquet", index=False)
        pd.DataFrame(
            [
                {"query_id": query_id, "item_id": item_id}
                for query_id, items in self.relevant.items()
                for item_id in sorted(items)
            ]
        ).to_parquet(directory / "relevant.parquet", index=False)
        pd.DataFrame({"item_id": self.corpus_item_ids}).to_parquet(
            directory / "corpus.parquet", index=False
        )
        np.save(directory / "fit_rows.npy", self.fit_rows)
        (directory / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> LocalBenchmark:
        relevant: defaultdict[str, set[str]] = defaultdict(set)
        for query_id, item_id in pd.read_parquet(directory / "relevant.parquet").itertuples(
            index=False
        ):
            relevant[str(query_id)].add(str(item_id))
        return cls(
            queries=pd.read_parquet(directory / "queries.parquet"),
            relevant=dict(relevant),
            corpus_item_ids=pd.read_parquet(directory / "corpus.parquet")["item_id"].to_numpy(),
            fit_rows=np.load(directory / "fit_rows.npy"),
            meta=json.loads((directory / "meta.json").read_text(encoding="utf-8")),
        )


def _sample_queries(
    pool: pd.DataFrame,
    strata: dict[tuple[bool, int], float],
    n_queries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Набрать отложенные запросы по заданному распределению страт

    Если в какой-то страте запросов меньше, чем просит распределение, беру сколько есть
    и добираю недостачу пропорционально остальным. Такое случается на длинных редких
    запросах, и об этом честнее сообщить в meta, чем молча промахнуться по распределению
    """
    cells = pool.groupby(["_empty_filter", "_words"]).indices
    chosen: list[np.ndarray] = []
    shortfall = 0
    for cell, share in sorted(strata.items()):
        wanted = round(share * n_queries)
        available = cells.get(cell, np.array([], dtype=int))
        take = min(wanted, len(available))
        shortfall += wanted - take
        if take:
            chosen.append(rng.choice(available, size=take, replace=False))

    picked = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if shortfall > 0 or len(picked) < n_queries:
        rest = np.setdiff1d(np.arange(len(pool)), picked)
        extra = min(n_queries - len(picked), len(rest))
        picked = np.concatenate([picked, rng.choice(rest, size=extra, replace=False)])
    return picked[:n_queries]


def _split_donor(
    keys: np.ndarray,
    item_ids: np.ndarray,
    available_rows: np.ndarray,
    needed_unseen: int,
    rng: np.random.Generator,
) -> tuple[set[str], np.ndarray]:
    """Отделить пул-донор так, чтобы набралось нужное число незнакомых объявлений

    Иду по перемешанным ключам запросов и переношу их из обучающей части в донор,
    считая для каждого объявления, сколько у него осталось обучающих пар. Как только
    счётчик обнуляется, объявление становится незнакомым. Один проход, без подбора долей
    """
    remaining: Counter[str] = Counter(item_ids[available_rows])
    rows_by_key: defaultdict[str, list[int]] = defaultdict(list)
    for row in available_rows:
        rows_by_key[keys[row]].append(int(row))

    order = list(rows_by_key)
    rng.shuffle(order)

    donor: set[str] = set()
    unseen = 0
    for key in order:
        if unseen >= needed_unseen:
            break
        donor.add(key)
        for row in rows_by_key[key]:
            item = item_ids[row]
            remaining[item] -= 1
            if remaining[item] == 0:
                unseen += 1

    fit_rows = np.array(
        [row for key, rows in rows_by_key.items() if key not in donor for row in rows], dtype=int
    )
    fit_rows.sort()
    return donor, fit_rows


def build(
    train: pd.DataFrame,
    benchmark_queries: pd.DataFrame,
    *,
    n_queries: int = N_QUERIES,
    corpus_size: int = CORPUS_SIZE,
    seen_share: float = SEEN_SHARE,
    match_strata: bool = True,
    seed: int = RANDOM_SEED,
) -> LocalBenchmark:
    """Собрать локальный бенчмарк

    seen_share задаёт долю корпуса, знакомую обучающей части. Значение по умолчанию
    снято с настоящих данных. Поставить сюда 1.0 означает вернуться к наивному сплиту,
    это бывает полезно для быстрых прогонов, но абсолютные числа тогда завышены
    """
    rng = np.random.default_rng(seed)
    keys = query_key(train).to_numpy()
    item_ids = train["item_id"].astype(str).to_numpy()

    pool = train.drop_duplicates(subset=list(QUERY_KEY_COLUMNS))[list(QUERY_FEATURE_COLUMNS)].copy()
    pool["_key"] = query_key(pool).to_numpy()
    pool["_empty_filter"] = pool["search_infm_params_text"].astype(str).str.len() == 0
    pool["_words"] = pool["search_query"].astype(str).map(_word_bin)
    pool = pool.reset_index(drop=True)

    if match_strata:
        positions = _sample_queries(pool, target_strata(benchmark_queries), n_queries, rng)
    else:
        positions = rng.choice(len(pool), size=n_queries, replace=False)
    validation_keys = set(pool.loc[positions, "_key"])

    is_validation = np.isin(keys, list(validation_keys))
    validation_rows = np.flatnonzero(is_validation)
    available_rows = np.flatnonzero(~is_validation)

    relevant: defaultdict[str, set[str]] = defaultdict(set)
    for row in validation_rows:
        relevant[synthetic_query_id(keys[row])].add(item_ids[row])
    positives = {item for items in relevant.values() for item in items}

    seen_target = round(seen_share * corpus_size)
    needed_unseen = corpus_size - seen_target
    donor, fit_rows = _split_donor(keys, item_ids, available_rows, needed_unseen, rng)

    # позитивы уже лежат в корпусе, поэтому из обоих пулов их надо вычесть,
    # иначе одно и то же объявление попадёт в корпус дважды
    fit_items = set(item_ids[fit_rows])
    unseen_pool = np.array(
        sorted(set(item_ids[available_rows]) - fit_items - positives), dtype=object
    )
    seen_pool = np.array(sorted(fit_items - positives), dtype=object)

    seen_from_positives = len(positives & fit_items)
    take_seen = max(0, min(seen_target - seen_from_positives, len(seen_pool)))
    take_unseen = corpus_size - len(positives) - take_seen
    if take_unseen > len(unseen_pool):
        raise ValueError(
            f"не хватает незнакомых объявлений: нужно {take_unseen}, есть {len(unseen_pool)}. "
            f"подними seen_share или уменьши corpus_size"
        )

    corpus = np.concatenate(
        [
            np.array(sorted(positives), dtype=object),
            rng.choice(seen_pool, size=take_seen, replace=False)
            if take_seen
            else np.array([], dtype=object),
            rng.choice(unseen_pool, size=take_unseen, replace=False),
        ]
    )
    rng.shuffle(corpus)

    queries = pool.loc[positions, [*QUERY_FEATURE_COLUMNS, "_key"]].copy()
    queries["query_id"] = queries["_key"].map(synthetic_query_id)
    queries = queries[["query_id", *QUERY_FEATURE_COLUMNS]].reset_index(drop=True)

    meta = {
        "запросов": len(queries),
        "корпус": len(corpus),
        "релевантных всего": int(sum(len(items) for items in relevant.values())),
        "заданная доля знакомых": seen_share,
        "доля корпуса, знакомая обучению": float(len(set(corpus) & fit_items) / len(corpus)),
        "релевантных, знакомых обучению": float(seen_from_positives / max(1, len(positives))),
        "обучающих пар": len(fit_rows),
        "пар в доноре": int(len(available_rows) - len(fit_rows)),
        "запросов в доноре": len(donor),
        "перевзвешивание страт": match_strata,
        "seed": seed,
    }
    return LocalBenchmark(queries, dict(relevant), corpus, fit_rows, meta)


def describe(
    local: LocalBenchmark,
    train: pd.DataFrame,
    benchmark_queries: pd.DataFrame,
    benchmark_items: pd.DataFrame,
) -> pd.DataFrame:
    """Сверить локальный бенчмарк с настоящим по всем величинам, которые я умею мерить

    Это и есть проверка валидации: если какая-то строка сильно разъехалась,
    локальная метрика будет мерить не то, что платформенная
    """
    corpus = local.corpus(train, columns=["item_location_id", "item_microcat_id"])
    sizes = np.array([len(items) for items in local.relevant.values()])

    def words(frame: pd.DataFrame) -> float:
        return float(frame["search_query"].astype(str).str.split().str.len().mean())

    def empty(frame: pd.DataFrame) -> float:
        return float((frame["search_infm_params_text"].astype(str).str.len() == 0).mean())

    rows = [
        ("запросов", len(local.queries), str(len(benchmark_queries))),
        ("объявлений в корпусе", len(corpus), str(len(benchmark_items))),
        (
            "слов в запросе, среднее",
            round(words(local.queries), 3),
            str(round(words(benchmark_queries), 3)),
        ),
        (
            "доля пустых фильтров",
            round(empty(local.queries), 3),
            str(round(empty(benchmark_queries), 3)),
        ),
        (
            "доля корпуса, знакомая обучению",
            round(local.meta["доля корпуса, знакомая обучению"], 3),
            "0.096",
        ),
        (
            "локаций в корпусе",
            int(corpus["item_location_id"].nunique()),
            str(int(benchmark_items["item_location_id"].nunique())),
        ),
        (
            "микрокатегорий в корпусе",
            int(corpus["item_microcat_id"].nunique()),
            f"{int(benchmark_items['item_microcat_id'].nunique())}, но в train их всего 212",
        ),
        # разметки бенчмарка у нас нет, так что сравнивать тут не с чем:
        # 1.40 это маргинальное среднее по train, а не значение на бенчмарке
        (
            "релевантных на запрос, среднее",
            round(float(sizes.mean()), 3),
            "неизвестно, в train 1.40",
        ),
        (
            "релевантных на запрос, ровно один",
            round(float((sizes == 1).mean()), 3),
            "неизвестно, в train 0.836",
        ),
    ]
    return pd.DataFrame(
        [(name, str(local_value), real_value) for name, local_value, real_value in rows],
        columns=["величина", "локальный", "настоящий"],
    )


def leakage_checks(local: LocalBenchmark, train: pd.DataFrame) -> list[str]:
    """Претензии к сплиту, пустой список означает что утечек не нашлось"""
    problems: list[str] = []

    fit_query_ids = {synthetic_query_id(key) for key in query_key(train.iloc[local.fit_rows])}
    if overlap := fit_query_ids & set(local.relevant):
        problems.append(f"отложенные запросы попали в обучающую часть: {len(overlap)}")

    corpus = set(local.corpus_item_ids)
    missing = {item for items in local.relevant.values() for item in items} - corpus
    if missing:
        problems.append(f"релевантных объявлений нет в корпусе: {len(missing)}")

    if len(corpus) != len(local.corpus_item_ids):
        problems.append(f"в корпусе дубли: {len(local.corpus_item_ids) - len(corpus)}")

    empty_queries = [query for query, items in local.relevant.items() if not items]
    if empty_queries:
        problems.append(f"запросов без релевантных: {len(empty_queries)}")

    if set(local.queries["query_id"]) != set(local.relevant):
        problems.append("список запросов и разметка разошлись")

    fit_items = set(train.iloc[local.fit_rows]["item_id"].astype(str))
    actual_seen = len(corpus & fit_items) / len(corpus)
    expected_seen = local.meta.get("заданная доля знакомых", SEEN_SHARE)
    if abs(actual_seen - expected_seen) > 0.02:
        problems.append(
            f"доля знакомых объявлений в корпусе {actual_seen:.3f}, а просили {expected_seen:.3f}"
        )

    return problems
