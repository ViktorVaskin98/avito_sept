"""Сбор кандидатов и его согласованность с выдачей

collect и retrieve считают одно и то же разными кусками кода: retrieve складывает
сигналы в один скор и отдаёт топ, collect дополнительно возвращает каждую составляющую
для переранжировщика. Разойтись они могут молча, а отвечает на платформу именно collect,
поэтому равенство проверяется тестом, а не чтением.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import BM25FIndex, Field
from avito_cg.retrieval.candidates import FINAL, collect
from avito_cg.retrieval.fusion import FusionConfig, retrieve
from avito_cg.retrieval.signals import GeoSignal

CENTRE = (55.75, 37.62)
TEXTS = ["маникюр", "педикюр", "маникюр педикюр"]


def words(text):
    return text.split()


@pytest.fixture
def corpus():
    return pd.DataFrame(
        {
            "item_latitude": [55.75, 55.80, 59.94, 43.12, 55.76],
            "item_longitude": [37.62, 37.70, 30.31, 131.89, 37.61],
        }
    )


@pytest.fixture
def geo(corpus):
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame(
        {
            "search_location_id": [1] * 2000,
            "item_latitude": CENTRE[0] + rng.normal(0, 0.02, 2000),
            "item_longitude": CENTRE[1] + rng.normal(0, 0.02, 2000),
        }
    )
    background = pd.DataFrame(
        {
            "item_latitude": rng.uniform(43, 60, 4000),
            "item_longitude": rng.uniform(30, 132, 4000),
        }
    )
    return replace(
        GeoIndex.fit(pairs, background),
        item_latitude=corpus["item_latitude"].to_numpy(),
        item_longitude=corpus["item_longitude"].to_numpy(),
    )


@pytest.fixture
def index():
    documents = {
        "title": ["маникюр", "маникюр педикюр", "педикюр", "маникюр", "педикюр маникюр"],
        "description": ["", "", "", "", ""],
    }
    fields = [Field("title", 1.0, 0.0), Field("description", 1.0, 0.0)]
    return BM25FIndex(fields, tokenizer=words).fit(documents)


@pytest.fixture
def signals(geo):
    queries = pd.DataFrame({"search_location_id": [1, 1, 1]})
    return [(GeoSignal.build(geo, queries), 0.5)]


@pytest.mark.parametrize("mode", ["raw", "normalized", "rrf"])
def test_collect_order_matches_retrieve(index, signals, mode):
    """Пул и выдача обязаны строиться по одной формуле во всех режимах слияния"""
    config = FusionConfig(mode=mode)
    expected = retrieve(index, signals, TEXTS, top_k=3, config=config)
    got = collect(index, signals, TEXTS, depth=5, config=config).order(3)
    assert got.tolist() == expected.tolist()


def test_collect_matches_retrieve_with_extra_candidates(index, signals):
    """Кандидаты от плотного поиска должны попадать в оба пути одинаково"""
    extra = np.array([[3, 4], [4, -1], [-1, -1]], dtype=np.int64)
    expected = retrieve(index, signals, TEXTS, top_k=4, extra_candidates=extra)
    got = collect(index, signals, TEXTS, depth=8, extra_candidates=extra).order(4)
    assert got.tolist() == expected.tolist()


def test_collect_keeps_raw_signal_scores(index, signals):
    """В таблицу признаков идёт сырой скор сигнала, а не его вклад в сумму"""
    collected = collect(index, signals, TEXTS, depth=5)
    signal, _ = signals[0]
    for query, items in enumerate(collected.items):
        if items.size:
            expected = signal.score(query, items)
            assert np.allclose(collected.scores[signal.name][query], expected)


def test_collect_reports_final_score(index, signals):
    """Итоговый скор нужен переранжировщику рангом, так что он должен быть убывающим"""
    collected = collect(index, signals, TEXTS, depth=5)
    for values in collected.scores[FINAL]:
        if values.size:
            assert np.all(np.diff(values) <= 1e-12)


def test_query_without_candidates_stays_empty(index, signals):
    collected = collect(index, signals, ["совершенно другое", "маникюр", "педикюр"], depth=5)
    assert collected.items[0].size == 0
    assert collected.order(3)[0].tolist() == [-1, -1, -1]
