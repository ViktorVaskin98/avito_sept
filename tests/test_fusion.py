from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import BM25FIndex, Field
from avito_cg.retrieval.fusion import FusionConfig, retrieve
from avito_cg.retrieval.signals import GeoSignal

CENTRE = (55.75, 37.62)


def words(text):
    return text.split()


@pytest.fixture
def corpus():
    """Четыре объявления с одинаковым текстом, но разной удалённостью от Москвы"""
    return pd.DataFrame(
        {
            "item_latitude": [55.75, 55.80, 59.94, 43.12],
            "item_longitude": [37.62, 37.70, 30.31, 131.89],
        }
    )


@pytest.fixture
def geo(corpus):
    """Калибровка учится на широком фоне, координаты подменяются на корпус теста"""
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
    fitted = GeoIndex.fit(pairs, background)
    return replace(
        fitted,
        item_latitude=corpus["item_latitude"].to_numpy(),
        item_longitude=corpus["item_longitude"].to_numpy(),
    )


@pytest.fixture
def index():
    documents = {
        "title": ["маникюр", "маникюр", "маникюр", "маникюр"],
        "description": ["", "", "", ""],
    }
    fields = [Field("title", 1.0, 0.0), Field("description", 1.0, 0.0)]
    return BM25FIndex(fields, tokenizer=words).fit(documents)


def signal(geo, location=1):
    return GeoSignal.build(geo, pd.DataFrame({"search_location_id": [location]}))


def test_geo_breaks_ties_between_identical_texts(index, geo):
    """Ровно тот случай, ради которого всё затевалось: текст одинаковый, решает расстояние"""
    order = retrieve(index, [(signal(geo), 1.0)], ["маникюр"], top_k=4)
    assert order[0, 0] == 0
    assert order[0, 3] == 3


def test_zero_weight_leaves_lexical_order_alone(index, geo):
    order = retrieve(index, [(signal(geo), 0.0)], ["маникюр"], top_k=4)
    lexical, _ = index.search(["маникюр"], top_k=4)
    assert order.tolist() == lexical.tolist()


def test_unknown_location_falls_back_to_lexical(index, geo):
    """Гео про такой запрос ничего не знает и не должно ничего сломать"""
    order = retrieve(index, [(signal(geo, 999), 1.0)], ["маникюр"], top_k=4)
    assert sorted(order[0].tolist()) == [0, 1, 2, 3]


def test_query_without_lexical_match_returns_nothing_by_default(index, geo):
    order = retrieve(index, [(signal(geo), 1.0)], ["совершенно другое"], top_k=3)
    assert (order == -1).all()


def test_padding_fills_empty_slots_by_distance(index, geo):
    """Пустое место в ответе это чистая потеря: метрика не штрафует за лишних кандидатов"""
    coordinates = (np.array([CENTRE[0]]), np.array([CENTRE[1]]))
    order = retrieve(
        index,
        [(signal(geo), 1.0)],
        ["совершенно другое"],
        top_k=3,
        padding=geo,
        query_coordinates=coordinates,
    )
    assert (order >= 0).all()
    assert order[0, 0] == 0


@pytest.mark.parametrize("mode", ["raw", "normalized", "rrf"])
def test_all_modes_return_a_full_ranking(index, geo, mode):
    order = retrieve(
        index, [(signal(geo), 1.0)], ["маникюр"], top_k=4, config=FusionConfig(mode=mode)
    )
    assert sorted(order[0].tolist()) == [0, 1, 2, 3]


def test_several_signals_are_summed(index, geo):
    """Второй сигнал должен переставлять порядок, а не молча игнорироваться"""

    class Flip:
        name = "перевёртыш"

        def score(self, query, candidates):
            return -np.arange(candidates.size, dtype=float)

    geo_only = retrieve(index, [(signal(geo), 1.0)], ["маникюр"], top_k=4)
    with_flip = retrieve(index, [(signal(geo), 1.0), (Flip(), 100.0)], ["маникюр"], top_k=4)
    assert geo_only.tolist() != with_flip.tolist()
