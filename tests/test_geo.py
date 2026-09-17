import numpy as np
import pandas as pd
import pytest

from avito_cg.index.geo import BIN_EDGES, GeoIndex

CENTRES = {100: (55.75, 37.62), 200: (59.94, 30.31)}


@pytest.fixture
def pairs():
    """Выбранные объявления кучкуются вокруг центра своей поисковой локации"""
    rng = np.random.default_rng(0)
    rows = [
        {
            "search_location_id": location,
            "item_latitude": lat + rng.normal(0, 0.03),
            "item_longitude": lon + rng.normal(0, 0.03),
        }
        for location, (lat, lon) in CENTRES.items()
        for _ in range(3000)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def corpus():
    """Корпус размазан по всей стране, иначе фон совпадёт с позитивами
    и отношение правдоподобий окажется плоским
    """
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "item_latitude": rng.uniform(45, 65, size=6000),
            "item_longitude": rng.uniform(30, 100, size=6000),
        }
    )


def test_centroid_is_the_median_of_chosen_items(pairs, corpus):
    geo = GeoIndex.fit(pairs, corpus)
    assert geo.latitude[100] == pytest.approx(55.75, abs=0.01)
    assert geo.longitude[200] == pytest.approx(30.31, abs=0.01)


def test_close_is_worth_more_than_far(pairs, corpus):
    """Единственное, что здесь реально утверждается: близко лучше, чем далеко

    Поточечную монотонность отношения правдоподобий не проверяю: она не гарантирована,
    на редких дальних корзинах оно шумит даже после сглаживания
    """
    geo = GeoIndex.fit(pairs, corpus)
    near, middle, far = geo.score(np.array([1.0, 20.0, 800.0]))
    assert near > middle > far
    assert near > 0 > far


def test_empty_candidate_slots_get_infinite_distance(pairs, corpus):
    geo = GeoIndex.fit(pairs, corpus)
    distance = geo.distance(np.array([100]), np.array([[0, 1, -1]]))
    assert np.isfinite(distance[0, 0])
    assert distance[0, 2] == np.inf


def test_unknown_location_yields_neutral_score(pairs, corpus):
    """У запроса без восстановленного центра гео не должно ни помогать, ни мешать"""
    geo = GeoIndex.fit(pairs, corpus)
    latitude, longitude = geo.query_coordinates(np.array([999]))
    assert np.isnan(latitude[0]) and np.isnan(longitude[0])
    distance = geo.distance(np.array([999]), np.array([[0, 1]]))
    assert geo.score(distance).tolist() == [[0.0, 0.0]]


def test_bins_cover_everything():
    assert BIN_EDGES[0] == 0
    assert BIN_EDGES[-1] == np.inf
