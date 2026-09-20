import numpy as np
import pandas as pd
import pytest

from avito_cg.index.lexical import BM25FIndex, Field
from avito_cg.train.reranker import FEATURES, ItemContext, _ranks, build_features


def words(text):
    return str(text).split()


@pytest.fixture
def index():
    return BM25FIndex(
        [Field("title", 1.0, 0.75), Field("params", 1.0, 0.75), Field("description", 1.0, 0.75)],
        tokenizer=words,
    ).fit(
        {
            "title": ["маникюр на дому", "педикюр", "маникюр на дому", "ремонт ванной"],
            "params": ["", "", "", ""],
            "description": ["работаю на дому", "педикюр аппаратный", "", "плитка и сантехника"],
        }
    )


@pytest.fixture
def corpus():
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "item_title_raw": ["Маникюр на дому", "Педикюр", "маникюр на дому", "Ремонт ванной"],
            "item_latitude": [55.0, 55.1, 60.0, 55.0],
            "item_longitude": [37.0, 37.1, 30.0, 37.0],
            "item_location_id": [1, 1, 2, 1],
            "item_price": [1000.0, 1500.0, 900.0, 20000.0],
            "item_rating": [5.0, 4.0, 4.5, 3.0],
            "item_rating_reviews_count": [10.0, 2.0, 7.0, 1.0],
            "item_is_phone_hidden": [False, True, False, False],
            "item_is_message_forbidden": [False, False, True, False],
        }
    )


@pytest.fixture
def pairs():
    return pd.DataFrame(
        {
            "search_query": ["маникюр", "маникюр", "педикюр"],
            "item_id": ["a", "a", "b"],
        }
    )


def test_coverage_matches_direct_slice(index):
    """Быстрый путь через обратный список должен совпадать с прямым срезом матрицы частот"""
    presence = index.presence("title")
    terms = index.query_terms("маникюр на дому")
    items = np.array([0, 1, 2, 3])

    direct = np.asarray((index._field_counts["title"][items][:, terms] > 0).todense())
    assert np.allclose(presence.coverage(terms, items, None), direct.mean(axis=1))


def test_coverage_weighted_by_idf(index):
    """С весами по idf редкий терм должен тянуть покрытие сильнее частого"""
    presence = index.presence("title")
    terms = index.query_terms("маникюр ванной")
    weights = index.idf[terms]
    covered = presence.coverage(terms, np.array([3]), weights)[0]
    assert 0.5 < covered < 1.0


def test_coverage_without_terms_is_zero(index):
    presence = index.presence("title")
    assert presence.coverage(np.array([], dtype=int), np.array([0, 1]), None).tolist() == [0.0, 0.0]


def test_popularity_is_scale_free(index, corpus, pairs):
    """Счётчик делится на средний по знакомым, иначе он не переносится между корпусами"""
    context = ItemContext.build(corpus, index, pairs)
    doubled = ItemContext.build(corpus, index, pd.concat([pairs, pairs], ignore_index=True))
    assert np.allclose(context.popularity, doubled.popularity)
    assert context.popularity[[2, 3]].tolist() == [0.0, 0.0]
    assert context.popularity[0] > context.popularity[1]


def test_memory_keeps_only_corpus_items(index, corpus, pairs):
    context = ItemContext.build(corpus, index, pairs)
    assert context.memory["маникюр"].tolist() == [0]
    assert "стрижка" not in context.memory


def test_context_without_pairs_disables_priors(index, corpus):
    context = ItemContext.build(corpus, index, None)
    assert context.popularity.sum() == 0
    assert context.memory == {}


def test_twins_counted_after_normalization(index, corpus):
    """«Маникюр на дому» и «маникюр на дому» это один и тот же заголовок"""
    context = ItemContext.build(corpus, index, None)
    assert context.twins[0] == 2
    assert context.twins[3] == 1


def test_ranks_are_positions_by_descending_score():
    assert _ranks(np.array([0.1, 0.9, 0.5])).tolist() == [2.0, 0.0, 1.0]


def test_features_have_declared_shape_and_labels(index, corpus, pairs):
    context = ItemContext.build(corpus, index, pairs)
    queries = pd.DataFrame(
        {
            "query_id": ["q1"],
            "search_query": ["маникюр на дому"],
            "search_location_id": [1],
            "search_infm_params_text": [""],
        }
    )
    candidates = [np.array([2, 0, 3])]
    scores = {
        name: [np.array([0.3, 0.9, 0.1])]
        for name in ("гео", "фасеты", "микрокатегория", "плотный", "итог")
    }
    data = build_features(
        candidates,
        [np.array([1.0, 2.0, 0.5])],
        scores,
        queries,
        context,
        index,
        query_latitude=np.array([55.0]),
        query_longitude=np.array([37.0]),
        query_ids=["q1"],
        relevant={"q1": {"a"}},
    )

    assert data.features.shape == (3, len(FEATURES))
    assert data.item.tolist() == [2, 0, 3]
    assert data.label.tolist() == [0, 1, 0]

    column = {name: number for number, name in enumerate(FEATURES)}
    # кандидаты идут в порядке 2, 0, 3, а не по возрастанию номера: признаки не должны
    # перепутаться между строками из-за сортировки внутри обратного списка
    assert data.features[:, column["покрытие_заголовка"]].tolist() == [1.0, 1.0, 0.0]
    assert data.features[:, column["та_же_локация"]].tolist() == [0.0, 1.0, 1.0]
    assert data.features[:, column["знакомое"]].tolist() == [0.0, 1.0, 0.0]


def test_features_without_labels(index, corpus, pairs):
    """На настоящем бенчмарке разметки нет, метки должны выходить нулями, а не падать"""
    context = ItemContext.build(corpus, index, pairs)
    queries = pd.DataFrame(
        {
            "query_id": ["q1"],
            "search_query": ["педикюр"],
            "search_location_id": [1],
            "search_infm_params_text": [""],
        }
    )
    scores = {
        name: [np.array([0.5, 0.5])]
        for name in ("гео", "фасеты", "микрокатегория", "плотный", "итог")
    }
    data = build_features(
        [np.array([0, 1])],
        [np.array([1.0, 1.0])],
        scores,
        queries,
        context,
        index,
        query_latitude=np.array([55.0]),
        query_longitude=np.array([37.0]),
        query_ids=["q1"],
    )
    assert data.label.tolist() == [0, 0]
