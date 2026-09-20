import numpy as np
import pandas as pd
import pytest

from avito_cg.data.io import QUERY_FEATURE_COLUMNS, query_key
from avito_cg.eval.benchmark import LocalBenchmark, synthetic_query_id
from avito_cg.train.extra import build_donor_set


def row(query, item):
    return {
        "search_query": query,
        "search_location_id": 1,
        "search_is_delivery_search": 0,
        "search_infm_params_text": "",
        "search_category": 114,
        "item_id": item,
    }


@pytest.fixture
def train():
    """Одна обучающая пара, одна отложенная, пять донорских

    Отложенный запрос выбрал знакомое объявление A и незнакомое U1, то есть доля знакомых
    среди его позитивов ровно 0.5. Столько же должно получиться у донорского набора
    """
    return pd.DataFrame(
        [
            row("маникюр", "A"),
            row("маникюр на дому", "A"),
            row("маникюр на дому", "U1"),
            row("педикюр", "A"),
            row("брови", "U2"),
            row("ресницы", "U3"),
            row("стрижка", "U4"),
            row("массаж", "U5"),
        ]
    )


@pytest.fixture
def local(train):
    keys = query_key(train).to_numpy()
    validation = synthetic_query_id(keys[1])
    return LocalBenchmark(
        queries=train.iloc[[1]][list(QUERY_FEATURE_COLUMNS)].assign(query_id=validation),
        relevant={validation: {"A", "U1"}},
        corpus_item_ids=np.array(["A", "U1", "U2", "U3", "U4", "U5"], dtype=object),
        fit_rows=np.array([0]),
        meta={},
    )


def test_donor_set_matches_seen_share(train, local):
    queries, relevant, report = build_donor_set(train, local, n_queries=10)

    assert report["доля знакомых у позитивов"] == pytest.approx(report["цель по отложенным"])
    assert report["цель по отложенным"] == pytest.approx(0.5)
    assert set(queries.columns) == {"query_id", *QUERY_FEATURE_COLUMNS}
    assert len(queries) == len(relevant)


def test_donor_set_excludes_fit_and_validation(train, local):
    queries, _, _ = build_donor_set(train, local, n_queries=10)
    keys = query_key(train).to_numpy()
    forbidden = {synthetic_query_id(keys[0]), synthetic_query_id(keys[1])}
    assert forbidden.isdisjoint(set(queries["query_id"]))


def test_donor_positives_lie_in_corpus(train, local):
    _, relevant, _ = build_donor_set(train, local, n_queries=10)
    corpus = set(local.corpus_item_ids.tolist())
    assert all(items <= corpus for items in relevant.values())
