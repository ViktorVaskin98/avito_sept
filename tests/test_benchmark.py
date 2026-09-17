import numpy as np
import pandas as pd
import pytest

from avito_cg.eval import benchmark as bm

N_QUERIES = 60
CORPUS_SIZE = 600


@pytest.fixture
def train() -> pd.DataFrame:
    """Синтетический train той же формы, что настоящий

    Запросов заметно больше, чем объявлений в корпусе: иначе донору неоткуда взять
    незнакомые объявления, и сплит честно падает
    """
    rng = np.random.default_rng(0)
    rows = []
    for index in range(2500):
        words = " ".join(f"слово{rng.integers(0, 200)}" for _ in range(rng.integers(1, 5)))
        empty = rng.random() < 0.35
        for _ in range(rng.choice([1, 1, 1, 2, 3])):
            rows.append(
                {
                    "search_query": words,
                    "search_location_id": int(rng.integers(600_000, 600_050)),
                    "search_is_delivery_search": 0,
                    "search_infm_params_text": "" if empty else "Вид услуги Ремонт и отделка",
                    "search_category": 114,
                    "item_id": f"{rng.integers(0, 3000):016x}",
                    "item_title_raw": f"объявление {index}",
                    "item_infm_params_text": "Вид услуги Ремонт и отделка",
                    "item_category_id": 114,
                    "item_microcat_id": int(rng.integers(0, 40)),
                    "item_location_id": int(rng.integers(600_000, 600_050)),
                    "item_latitude": float(rng.uniform(55, 60)),
                    "item_longitude": float(rng.uniform(30, 40)),
                    "item_price": 1000.0,
                    "item_rating": 5.0,
                    "item_rating_reviews_count": 10.0,
                    "item_is_phone_hidden": False,
                    "item_is_message_forbidden": False,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def benchmark_queries() -> pd.DataFrame:
    """Настоящие запросы почти всегда без фильтра, отсюда и целевое распределение"""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        [
            {
                "query_id": f"{index:016x}",
                "search_query": " ".join(f"с{rng.integers(0, 50)}" for _ in range(3)),
                "search_location_id": int(rng.integers(600_000, 600_050)),
                "search_is_delivery_search": 0,
                "search_infm_params_text": "" if rng.random() < 0.9 else "Вид услуги Ремонт",
                "search_category": 114,
            }
            for index in range(300)
        ]
    )


@pytest.fixture
def local(train, benchmark_queries):
    return bm.build(
        train,
        benchmark_queries,
        n_queries=N_QUERIES,
        corpus_size=CORPUS_SIZE,
        seen_share=0.1,
        seed=7,
    )


def test_sizes_match_request(local):
    assert len(local.queries) == N_QUERIES
    assert len(local.corpus_item_ids) == CORPUS_SIZE
    assert len(local.relevant) == N_QUERIES


def test_no_leakage(local, train):
    assert bm.leakage_checks(local, train) == []


def test_held_out_queries_are_absent_from_fit(local, train):
    from avito_cg.data.io import query_key

    fit_ids = {bm.synthetic_query_id(key) for key in query_key(train.iloc[local.fit_rows])}
    assert not fit_ids & set(local.relevant)


def test_all_positives_are_in_corpus(local):
    positives = {item for items in local.relevant.values() for item in items}
    assert positives <= set(local.corpus_item_ids)


def test_corpus_has_no_duplicates(local):
    assert len(set(local.corpus_item_ids)) == len(local.corpus_item_ids)


def test_seen_share_is_respected(local, train):
    fit_items = set(train.iloc[local.fit_rows]["item_id"].astype(str))
    seen = len(set(local.corpus_item_ids) & fit_items) / len(local.corpus_item_ids)
    assert abs(seen - 0.1) < 0.02


def test_distribution_matching_moves_empty_filter_share(train, benchmark_queries):
    """Без перевзвешивания доля пустых фильтров тянется к train, с ним к бенчмарку"""
    target = (benchmark_queries["search_infm_params_text"].str.len() == 0).mean()

    def empty_share(sample):
        return (sample.queries["search_infm_params_text"].astype(str).str.len() == 0).mean()

    matched = bm.build(
        train,
        benchmark_queries,
        n_queries=N_QUERIES,
        corpus_size=CORPUS_SIZE,
        seen_share=0.1,
        seed=7,
    )
    plain = bm.build(
        train,
        benchmark_queries,
        n_queries=N_QUERIES,
        corpus_size=CORPUS_SIZE,
        seen_share=0.1,
        match_distribution=False,
        seed=7,
    )
    assert abs(empty_share(matched) - target) < abs(empty_share(plain) - target)


def test_query_id_is_stable_and_well_formed():
    first = bm.synthetic_query_id("какой-то ключ")
    assert first == bm.synthetic_query_id("какой-то ключ")
    assert len(first) == 16
    assert set(first) <= set("0123456789abcdef")


def test_roundtrip_through_disk(local, tmp_path):
    local.save(tmp_path)
    restored = bm.LocalBenchmark.load(tmp_path)
    assert restored.relevant == local.relevant
    assert list(restored.corpus_item_ids) == list(local.corpus_item_ids)
    assert np.array_equal(restored.fit_rows, local.fit_rows)
    assert restored.meta == local.meta


def test_corpus_materialises_requested_columns(local, train):
    corpus = local.corpus(train, columns=["item_id", "item_title_raw", "item_microcat_id"])
    assert list(corpus.columns) == ["item_id", "item_title_raw", "item_microcat_id"]
    assert len(corpus) == CORPUS_SIZE


def test_impossible_split_fails_loudly(train, benchmark_queries):
    """Если незнакомых объявлений взять неоткуда, лучше упасть, чем тихо соврать"""
    with pytest.raises(ValueError, match="не хватает незнакомых"):
        bm.build(
            train,
            benchmark_queries,
            n_queries=N_QUERIES,
            corpus_size=len(train) * 10,
            seen_share=0.1,
            seed=7,
        )


def test_raking_matches_all_marginals_at_once():
    """Раскинг должен согласовать выборку сразу по трём признакам, а не по одному

    Именно это и было сломано в первой версии сплита: длина и фильтр совпадали,
    а локация нет, и локальный корпус вокруг запросов оказался вдвое разреженнее
    """
    pool = pd.DataFrame(
        {
            "_location": [1] * 900 + [2] * 100,
            "_empty_filter": ([True] * 450 + [False] * 450) + ([True] * 50 + [False] * 50),
            "_words": list(range(1, 4)) * 333 + [1],
        }
    )
    targets = {
        "_location": {1: 0.3, 2: 0.7},
        "_empty_filter": {True: 0.8, False: 0.2},
    }
    weights = bm.raking_weights(pool, targets)

    for column, target in targets.items():
        for value, share in target.items():
            got = weights[pool[column].to_numpy() == value].sum() / weights.sum()
            assert got == pytest.approx(share, abs=0.01), f"{column}={value}"


def test_raking_zeroes_categories_absent_from_benchmark():
    """Если по такой локации на платформе не ищут, в локальной выборке ей делать нечего"""
    pool = pd.DataFrame({"_location": [1, 1, 2, 3]})
    weights = bm.raking_weights(pool, {"_location": {1: 0.5, 2: 0.5}})
    assert weights[3] == 0
    assert weights[2] > 0


def test_raking_refuses_an_impossible_target():
    pool = pd.DataFrame({"_location": [7, 8]})
    with pytest.raises(ValueError, match="ни один запрос"):
        bm.raking_weights(pool, {"_location": {1: 1.0}})
