import numpy as np
import pytest

from avito_cg.eval.metrics import candidate_coverage, recall_at_k, recall_curve, recall_per_query


def test_recall_matches_example_from_task():
    """Пример из условия: три запроса с вкладами 1.0, 0.5 и 0.0 дают Recall@50 = 0.5"""
    relevant = {"a": {"i1"}, "b": {"i1", "i2"}, "c": {"i9"}}
    predicted = {"a": ["i1"], "b": ["i1", "i7"], "c": ["i3", "i4"]}
    assert recall_at_k(relevant, predicted, k=50) == 0.5


def test_order_does_not_matter_within_k():
    relevant = {"a": {"i5"}}
    assert recall_at_k(relevant, {"a": ["i5", "i1"]}) == recall_at_k(relevant, {"a": ["i1", "i5"]})


def test_cutoff_is_respected():
    relevant = {"a": {"i50"}}
    predicted = {"a": [f"i{n}" for n in range(1, 50)] + ["i50"]}
    assert recall_at_k(relevant, predicted, k=50) == 1.0
    assert recall_at_k(relevant, predicted, k=49) == 0.0


def test_missing_query_counts_as_zero():
    relevant = {"a": {"i1"}, "b": {"i2"}}
    assert recall_at_k(relevant, {"a": ["i1"]}) == 0.5


def test_per_query_scores_align_with_relevant_keys():
    relevant = {"a": {"i1"}, "b": {"i2"}}
    scores = recall_per_query(relevant, {"a": ["i1"], "b": []})
    assert np.array_equal(scores, np.array([1.0, 0.0]))


def test_recall_curve_is_monotonic():
    relevant = {"a": {"i3"}, "b": {"i1", "i9"}}
    predicted = {"a": ["i1", "i2", "i3"], "b": ["i1", "i2", "i3"]}
    curve = recall_curve(relevant, predicted, ks=(1, 2, 3, 10))
    values = [curve[k] for k in (1, 2, 3, 10)]
    assert values == sorted(values)


def test_candidate_coverage_ignores_top_k_cutoff():
    relevant = {"a": {"i99"}}
    candidates = {"a": [f"i{n}" for n in range(200)]}
    assert candidate_coverage(relevant, candidates) == 1.0


def test_paired_bootstrap_detects_a_consistent_small_gain():
    """Маленький, но систематический прирост должен быть значимым, а шум нет

    Непарное сравнение такого не увидит: собственная дисперсия каждой оценки
    на порядок больше разности
    """
    from avito_cg.eval.metrics import bootstrap_ci, paired_bootstrap

    rng = np.random.default_rng(0)
    before = rng.random(2452)
    after = np.clip(before + 0.01, 0, 1)

    delta, low, high = paired_bootstrap(before, after)
    assert delta == pytest.approx(0.01, abs=0.001)
    assert low > 0

    wide = bootstrap_ci(before)
    assert (wide[1] - wide[0]) > (high - low)


def test_paired_bootstrap_calls_noise_insignificant():
    from avito_cg.eval.metrics import paired_bootstrap

    rng = np.random.default_rng(1)
    before = rng.random(2452)
    after = rng.random(2452)
    _, low, high = paired_bootstrap(before, after)
    assert low < 0 < high


def test_paired_bootstrap_skips_queries_without_truth():
    from avito_cg.eval.metrics import paired_bootstrap

    before = np.array([0.0, np.nan, 1.0])
    after = np.array([1.0, 0.5, 1.0])
    delta, _, _ = paired_bootstrap(before, after)
    assert delta == pytest.approx(0.5)


def test_root_can_be_overridden_by_environment(monkeypatch, tmp_path):
    """При установке не в editable-режиме корень уезжает в site-packages

    Наступил бы на это на Kaggle: пакет ставится через pip, а данные лежат рядом
    с ноутбуком, и без переопределения avito-cg искал бы их внутри site-packages
    """
    import importlib

    monkeypatch.setenv("AVITO_CG_ROOT", str(tmp_path))
    import avito_cg.config as config

    importlib.reload(config)
    assert config.PATHS.raw == tmp_path / "data" / "raw"

    monkeypatch.delenv("AVITO_CG_ROOT")
    importlib.reload(config)
