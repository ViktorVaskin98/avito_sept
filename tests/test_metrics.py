import numpy as np

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
