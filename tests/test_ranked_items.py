"""Отбор топ-k внутри запроса

Одна функция обслуживает четыре места: выдачу до слияния, выдачу после переранжирования,
сборку ответа и разбор промахов. Раньше это были четыре копии одного цикла, и каждая
могла разъехаться со своей. Тест сторожит инварианты, на которые они все опираются.
"""

import numpy as np

from avito_cg.train.reranker import CandidateSet, ranked_items


def candidates(query, item):
    size = len(query)
    return CandidateSet(
        query=np.array(query),
        item=np.array(item),
        features=np.zeros((size, 1), dtype=np.float32),
        label=np.zeros(size, dtype=int),
    )


def test_orders_by_descending_score_inside_query():
    data = candidates([0, 0, 0, 1, 1], [10, 11, 12, 20, 21])
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    order = ranked_items(data, scores, 2, top_k=3)
    assert order[0].tolist() == [11, 12, 10]
    assert order[1].tolist() == [21, 20, -1]


def test_none_keeps_row_order():
    """collect кладёт кандидатов уже по убыванию скора слияния, пересортировка не нужна"""
    data = candidates([0, 0, 0], [10, 11, 12])
    assert ranked_items(data, None, 1, top_k=3)[0].tolist() == [10, 11, 12]


def test_truncates_to_top_k():
    data = candidates([0] * 5, [1, 2, 3, 4, 5])
    scores = np.arange(5, dtype=float)[::-1]
    assert ranked_items(data, scores, 1, top_k=2)[0].tolist() == [1, 2]


def test_query_without_rows_is_all_empty():
    data = candidates([0, 2], [7, 9])
    order = ranked_items(data, np.array([1.0, 1.0]), 3, top_k=2)
    assert order[1].tolist() == [-1, -1]
    assert order[0].tolist() == [7, -1]
    assert order[2].tolist() == [9, -1]


def test_ties_are_broken_stably():
    """Одинаковые скоры не должны давать разный ответ от запуска к запуску:
    воспроизводимость answer.csv держится в том числе на этом"""
    data = candidates([0, 0, 0], [10, 11, 12])
    scores = np.array([0.5, 0.5, 0.5])
    first = ranked_items(data, scores, 1, top_k=3)
    assert first.tolist() == ranked_items(data, scores, 1, top_k=3).tolist()
    assert first[0].tolist() == [10, 11, 12]


def test_rows_of_one_query_do_not_leak_into_another():
    data = candidates([0, 1, 0, 1], [1, 2, 3, 4])
    order = ranked_items(data, np.array([0.9, 0.9, 0.1, 0.1]), 2, top_k=2)
    assert set(order[0].tolist()) == {1, 3}
    assert set(order[1].tolist()) == {2, 4}
