"""Отпечаток конфигурации у кэша таблицы признаков

Сбор кандидатов это семь минут, поэтому таблица кэшируется. Кэш без привязки
к конфигурации опаснее, чем его отсутствие: поменял вес поля, команда отработала
за минуту и выдала ровно тот же ответ, что и до правки, потому что признаки приехали
из файла, собранного под старые веса. Молча.
"""

import pytest

from avito_cg.index.lexical import DEFAULT_FIELDS
from avito_cg.train.reranker import FEATURES, features_fingerprint

FIELDS = [(field.name, field.boost, field.b) for field in DEFAULT_FIELDS]
WEIGHTS = {"гео": 0.10, "фасеты": 0.10, "микрокатегория": 0.02, "плотный": 0.5}


def stamp(**overrides):
    payload = {
        "fields": FIELDS,
        "weights": WEIGHTS,
        "k1": 1.2,
        "mode": "normalized",
        "depth": 200,
        "dense_depth": 200,
    }
    payload.update(overrides)
    return features_fingerprint(
        payload["fields"],
        payload["weights"],
        k1=payload["k1"],
        mode=payload["mode"],
        depth=payload["depth"],
        dense_depth=payload["dense_depth"],
    )


def test_same_configuration_gives_same_stamp():
    assert stamp() == stamp()


def test_weight_order_does_not_matter():
    """Словарь весов приходит из разных мест, порядок ключей не должен ломать кэш"""
    shuffled = dict(reversed(list(WEIGHTS.items())))
    assert stamp(weights=shuffled) == stamp()


@pytest.mark.parametrize(
    "overrides",
    [
        {"fields": [("title", 5.0, 0.6), ("params", 0.5, 0.75), ("description", 1.0, 0.75)]},
        {"fields": [("title", 20.0, 0.9), ("params", 0.5, 0.75), ("description", 1.0, 0.75)]},
        {"weights": {**WEIGHTS, "гео": 0.15}},
        {"weights": {**WEIGHTS, "плотный": 0.0}},
        {"k1": 0.6},
        {"mode": "rrf"},
        {"depth": 500},
        {"dense_depth": 1000},
    ],
    ids=[
        "вес поля",
        "нормировка поля",
        "вес гео",
        "вес плотного",
        "насыщение k1",
        "режим",
        "глубина",
        "плотных",
    ],
)
def test_any_change_moves_the_stamp(overrides):
    assert stamp(**overrides) != stamp()


def test_feature_list_is_part_of_the_stamp(monkeypatch):
    """Смена набора признаков меняет ширину таблицы, кэш обязан протухнуть"""
    before = stamp()
    monkeypatch.setattr("avito_cg.train.reranker.FEATURES", [*FEATURES, "новый_признак"])
    assert stamp() != before
