"""Закреплённый словарь ключей параметров

Пока словарь собирался на лету, его содержимое зависело от того, какая команда
запустилась первой: make-answer выводил его из benchmark_items, baseline из локального
корпуса, encode-corpus то из одного, то из другого. Получались словари на 99, 92 и 96
ключей, расходящиеся на три десятка. Вместе с ними расходилось поле params индекса,
скоры BM25F и итоговый answer.csv, то есть повторный прогон давал другой ответ.

Теперь словарь версионируется вместе с кодом. Эти тесты стерегут ровно это свойство.
"""

import inspect
import json

from avito_cg.data.params import ParamsParser
from avito_cg.index.fields import PINNED_KEYS, load_parser, params_text

SAMPLE = (
    "Вид услуги Автосервис, аренда Тип услуги Аренда авто "
    "Место оказания услуг Санкт-Петербург, Невский пр-т, 190 "
    "Тип транспорта Водный транспорт"
)


def test_pinned_file_ships_with_the_package():
    assert PINNED_KEYS.exists(), f"нет закреплённого словаря: {PINNED_KEYS}"
    keys = json.loads(PINNED_KEYS.read_text(encoding="utf-8"))
    assert isinstance(keys, list)
    assert len(keys) > 50
    assert all(isinstance(key, str) and key for key in keys)
    assert len(set(keys)) == len(keys), "в словаре дубли"


def test_loader_takes_no_corpus():
    """Главное свойство: словарь один и тот же, кто бы его ни просил

    У load_parser нет аргументов, и это часть контракта: пока он выводил словарь
    из переданных текстов, разные команды получали разные словари
    """
    assert not inspect.signature(load_parser).parameters
    assert load_parser().keys == load_parser().keys


def test_pinned_dictionary_parses_a_real_string():
    """Ключи из словаря режут строку на пары, значения остаются между ними

    Разбор не идеальный, и это записано в EDA: «Тип транспорта» в словарь не попал,
    потому что стоит редко, и его кусок уезжает в значение предыдущего ключа.
    Тест сторожит то, что должно держаться, а не притворяется, что разбор точный
    """
    parser = load_parser()
    parsed = parser.parse(SAMPLE)
    assert parsed.first("Вид услуги") == "Автосервис, аренда"
    assert parsed.order[0] == "Вид услуги"
    assert "Место оказания услуг" in parsed.order


def test_string_starts_with_a_key():
    """Покрытие меряет, сколько токенов ушло в «ничей» префикс до первого ключа"""
    assert load_parser().coverage(SAMPLE) == 1.0


def test_index_field_keeps_values_and_drops_keys():
    """В индекс идут только значения: ключи это служебная лексика без различающей силы"""
    parser = load_parser()
    text = params_text(parser, [SAMPLE])[0]
    assert "Автосервис, аренда" in text
    assert "Вид услуги" not in text
    assert "Место оказания услуг" not in text


def test_numeric_values_are_dropped():
    parser = load_parser()
    text = params_text(parser, ["График работы от 9 График работы до 18"])[0]
    assert text == ""


def test_load_and_save_roundtrip(tmp_path):
    parser = load_parser()
    copy = tmp_path / "keys.json"
    parser.save(copy)
    assert ParamsParser.load(copy).keys == parser.keys
