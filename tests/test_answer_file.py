"""Отправленный ответ лежит в репозитории и не должен испортиться

Он там для того, чтобы проверяющий мог пересобрать решение и сверить файлы побайтово,
а не верить на слово. Значит файл - часть решения, и у него должны быть тесты.

Всё, что требует настоящих данных (существуют ли item_id в корпусе, какой получается
Recall), проверяется командой `avito-cg validate-answer`. Здесь только то, что
проверяется без них.
"""

import re

import pytest

from avito_cg.config import ID_LENGTH, PATHS, TOP_K
from avito_cg.eval.submission import read_submission

ANSWER = PATHS.submissions / "answer.csv"
N_QUERIES = 2452
ITEM_ID = re.compile(rf"^[0-9a-f]{{{ID_LENGTH}}}$")

pytestmark = pytest.mark.skipif(not ANSWER.exists(), reason="ответ ещё не собран")


@pytest.fixture(scope="module")
def answer():
    return read_submission(ANSWER)


def test_columns_are_exactly_two(answer):
    assert list(answer.columns) == ["query_id", "answer"]


def test_one_row_per_benchmark_query(answer):
    assert len(answer) == N_QUERIES
    assert not answer["query_id"].duplicated().any()


def test_query_ids_are_strings_of_fixed_length(answer):
    """Приведение идентификаторов к числу теряет ведущие нули, и это молчаливая потеря"""
    assert (answer["query_id"].str.len() == ID_LENGTH).all()


def test_every_row_is_full_and_without_repeats(answer):
    items = answer["answer"].str.split()
    assert items.map(len).max() <= TOP_K
    assert items.map(len).min() > 0, "пустой ответ это гарантированный ноль по этому запросу"
    assert items.map(lambda row: len(set(row)) == len(row)).all()


def test_item_ids_are_lowercase_hex_of_fixed_length(answer):
    """Условие предупреждает отдельно: кривой item_id не вызовет ошибки при загрузке,
    он просто не засчитается и займёт место в ответе, а метрика молча просядет"""
    broken = {
        item
        for row in answer["answer"].str.split()
        for item in row
        if not ITEM_ID.match(item)
    }
    assert not broken, sorted(broken)[:5]


def test_file_is_stored_with_lf():
    """Иначе пересобранный на другой операционной системе файл не совпадёт побайтово"""
    assert b"\r\n" not in ANSWER.read_bytes()
