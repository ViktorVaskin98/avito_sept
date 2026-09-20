"""Сборка и проверка answer.csv

В условии отдельно предупреждают: криво записанные item_id не вызывают ошибку при загрузке,
они просто не засчитываются и молча занимают место в ответе. Терять из-за этого одну
из семи попыток не хочется, поэтому формат проверяю здесь жёстко и до отправки
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import pandas as pd

from avito_cg.config import ID_LENGTH, TOP_K

ITEM_ID_PATTERN = re.compile(rf"^[0-9a-f]{{{ID_LENGTH}}}$")

# формат такое пропускает, метрика молча просядет. Ошибкой не считаю,
# но и молчать о таком нельзя
WARNING_PREFIXES = ("пустых ответов",)


def split_problems(problems: Sequence[str]) -> tuple[list[str], list[str]]:
    """Разделить претензии на те, что ломают формат, и те, что просто стоят метрики"""
    warnings = [p for p in problems if p.startswith(WARNING_PREFIXES)]
    return [p for p in problems if p not in warnings], warnings


class SubmissionError(ValueError):
    """Формат ответа нарушен, лучше упасть локально"""


def build_submission(predictions: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Словарь query_id -> список item_id превращаю в две колонки нужного формата"""
    query_ids = list(predictions)
    answers = [" ".join(predictions[query_id]) for query_id in query_ids]
    return pd.DataFrame({"query_id": query_ids, "answer": answers})


def validate_submission(
    frame: pd.DataFrame,
    *,
    expected_query_ids: Collection[str],
    corpus_item_ids: Collection[str] | None = None,
    top_k: int = TOP_K,
) -> list[str]:
    """Список претензий к файлу, пустой означает что файл готов к отправке"""
    problems: list[str] = []

    if list(frame.columns) != ["query_id", "answer"]:
        problems.append(
            f"колонки должны быть ровно ['query_id', 'answer'], а не {list(frame.columns)}"
        )
        return problems

    query_ids = frame["query_id"].astype(str)
    expected = set(expected_query_ids)
    got = set(query_ids)

    if query_ids.duplicated().any():
        problems.append(f"повторяющихся query_id: {int(query_ids.duplicated().sum())}")
    if missing := expected - got:
        problems.append(f"не хватает query_id: {len(missing)}, например {sorted(missing)[:3]}")
    if extra := got - expected:
        problems.append(f"лишние query_id: {len(extra)}, например {sorted(extra)[:3]}")

    bad_length = query_ids[query_ids.str.len() != ID_LENGTH]
    if not bad_length.empty:
        problems.append(
            f"query_id не длины {ID_LENGTH}: {len(bad_length)} шт, "
            f"скорее всего идентификаторы где-то привелись к числу"
        )

    corpus = set(corpus_item_ids) if corpus_item_ids is not None else None
    too_long = 0
    duplicated_inside = 0
    malformed: set[str] = set()
    outside: set[str] = set()
    empty_answers = 0

    for raw in frame["answer"].astype(str):
        items = [item for item in raw.split(" ") if item] if raw else []
        if not items:
            empty_answers += 1
            continue
        if len(items) > top_k:
            too_long += 1
        if len(set(items)) != len(items):
            duplicated_inside += 1
        for item in items:
            if not ITEM_ID_PATTERN.match(item):
                malformed.add(item)
            elif corpus is not None and item not in corpus:
                outside.add(item)

    if too_long:
        problems.append(f"строк с более чем {top_k} item_id: {too_long}")
    if duplicated_inside:
        problems.append(f"строк с повторами item_id внутри ответа: {duplicated_inside}")
    if malformed:
        problems.append(
            f"item_id не в формате {ID_LENGTH} символов [0-9a-f]: {len(malformed)} шт, "
            f"например {sorted(malformed)[:3]}"
        )
    if outside:
        problems.append(
            f"item_id, которых нет в корпусе: {len(outside)} шт, например {sorted(outside)[:3]}"
        )
    if empty_answers:
        problems.append(
            f"пустых ответов: {empty_answers}, формально можно но это гарантированный ноль"
        )

    return problems


def save_submission(
    predictions: Mapping[str, Sequence[str]],
    path: Path,
    *,
    expected_query_ids: Collection[str],
    corpus_item_ids: Collection[str] | None = None,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Собрать, проверить и записать, при нарушении формата кидаю исключение а не файл"""
    frame = build_submission(predictions)
    problems = validate_submission(
        frame,
        expected_query_ids=expected_query_ids,
        corpus_item_ids=corpus_item_ids,
        top_k=top_k,
    )
    blocking, _ = split_problems(problems)
    if blocking:
        raise SubmissionError("; ".join(blocking))
    path.parent.mkdir(parents=True, exist_ok=True)
    # перевод строки фиксирую явно: по умолчанию pandas берёт его у операционной системы,
    # и один и тот же ответ на Windows и на Linux получается побайтово разным. Платформе
    # это безразлично, а вот проверить воспроизводимость сверкой файлов уже нельзя
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return frame


def read_submission(path: Path) -> pd.DataFrame:
    """Читаю ответ обратно только как строки, иначе теряются ведущие нули"""
    return pd.read_csv(path, dtype={"query_id": str, "answer": str}, keep_default_na=False)
