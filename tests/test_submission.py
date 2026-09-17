import pandas as pd
import pytest

from avito_cg.eval.submission import (
    SubmissionError,
    build_submission,
    read_submission,
    save_submission,
    validate_submission,
)

QUERY_IDS = ["00WuFMaXSFZBxSzT", "03ztb1gtRFC4K4vP"]
ITEMS = ["1382564bf8994a83", "121fa7f5e765ce00", "f3801ec4fa472597"]


def test_build_joins_items_with_single_space():
    frame = build_submission({QUERY_IDS[0]: ITEMS})
    assert frame.loc[0, "answer"] == " ".join(ITEMS)
    assert list(frame.columns) == ["query_id", "answer"]


def test_valid_submission_has_no_problems():
    frame = build_submission({QUERY_IDS[0]: ITEMS[:2], QUERY_IDS[1]: ITEMS[2:]})
    assert validate_submission(frame, expected_query_ids=QUERY_IDS, corpus_item_ids=ITEMS) == []


def test_uppercase_item_id_is_rejected():
    frame = build_submission({QUERY_IDS[0]: [ITEMS[0].upper()], QUERY_IDS[1]: [ITEMS[1]]})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS, corpus_item_ids=ITEMS)
    assert any("0-9a-f" in problem for problem in problems)


def test_item_outside_corpus_is_reported():
    frame = build_submission({QUERY_IDS[0]: ["0" * 16], QUERY_IDS[1]: [ITEMS[0]]})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS, corpus_item_ids=ITEMS)
    assert any("нет в корпусе" in problem for problem in problems)


def test_more_than_top_k_is_reported():
    frame = build_submission({QUERY_IDS[0]: [f"{n:016x}" for n in range(51)]})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS[:1])
    assert any("более чем 50" in problem for problem in problems)


def test_duplicates_inside_answer_are_reported():
    frame = build_submission({QUERY_IDS[0]: [ITEMS[0], ITEMS[0]]})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS[:1])
    assert any("повторами item_id" in problem for problem in problems)


def test_missing_query_id_is_reported():
    frame = build_submission({QUERY_IDS[0]: ITEMS[:1]})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS)
    assert any("не хватает query_id" in problem for problem in problems)


def test_save_refuses_broken_submission(tmp_path):
    with pytest.raises(SubmissionError):
        save_submission(
            {QUERY_IDS[0]: [ITEMS[0].upper()]},
            tmp_path / "answer.csv",
            expected_query_ids=QUERY_IDS[:1],
        )
    assert not (tmp_path / "answer.csv").exists()


def test_roundtrip_keeps_ids_as_strings(tmp_path):
    """Ловушка из условия: id из цифр читается как число и теряет ведущие нули"""
    numeric_like = "0012345678901234"
    path = tmp_path / "answer.csv"
    save_submission(
        {numeric_like: [ITEMS[0]]},
        path,
        expected_query_ids=[numeric_like],
        corpus_item_ids=ITEMS,
    )
    frame = read_submission(path)
    assert frame.loc[0, "query_id"] == numeric_like
    assert pd.api.types.is_string_dtype(frame["query_id"])


def test_wrong_columns_short_circuit():
    frame = pd.DataFrame({"query_id": QUERY_IDS[:1], "answer": ["x"], "score": [1.0]})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS[:1])
    assert len(problems) == 1
    assert "колонки" in problems[0]


def test_empty_answer_is_a_warning_not_an_error():
    """Пустой ответ формат не ломает, но метрику стоит, поэтому он отдельной категорией"""
    from avito_cg.eval.submission import split_problems

    frame = build_submission({QUERY_IDS[0]: ITEMS[:1], QUERY_IDS[1]: []})
    problems = validate_submission(frame, expected_query_ids=QUERY_IDS, corpus_item_ids=ITEMS)
    errors, warnings = split_problems(problems)
    assert errors == []
    assert len(warnings) == 1
    assert "пустых ответов" in warnings[0]


def test_save_allows_empty_answers_but_not_broken_ids(tmp_path):
    path = tmp_path / "answer.csv"
    save_submission(
        {QUERY_IDS[0]: ITEMS[:1], QUERY_IDS[1]: []},
        path,
        expected_query_ids=QUERY_IDS,
        corpus_item_ids=ITEMS,
    )
    assert path.exists()
