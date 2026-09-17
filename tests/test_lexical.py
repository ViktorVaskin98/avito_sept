import numpy as np
import pytest

from avito_cg.index.lexical import BM25FIndex, Field


def words(text):
    """Тривиальный токенизатор: в тестах нужна проверяемая руками арифметика, без стеммера"""
    return text.split()


def build(documents, fields=None, **kwargs):
    index = BM25FIndex(
        fields or [Field("title", 1.0, 0.75), Field("description", 1.0, 0.75)],
        tokenizer=words,
        **kwargs,
    )
    return index.fit(documents)


def test_exact_match_wins():
    index = build(
        {
            "title": ["кран аренда", "маникюр", "репетитор математика"],
            "description": ["", "", ""],
        }
    )
    order, scores = index.search(["кран"], top_k=3)
    assert order[0, 0] == 0
    assert scores[0, 0] > 0
    assert scores[0, 1] == 0


def test_rare_term_outweighs_common_one():
    """Терм из одного документа должен весить больше, чем терм из всех"""
    common = ["услуга"] * 20
    index = build(
        {
            "title": [f"{word} уникальный" if n == 0 else word for n, word in enumerate(common)],
            "description": [""] * 20,
        }
    )
    _, rare_scores = index.search(["уникальный"], top_k=1)
    _, common_scores = index.search(["услуга"], top_k=1)
    assert rare_scores[0, 0] > common_scores[0, 0]


def test_field_boost_changes_ranking():
    documents = {
        "title": ["кран", "бетон"],
        "description": ["бетон", "кран"],
    }
    index = build(documents, fields=[Field("title", 5.0, 0.75), Field("description", 1.0, 0.75)])
    assert index.search(["кран"], top_k=2)[0][0, 0] == 0

    index.reweight([Field("title", 0.1, 0.75), Field("description", 5.0, 0.75)])
    assert index.search(["кран"], top_k=2)[0][0, 0] == 1


def test_length_normalisation_prefers_short_documents():
    filler = " ".join(f"слово{n}" for n in range(30))
    index = build(
        {
            "title": ["кран", f"кран {filler}"],
            "description": ["", ""],
        }
    )
    order, scores = index.search(["кран"], top_k=2)
    assert order[0, 0] == 0
    assert scores[0, 0] > scores[0, 1]

    index.reweight([Field("title", 1.0, 0.0), Field("description", 1.0, 0.0)])
    _, flat = index.search(["кран"], top_k=2)
    assert flat[0, 0] == pytest.approx(flat[0, 1])


def test_saturation_is_sublinear():
    """Десять вхождений терма должны стоить меньше, чем десять документов по одному"""
    index = build(
        {
            "title": ["кран", "кран кран кран кран кран кран кран кран кран кран"],
            "description": ["", ""],
        },
        fields=[Field("title", 1.0, 0.0), Field("description", 1.0, 0.0)],
    )
    _, scores = index.search(["кран"], top_k=2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[0, 0] < 10 * scores[0, 1]


def test_fields_are_combined_before_saturation():
    """Терм в двух полях весит больше, чем в одном, но меньше, чем вдвое

    Это и отличает BM25F от суммы независимых BM25: насыщение применяется
    один раз к общему весу
    """
    index = build(
        {
            "title": ["кран", "кран"],
            "description": ["", "кран"],
        },
        fields=[Field("title", 1.0, 0.0), Field("description", 1.0, 0.0)],
    )
    _, scores = index.search(["кран"], top_k=2)
    single, both = sorted(scores[0, :2])
    assert both > single
    assert both < 2 * single


def test_unknown_and_empty_queries_return_nothing():
    index = build({"title": ["кран"], "description": [""]})
    order, scores = index.search(["совершенно другое слово", ""], top_k=3)
    assert (order == -1).all()
    assert (scores == 0).all()


def test_document_frequency_counts_documents_not_fields():
    """Терм в заголовке и описании одного объявления это один документ, а не два"""
    index = build({"title": ["кран"], "description": ["кран"]})
    column = index.vocabulary["кран"]
    expected = np.log(1 + (1 - 1 + 0.5) / (1 + 0.5))
    assert index.idf[column] == pytest.approx(expected, rel=1e-5)


def test_min_df_prunes_vocabulary():
    documents = {"title": ["кран", "кран", "редкость"], "description": ["", "", ""]}
    assert "редкость" in build(documents, min_df=1).vocabulary
    assert "редкость" not in build(documents, min_df=2).vocabulary


def test_missing_field_is_reported():
    with pytest.raises(KeyError, match="нет текстов для полей"):
        build({"title": ["кран"]})


def test_fields_of_different_length_are_reported():
    with pytest.raises(ValueError, match="поля разной длины"):
        build({"title": ["кран", "бетон"], "description": [""]})
