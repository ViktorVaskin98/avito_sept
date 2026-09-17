from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

import pandas as pd

try:
    import Stemmer as _snowball  # noqa: N813

    _STEMMER = _snowball.Stemmer("russian")
except ImportError:  # pragma: no cover
    _STEMMER = None

_NON_WORD = re.compile(r"[^0-9a-zа-я]+")
_MULTISPACE = re.compile(r"\s+")

MIN_TOKEN_LENGTH = 2

# обычный русский стоп-лист. В заголовках таких слов мало, а вот в описаниях они
# заметно шумят в BM25
STOPWORDS: frozenset[str] = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей
    может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
    тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом
    один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец
    два об другой хоть после над больше тот через эти нас про всего них какая много
    разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой
    им более всегда конечно всю между
    """.split()
)


def normalize(text: str | None) -> str:
    """Нижний регистр, только буквы и цифры, одиночные пробелы

    ё схлопываю в е, в пользовательских запросах написание не устойчивое
    """
    if not text:
        return ""
    lowered = text.lower().replace("ё", "е")
    cleaned = _NON_WORD.sub(" ", lowered)
    return _MULTISPACE.sub(" ", cleaned).strip()


@lru_cache(maxsize=1_000_000)
def stem(token: str) -> str:
    if _STEMMER is None:
        return token
    return _STEMMER.stemWord(token)


def tokenize(
    text: str | None,
    *,
    do_stem: bool = True,
    drop_stopwords: bool = True,
    min_length: int = MIN_TOKEN_LENGTH,
) -> list[str]:
    tokens = normalize(text).split()
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if min_length > 1:
        tokens = [token for token in tokens if len(token) >= min_length]
    if do_stem:
        tokens = [stem(token) for token in tokens]
    return tokens


def char_ngrams(text: str | None, sizes: Sequence[int] = (3, 4, 5)) -> list[str]:
    """Символьные n-граммы по словам с граничными маркерами

    Маркер _ на краях слова не даёт n-грамме склеиться через пробел и заодно отличает
    начало слова от середины, для пары «ремонт» и «капремонт» это важно
    """
    normalized = normalize(text)
    if not normalized:
        return []
    grams: list[str] = []
    for word in normalized.split():
        padded = f"_{word}_"
        for size in sizes:
            if len(padded) < size:
                continue
            grams.extend(padded[start : start + size] for start in range(len(padded) - size + 1))
    return grams


def normalize_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        [normalize(value) for value in series.astype("string").fillna("")], index=series.index
    )


def join_fields(*fields: Iterable[str | None]) -> list[str]:
    """Склеить несколько текстовых колонок построчно, пропуская пустые значения"""
    columns = [list(field) for field in fields]
    length = len(columns[0])
    if any(len(column) != length for column in columns):
        raise ValueError("колонки разной длины")
    return [
        " ".join(str(column[row]) for column in columns if column[row]) for row in range(length)
    ]


def truncate(text: str | None, max_chars: int) -> str:
    """Обрезать по границе слова

    Описания в корпусе доходят до нескольких тысяч символов, но полезное почти всегда
    в начале, дальше идут условия работы и контакты
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars // 2 else cut
