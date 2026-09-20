from __future__ import annotations

import re
from functools import lru_cache

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

    Проверка на строку тут не для красоты: пустое описание приезжает из parquet
    как NA, astype(str) на nullable-строке его не трогает, и в токенизатор
    прилетает float('nan'), который к тому же истинный, так что «if not text» его
    не ловит
    """
    if not isinstance(text, str) or not text:
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
    """Нормализация, стоп-слова, отсев коротышей и стемминг - в этом порядке

    Стемминг последним: стоп-лист и длина считаются по исходным словам, иначе
    «его» после стемминга перестанет совпадать со стоп-листом. Один и тот же
    токенизатор обязан применяться и к корпусу, и к запросу, поэтому он тут один
    """
    tokens = normalize(text).split()
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if min_length > 1:
        tokens = [token for token in tokens if len(token) >= min_length]
    if do_stem:
        tokens = [stem(token) for token in tokens]
    return tokens


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
