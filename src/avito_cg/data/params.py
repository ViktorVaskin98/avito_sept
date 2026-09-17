"""Разбор item_infm_params_text в структуру «ключ - значения»

Параметры объявления приходят одной слипшейся строкой, в среднем 974 символа:

    Вид услуги Автосервис, аренда Тип услуги Аренда авто Место оказания услуг
    Санкт-Петербург, Невский пр-т, 190 Тип транспорта Водный транспорт

Разделителей нет вообще. Значения бывают многословными и с запятыми внутри
(«Красота, здоровье»), ключи повторяются («Рабочие дни Понедельник Рабочие дни Вторник»),
часть ключей это булевы флаги без значения («Предоплата», «Выезд», «Гарантия на работу»),
а иногда значение пустое («Вид услуги Место оказания услуг ...»).

Захардкодить список ключей я не стал: он свой для каждой микрокатегории, их 752,
руками это не собрать и оно развалится на новых данных. Вместо этого ищу ключи
непараметрически, через энтропию ветвления (branching entropy, Jin & Tanaka-Ishii).
Идея простая: настоящий ключ стоит на границе двух сущностей, поэтому слева от него
стоят концы самых разных значений, а справа начинаются самые разные значения, то есть
у него высокая энтропия и левого, и правого контекста. У куска внутри значения контекст
предсказуемый: «оказания услуг» слева почти всегда «Место», «Место оказания» справа
почти всегда «услуг», и оба отсеиваются.

На выходе порядок ключей сохраняю: он в данных стабильный и сам по себе признак.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_KEY_TOKENS = 5
MIN_DOCUMENT_FREQUENCY = 0.004
MIN_LEFT_ENTROPY = 2.0
MIN_RIGHT_ENTROPY = 1.2
MAX_NEIGHBOUR_SHARE = 0.7
MAX_LEFT_COMMA_SHARE = 0.4

_BOUNDARY = "\x02"
_DIGITS = frozenset("0123456789")


def _tokenize(text: str) -> list[str]:
    return text.split()


def _starts_key(token: str) -> bool:
    """Ключи всегда начинаются с заглавной буквы, это режет кандидатов примерно втрое"""
    return bool(token) and token[0].isupper()


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counter.values() if count > 0
    )


def _dominant_share(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return counter.most_common(1)[0][1] / total if total else 1.0


def _left_is_boundary(counter: Counter[str], min_share: float = 0.5) -> bool:
    """Ключ стоит в начале строки чаще, чем в середине

    Отдельный случай, потому что начало строки это настоящая граница, а не
    однообразный сосед. Без этой проверки «Вид услуги», который открывает
    99% строк, вылетает из словаря как «слева всегда одно и то же»
    """
    total = sum(counter.values())
    return bool(total) and counter.get(_BOUNDARY, 0) / total >= min_share


def _looks_like_key(gram: tuple[str, ...]) -> bool:
    """Отсев того, что ключом быть не может

    Цифры это всегда значение (часы, суммы, опыт работы). Запятая в конце бывает
    только у кусков адреса: настоящие ключи вроде «Время работы, с» держат запятую
    внутри, а не на хвосте
    """
    if gram[-1].endswith(","):
        return False
    return not any(set(token) & _DIGITS for token in gram)


def discover_keys(
    texts: Iterable[str],
    *,
    max_tokens: int = MAX_KEY_TOKENS,
    min_document_frequency: float = MIN_DOCUMENT_FREQUENCY,
    min_left_entropy: float = MIN_LEFT_ENTROPY,
    min_right_entropy: float = MIN_RIGHT_ENTROPY,
    max_neighbour_share: float = MAX_NEIGHBOUR_SHARE,
    max_left_comma_share: float = MAX_LEFT_COMMA_SHARE,
) -> list[str]:
    """Найти словарь ключей по энтропии ветвления

    Два прохода: сначала считаю частоты кандидатов и отбрасываю редкие, потом для
    выживших собираю распределения левого и правого соседа. Поверх энтропии стоят
    три фильтра, без которых в словарь лезут куски адресов и значений:

    - доля самого частого соседа: у настоящего ключа нет доминирующего продолжения,
      а у «Место оказания» справа почти всегда «услуг»
    - доля левых соседей с запятой на конце: так отсеиваются названия улиц и городов,
      которые сидят внутри адреса и всегда идут после запятой
    - минимальность: ключ не может содержать внутри себя другой ключ, иначе
      в словарь попадают склейки вида «Марка авто Audi Марка авто»
    """
    documents = [_tokenize(text) for text in texts if text]
    if not documents:
        return []
    min_count = max(5, int(min_document_frequency * len(documents)))

    document_frequency: Counter[tuple[str, ...]] = Counter()
    for tokens in documents:
        seen: set[tuple[str, ...]] = set()
        for start, token in enumerate(tokens):
            if not _starts_key(token):
                continue
            for size in range(1, max_tokens + 1):
                if start + size > len(tokens):
                    break
                seen.add(tuple(tokens[start : start + size]))
        document_frequency.update(seen)

    candidates = {
        gram
        for gram, count in document_frequency.items()
        if count >= min_count and _looks_like_key(gram)
    }
    if not candidates:
        return []

    left: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    right: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    left_comma: Counter[tuple[str, ...]] = Counter()
    total: Counter[tuple[str, ...]] = Counter()
    for tokens in documents:
        length = len(tokens)
        for start, token in enumerate(tokens):
            if not _starts_key(token):
                continue
            for size in range(1, max_tokens + 1):
                end = start + size
                if end > length:
                    break
                gram = tuple(tokens[start:end])
                if gram not in candidates:
                    continue
                previous = tokens[start - 1] if start > 0 else _BOUNDARY
                left[gram][previous] += 1
                right[gram][tokens[end] if end < length else _BOUNDARY] += 1
                total[gram] += 1
                if previous.endswith(","):
                    left_comma[gram] += 1

    def left_ok(gram: tuple[str, ...]) -> bool:
        if _left_is_boundary(left[gram]):
            return True
        return (
            _entropy(left[gram]) >= min_left_entropy
            and _dominant_share(left[gram]) <= max_neighbour_share
        )

    accepted = [
        gram
        for gram in candidates
        if left_ok(gram)
        and _entropy(right[gram]) >= min_right_entropy
        and _dominant_share(right[gram]) <= max_neighbour_share
        and left_comma[gram] / total[gram] <= max_left_comma_share
    ]

    # минимальность: иду от коротких к длинным и не беру кандидата,
    # внутри которого уже лежит принятый ключ
    accepted.sort(key=lambda gram: (len(gram), -document_frequency[gram]))
    minimal: set[tuple[str, ...]] = set()
    for gram in accepted:
        size = len(gram)
        if any(
            gram[start : start + inner] in minimal
            for inner in range(1, size)
            for start in range(size - inner + 1)
        ):
            continue
        minimal.add(gram)

    keys = sorted(minimal, key=lambda gram: (-len(gram), " ".join(gram)))
    return [" ".join(gram) for gram in keys]


ADDRESS_KEY = "Место оказания услуг"


def build_parser(
    texts: Sequence[str],
    *,
    address_key: str = ADDRESS_KEY,
    **kwargs: float | int,
) -> ParamsParser:
    """Собрать парсер в два прохода

    После первого прохода в словарь всё ещё лезут куски адресов («Карла», «Северная»):
    внутри адреса контекст такой же разнообразный, как на границе ключей, и энтропия
    их не отличает. Поэтому вторым проходом я вырезаю значение ключа «Место оказания услуг»
    и ищу ключи заново уже по тексту без адресов
    """
    first = discover_keys(texts, **kwargs)  # type: ignore[arg-type]
    if address_key not in first:
        return ParamsParser(first)

    rough = ParamsParser(first)
    masked = []
    for text in texts:
        parsed = rough.parse(text)
        address = parsed.get(address_key)
        if not address:
            masked.append(text)
            continue
        for value in address:
            if value:
                text = text.replace(value, " ", 1)
        masked.append(text)

    second = discover_keys(masked, **kwargs)  # type: ignore[arg-type]
    return ParamsParser(sorted({*second, address_key}, key=lambda key: (-len(key.split()), key)))


@dataclass(frozen=True, slots=True)
class ParsedParams:
    """Разобранные параметры одного объявления

    values: ключ -> список значений, пустая строка означает флаг без значения
    order: ключи в том порядке, в каком они встретились
    """

    values: dict[str, list[str]]
    order: tuple[str, ...]

    def get(self, key: str) -> list[str]:
        return self.values.get(key, [])

    def first(self, key: str) -> str:
        found = self.values.get(key)
        return found[0] if found else ""

    def flat_text(self, keys: Sequence[str] | None = None) -> str:
        """Склеить значения выбранных ключей, для подачи в текстовый индекс"""
        selected = keys if keys is not None else self.order
        parts = [value for key in selected for value in self.values.get(key, []) if value]
        return " ".join(parts)


class ParamsParser:
    """Жадный разбор по самому длинному совпадению из словаря ключей"""

    def __init__(self, keys: Sequence[str]) -> None:
        self._by_length: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        for key in keys:
            gram = tuple(key.split())
            self._by_length[len(gram)].add(gram)
        self._lengths = sorted(self._by_length, reverse=True)
        self.keys = tuple(keys)

    def _match(self, tokens: Sequence[str], position: int) -> tuple[str, ...] | None:
        for size in self._lengths:
            end = position + size
            if end > len(tokens):
                continue
            gram = tuple(tokens[position:end])
            if gram in self._by_length[size]:
                return gram
        return None

    def parse(self, text: str | None) -> ParsedParams:
        if not text:
            return ParsedParams({}, ())
        tokens = _tokenize(text)
        values: dict[str, list[str]] = {}
        order: list[str] = []
        current_key: str | None = None
        buffer: list[str] = []
        position = 0

        def flush() -> None:
            if current_key is None:
                return
            values.setdefault(current_key, []).append(" ".join(buffer))
            if current_key not in order:
                order.append(current_key)

        while position < len(tokens):
            matched = self._match(tokens, position)
            if matched is not None:
                flush()
                current_key = " ".join(matched)
                buffer = []
                position += len(matched)
                continue
            buffer.append(tokens[position])
            position += 1
        flush()
        return ParsedParams(values, tuple(order))

    def coverage(self, text: str | None) -> float:
        """Доля токенов, попавших под какой-либо ключ или его значение

        Нужна для проверки качества словаря: если словарь дырявый, начало строки
        уедет в «ничей» префикс до первого ключа
        """
        if not text:
            return 1.0
        tokens = _tokenize(text)
        if not tokens:
            return 1.0
        position = 0
        first_key_at = len(tokens)
        while position < len(tokens):
            matched = self._match(tokens, position)
            if matched is not None:
                first_key_at = position
                break
            position += 1
        return 1.0 - first_key_at / len(tokens)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.keys), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ParamsParser:
        return cls(json.loads(path.read_text(encoding="utf-8")))
