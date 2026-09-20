"""Подготовка текстовых полей для индекса

Сырой item_infm_params_text в индекс класть нельзя. Там в среднем 974 символа, и большая
часть это служебные ключи («График работы от», «Тип стоимости за услугу») и числа: время
работы в секундах, минимальные суммы, годы опыта. Смысла в них ноль, а длину поля они
раздувают втрое, и нормировка BM25F начинает штрафовать объявления за то, что владелец
заполнил больше галочек.

Поэтому параметры проходят через парсер, и в индекс идут только значения.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from avito_cg.data.params import ADDRESS_KEY, ParamsParser, build_parser
from avito_cg.data.text import truncate

DISCOVERY_SAMPLE = 20_000

# Словарь ключей закреплён в пакете, а не собирается на лету в каждой команде, и это
# осознанное решение, а не кэш ради скорости.
#
# Поиск ключей детерминирован только при фиксированном входе, а вход у разных команд
# разный: make-answer видит benchmark_items, baseline - локальный корпус из train,
# encode-corpus - то один, то другой. На этих трёх источниках словарь получается
# на 99, 92 и 96 ключей соответственно, и расходятся они на три десятка ключей.
# Пока файл лежал в data/interim, его содержимое зависело от того, какую команду
# запустили первой, а вместе с ним - поле params индекса, скоры BM25F и итоговый
# answer.csv. То есть повторный прогон с чистого листа давал другой ответ.
#
# Поэтому словарь версионируется вместе с кодом: он такой же артефакт решения,
# как веса полей. Пересобрать его можно командой `avito-cg param-keys`, она пишет
# файл на место этого, и тогда индексы надо пересобрать тоже
PINNED_KEYS = Path(__file__).resolve().parent.parent / "data" / "param_keys.json"


def load_parser() -> ParamsParser:
    """Закреплённый словарь ключей параметров

    Аргументов нет намеренно: словарь не выводится из корпуса, который сейчас
    в руках, а читается из пакета. Раньше он выводился, и это ломало
    воспроизводимость - почему, написано выше
    """
    return ParamsParser.load(PINNED_KEYS)


def discover_parser(
    texts: Sequence[str], *, sample: int = DISCOVERY_SAMPLE, seed: int = 0
) -> ParamsParser:
    """Заново вывести словарь ключей из текстов параметров

    Занимает около полуминуты на выборке в 20 тысяч объявлений. Результат
    детерминирован при фиксированных texts, sample и seed
    """
    started = time.time()
    pool = pd.Series([text for text in texts if text])
    if len(pool) > sample:
        pool = pool.sample(sample, random_state=seed)
    parser = build_parser(pool.tolist())
    print(f"словарь параметров: {len(parser.keys)} ключей за {time.time() - started:.0f} c")
    return parser


_TIME = re.compile(r"^\d{1,2}[:.]\d{2}$")


def _numeric(value: str) -> bool:
    """Числовые и временные значения смысла не несут

    Время проверяю отдельно: «09:30» не ловится проверкой на цифры, а в параметрах
    его много, это график работы и время для связи
    """
    if _TIME.match(value):
        return True
    return value.replace(".", "", 1).replace(",", "", 1).isdigit()


def params_values(
    parser: ParamsParser, raw: Sequence[str], *, include_address: bool = True
) -> list[list[str]]:
    """Осмысленные значения параметров, по списку на объявление

    Адрес держу по умолчанию: в запросах регулярно встречаются названия городов
    («перевозки владикавказ тбилиси»), и они действительно попадают в адрес исполнителя
    """
    return [
        [
            value
            for key, values in parser.parse(text).values.items()
            if include_address or key != ADDRESS_KEY
            for value in values
            if value and not _numeric(value)
        ]
        for text in raw
    ]


def frequent_values(values: Sequence[Sequence[str]], *, max_share: float = 0.2) -> set[str]:
    """Значения параметров, которые встречаются у слишком многих объявлений

    «Начальная цена» или «Тип стоимости за услугу» стоят почти везде и не различают
    ничего, а место в окне энкодера занимают. Отбираю их по доле объявлений, а не
    чёрным списком: список пришлось бы вести руками и он развалился бы на новых данных.
    По сути это тот же idf, только на уровне значений, а не токенов
    """
    counter: Counter[str] = Counter()
    for row in values:
        counter.update(set(row))
    limit = max_share * len(values)
    return {value for value, count in counter.items() if count > limit}


def params_text(
    parser: ParamsParser, raw: Sequence[str], *, include_address: bool = True
) -> list[str]:
    """То же самое, но склеенное в строку: так его ждёт лексический индекс"""
    return [" ".join(row) for row in params_values(parser, raw, include_address=include_address)]


def item_fields(
    items: pd.DataFrame,
    parser: ParamsParser,
    *,
    description_chars: int | None = None,
    include_address: bool = True,
) -> dict[str, list[str]]:
    """Три поля документа в том виде, в каком их видит индекс"""
    descriptions = items["item_description_raw"].fillna("").astype(str).tolist()
    if description_chars is not None:
        descriptions = [truncate(text, description_chars) for text in descriptions]
    return {
        "title": items["item_title_raw"].fillna("").astype(str).tolist(),
        "params": params_text(
            parser,
            items["item_infm_params_text"].fillna("").astype(str).tolist(),
            include_address=include_address,
        ),
        "description": descriptions,
    }


def query_texts(queries: pd.DataFrame, *, with_filter: bool = False) -> list[str]:
    """Текст запроса, при желании вместе с текстом фильтра

    Фильтр поиска это тот же текст, что лежит в параметрах объявления, поэтому его
    можно просто дописать к запросу и получить фасетный матчинг бесплатно, силами того же
    BM25. Работает это или мешает, решаю замером, а не рассуждением: фильтр бывает длиннее
    самого запроса и тогда размывает его
    """
    text = queries["search_query"].fillna("").astype(str)
    if with_filter:
        text = text + " " + queries["search_infm_params_text"].fillna("").astype(str)
    return text.tolist()


ENCODER_DESCRIPTION_CHARS = 250


def encoder_texts(
    items: pd.DataFrame,
    parser: ParamsParser,
    *,
    description_chars: int = ENCODER_DESCRIPTION_CHARS,
    max_value_share: float = 0.2,
    common: set[str] | None = None,
) -> list[str]:
    """Текст объявления для би-энкодера

    Отдельно от item_fields по двум причинам. Первая: у энкодера окно 128 токенов,
    и класть туда всё описание бессмысленно, оно просто обрежется. Беру заголовок,
    значения параметров и голову описания - по EDA первые 250 символов описания дают
    примерно половину его вклада в лексическое покрытие, а дальше идут условия работы
    и контакты.

    Вторая: у BM25 частые термы сами получают низкий вес через idf, а у энкодера такого
    механизма нет, каждый токен занимает место в окне на равных. Поэтому значения,
    которые стоят у слишком многих объявлений, отсюда выбрасываются

    common передаётся снаружи там, где items это пары, а не корпус: в парах популярное
    объявление лежит десятки раз, и доля объявлений со значением считается неправильно.
    Одно и то же объявление обязано давать один и тот же текст при обучении и при
    кодировании корпуса, иначе энкодер учится на одном распределении, а применяется
    к другому
    """
    titles = items["item_title_raw"].fillna("").astype(str).tolist()
    values = params_values(
        parser,
        items["item_infm_params_text"].fillna("").astype(str).tolist(),
        include_address=False,
    )
    if common is None:
        common = frequent_values(values, max_share=max_value_share)
    descriptions = items["item_description_raw"].fillna("").astype(str).tolist()

    return [
        ". ".join(
            part
            for part in (
                title,
                " ".join(value for value in row if value not in common),
                truncate(body, description_chars),
            )
            if part
        )
        for title, row, body in zip(titles, values, descriptions, strict=True)
    ]
