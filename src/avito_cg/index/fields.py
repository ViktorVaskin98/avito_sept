"""Подготовка текстовых полей для индекса

Сырой item_infm_params_text в индекс класть нельзя. Там в среднем 974 символа, и большая
часть это служебные ключи («График работы от», «Тип стоимости за услугу») и числа: время
работы в секундах, минимальные суммы, годы опыта. Смысла в них ноль, а длину поля они
раздувают втрое, и нормировка BM25F начинает штрафовать объявления за то, что владелец
заполнил больше галочек.

Поэтому параметры проходят через парсер, и в индекс идут только значения.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from avito_cg.data.params import ADDRESS_KEY, ParamsParser, build_parser
from avito_cg.data.text import truncate

PARAM_KEYS_FILE = "param_keys.json"
DISCOVERY_SAMPLE = 20_000


def load_parser(
    texts: Sequence[str], cache: Path, *, sample: int = DISCOVERY_SAMPLE, seed: int = 0
) -> ParamsParser:
    """Достать словарь ключей из кэша или построить заново

    Поиск ключей занимает около полуминуты на выборке в 20 тысяч объявлений,
    и результат детерминирован, так что держу его на диске
    """
    if cache.exists():
        return ParamsParser.load(cache)

    started = time.time()
    pool = pd.Series([text for text in texts if text])
    if len(pool) > sample:
        pool = pool.sample(sample, random_state=seed)
    parser = build_parser(pool.tolist())
    parser.save(cache)
    print(f"словарь параметров: {len(parser.keys)} ключей за {time.time() - started:.0f} c")
    return parser


def _numeric(value: str) -> bool:
    return value.replace(".", "", 1).replace(",", "", 1).isdigit()


def params_text(
    parser: ParamsParser, raw: Sequence[str], *, include_address: bool = True
) -> list[str]:
    """Оставить от параметров только осмысленные значения

    Адрес держу по умолчанию: в запросах регулярно встречаются названия городов
    («перевозки владикавказ тбилиси»), и они действительно попадают в адрес исполнителя
    """
    prepared: list[str] = []
    for text in raw:
        parsed = parser.parse(text)
        parts = [
            value
            for key, values in parsed.values.items()
            if include_address or key != ADDRESS_KEY
            for value in values
            if value and not _numeric(value)
        ]
        prepared.append(" ".join(parts))
    return prepared


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
) -> list[str]:
    """Текст объявления для би-энкодера

    Отдельно от item_fields: у энкодера окно 128 токенов, и класть туда всё описание
    бессмысленно, оно просто обрежется. Беру заголовок, фасетные значения и голову описания -
    по EDA первые 250 символов описания дают примерно половину его вклада в лексическое
    покрытие, а дальше идут условия работы и контакты
    """
    titles = items["item_title_raw"].fillna("").astype(str).tolist()
    params = params_text(
        parser,
        items["item_infm_params_text"].fillna("").astype(str).tolist(),
        include_address=False,
    )
    descriptions = items["item_description_raw"].fillna("").astype(str).tolist()
    return [
        ". ".join(part for part in (title, param, truncate(body, description_chars)) if part)
        for title, param, body in zip(titles, params, descriptions, strict=True)
    ]
