"""Переранжирование кандидатов градиентным бустингом

Зачем он нужен, стало понятно из разбора промахов, а не из общих соображений. Два оставшихся
класса потерь описываются одной фразой: вес сигнала должен зависеть от силы другого сигнала.

Далёкие, но текстуально точные объявления теряются, потому что гео давит их даже при идеальном
совпадении текста: за 25 км доля попаданий падает с 0.92 до 0.46. Простое ослабление штрафа
проверено и не работает, оно ухудшает метрику монотонно, потому что вместе с далёкими
правильными в ответ лезут далёкие неправильные. Нужна условная поправка «ослабь гео, если текст
совпал отлично», а взвешенная сумма такого не выражает ни при каких весах.

**На чём обучать.** Отложенных запросов всего 2 452, и на них же считается метрика, поэтому
оценка идёт кросс-валидацией по фолдам, разбитым по запросам. Первая версия только так
и работала, и по приросту было видно, что модель недоучена: из доступных по потолку 0.041
она забрала 0.008 на трёх тысячах позитивов. Поэтому обучающие запросы теперь берутся
из пула-донора, см. extra.py: их не видел ни один сигнал, а меток там в девять раз больше.

Признаки намеренно двух сортов. Скоры сигналов отвечают за взаимодействия, ради которых всё
затевалось. Свойства самого объявления - популярность, цена, отзывы - отвечают за приор,
и они опасны: локальный корпус пришлось специально строить с долей знакомых обучению
объявлений 9.6%, как в настоящем, иначе любой признак популярности выглядел бы сильнее,
чем он есть на самом деле.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from avito_cg.analysis import haversine
from avito_cg.config import RANDOM_SEED
from avito_cg.data.text import normalize
from avito_cg.index.lexical import BM25FIndex, FieldPresence

FEATURES = [
    "лексика",
    "лексика_норм",
    "лексика_ранг",
    "гео",
    "гео_ранг",
    "расстояние",
    "расстояние_ранг",
    "фасеты",
    "микрокатегория",
    "плотный",
    "плотный_ранг",
    "итоговый_ранг",
    "слов_в_запросе",
    "пустой_фильтр",
    "кандидатов",
    "покрытие_заголовка",
    "покрытие_idf",
    "покрытие_описания",
    "та_же_локация",
    "популярность",
    "знакомое",
    "память",
    "близнецов",
    "цена",
    "цена_ранг",
    "цена_к_медиане",
    "рейтинг",
    "отзывов",
    # второй раунд признаков. Первый показал, что выигрыш идёт именно отсюда,
    # а не от размера обучающей выборки, поэтому добавляю ещё
    "лексика_заголовок",
    "лексика_описание",
    "длина_заголовка",
    "есть_описание",
    "запрос_в_заголовке",
    "телефон_скрыт",
    "сообщения_запрещены",
]


def features_fingerprint(
    fields: Sequence[tuple[str, float, float]],
    weights: Mapping[str, float],
    *,
    k1: float,
    mode: str,
    depth: int,
    dense_depth: int,
) -> str:
    """Отпечаток всего, от чего зависит таблица признаков

    Сбор кандидатов это семь минут на две с половиной тысячи запросов, поэтому
    таблица кэшируется. Кэш без привязки к конфигурации опаснее, чем отсутствие кэша:
    поменял вес поля, команда отработала за минуту и выдала ровно тот же результат,
    что и до правки, потому что признаки приехали из файла, собранного под старые веса.
    Молча. Поэтому отпечаток кладётся внутрь файла и сверяется при чтении
    """
    payload = {
        "fields": [list(field) for field in fields],
        "k1": k1,
        "weights": sorted(weights.items()),
        "mode": mode,
        "depth": depth,
        "dense_depth": dense_depth,
        "features": list(FEATURES),
    }
    digest = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.blake2b(digest.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True, slots=True)
class ItemContext:
    """Свойства корпуса, которые считаются один раз и переиспользуются всеми запросами"""

    item_ids: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    location: np.ndarray
    price: np.ndarray
    rating: np.ndarray
    reviews: np.ndarray
    twins: np.ndarray
    popularity: np.ndarray
    phone_hidden: np.ndarray
    messages_forbidden: np.ndarray
    normalized_title: list[str]
    title: FieldPresence
    description: FieldPresence
    idf: np.ndarray
    memory: Mapping[str, np.ndarray]

    @classmethod
    def build(
        cls, corpus: pd.DataFrame, index: BM25FIndex, pairs: pd.DataFrame | None
    ) -> ItemContext:
        """Собрать контекст по корпусу и разрешённым парам

        pairs это та часть обучающих пар, которой пользоваться можно. Из неё берутся
        популярность объявления и память «что уже выбирали под этот же текст запроса».
        Без pairs оба признака выключаются нулями
        """
        item_ids = corpus["item_id"].astype(str).to_numpy()
        titles = corpus["item_title_raw"].fillna("").astype(str).to_numpy()
        normalized = [normalize(text) for text in titles]
        counts = pd.Series(normalized).value_counts()
        twins = np.array([counts.get(text, 1) for text in normalized], dtype=np.float64)

        position = {item: number for number, item in enumerate(item_ids)}
        popularity = np.zeros(len(item_ids), dtype=np.float64)
        memory: dict[str, np.ndarray] = {}
        if pairs is not None:
            chosen = pairs["item_id"].astype(str).to_numpy()
            for item, count in pd.Series(chosen).value_counts().items():
                number = position.get(str(item))
                if number is not None:
                    popularity[number] = count

            # счётчик делю на средний счётчик знакомых объявлений корпуса. Иначе признак
            # не переносится: локально разрешено 213 449 пар, на настоящих данных 497 673,
            # и одно и то же по смыслу объявление получило бы там вдвое больший счётчик.
            # Доля знакомых у обоих корпусов одна и та же, 9.6%, а вот масштаб счётчика нет
            known = popularity[popularity > 0]
            if known.size:
                popularity = popularity / known.mean()

            # память кликлога: под каким текстом запроса какое объявление уже выбирали.
            # На настоящем бенчмарке точное совпадение текста есть у 37% запросов,
            # локально у 64.6%, так что признак локально выглядит сильнее, чем он есть
            grouped: dict[str, set[int]] = {}
            for text, item in zip(
                (normalize(value) for value in pairs["search_query"].astype(str)),
                chosen,
                strict=True,
            ):
                number = position.get(str(item))
                if number is not None:
                    grouped.setdefault(text, set()).add(number)
            memory = {text: np.array(sorted(numbers)) for text, numbers in grouped.items()}

        return cls(
            item_ids=item_ids,
            latitude=corpus["item_latitude"].to_numpy(dtype=float),
            longitude=corpus["item_longitude"].to_numpy(dtype=float),
            location=corpus["item_location_id"].to_numpy(dtype=np.int64),
            price=corpus["item_price"].to_numpy(dtype=float),
            rating=corpus["item_rating"].to_numpy(dtype=float),
            reviews=corpus["item_rating_reviews_count"].to_numpy(dtype=float),
            twins=twins,
            popularity=popularity,
            phone_hidden=corpus["item_is_phone_hidden"].fillna(False).to_numpy(dtype=float),
            messages_forbidden=corpus["item_is_message_forbidden"]
            .fillna(False)
            .to_numpy(dtype=float),
            normalized_title=normalized,
            title=index.presence("title"),
            description=index.presence("description"),
            idf=index.idf,
            memory=memory,
        )


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Кандидаты всех запросов вместе с признаками и метками"""

    query: np.ndarray
    item: np.ndarray
    features: np.ndarray
    label: np.ndarray


def build_features(
    candidates: Sequence[np.ndarray],
    lexical: Sequence[np.ndarray],
    signal_scores: Mapping[str, Sequence[np.ndarray]],
    queries: pd.DataFrame,
    context: ItemContext,
    index: BM25FIndex,
    *,
    query_latitude: np.ndarray,
    query_longitude: np.ndarray,
    query_ids: Sequence[str],
    relevant: dict[str, set[str]] | None = None,
) -> CandidateSet:
    """Собрать таблицу признаков по кандидатам всех запросов

    Ранги кладу рядом со скорами намеренно: дерево на скоре учит порог в абсолютных
    единицах, который между запросами не переносится, а ранг переносится
    """
    query_text = queries["search_query"].fillna("").astype(str).to_numpy()
    query_location = queries["search_location_id"].to_numpy(dtype=np.int64)
    empty_filter = (
        queries["search_infm_params_text"].fillna("").astype(str).str.len() == 0
    ).to_numpy()

    rows, labels, query_index, item_index = [], [], [], []
    for position, query_id in enumerate(query_ids):
        items = candidates[position]
        if items.size == 0:
            continue
        terms = index.query_terms(query_text[position])

        # обратный список отсортирован, и двоичный поиск по нему требует отсортированных
        # кандидатов, а порядок кандидатов менять нельзя: он задаёт строки таблицы
        order = np.argsort(items)
        sorted_items = items[order]

        distance = haversine(
            np.full(items.size, query_latitude[position]),
            np.full(items.size, query_longitude[position]),
            context.latitude[items],
            context.longitude[items],
        )
        distance = np.nan_to_num(distance, nan=-1.0, posinf=-1.0)

        coverage = np.empty(items.size)
        coverage_idf = np.empty(items.size)
        coverage_description = np.empty(items.size)
        coverage[order] = context.title.coverage(terms, sorted_items, None)
        coverage_idf[order] = context.title.coverage(terms, sorted_items, context.idf[terms])
        coverage_description[order] = context.description.coverage(terms, sorted_items, None)

        remembered = np.zeros(items.size)
        normalized_query = normalize(query_text[position])
        known = context.memory.get(normalized_query)
        if known is not None and known.size:
            remembered = np.isin(items, known).astype(float)

        title_score = np.empty(items.size)
        description_score = np.empty(items.size)
        title_score[order] = context.title.score(terms, context.idf[terms], sorted_items)
        description_score[order] = context.description.score(
            terms, context.idf[terms], sorted_items
        )
        verbatim = np.array(
            [float(normalized_query in context.normalized_title[item]) for item in items]
        )

        price = context.price[items]
        median_price = np.nanmedian(price) if np.isfinite(price).any() else np.nan

        lexical_scores = lexical[position]
        geo_scores = signal_scores["гео"][position]
        dense_scores = signal_scores["плотный"][position]
        block = np.column_stack(
            [
                lexical_scores,
                lexical_scores / max(lexical_scores.max(), 1e-9),
                _ranks(lexical_scores),
                geo_scores,
                _ranks(geo_scores),
                distance,
                _ranks(-distance),
                signal_scores["фасеты"][position],
                signal_scores["микрокатегория"][position],
                dense_scores,
                _ranks(dense_scores),
                _ranks(signal_scores["итог"][position]),
                np.full(items.size, len(query_text[position].split())),
                np.full(items.size, float(empty_filter[position])),
                np.full(items.size, float(items.size)),
                coverage,
                coverage_idf,
                coverage_description,
                (context.location[items] == query_location[position]).astype(float),
                np.log1p(context.popularity[items]),
                (context.popularity[items] > 0).astype(float),
                remembered,
                context.twins[items],
                price,
                _ranks(-np.nan_to_num(price, nan=np.inf)),
                price / median_price if median_price else np.zeros(items.size),
                context.rating[items],
                context.reviews[items],
                title_score,
                description_score,
                context.title.length[items],
                (context.description.length[items] > 0).astype(float),
                verbatim,
                context.phone_hidden[items],
                context.messages_forbidden[items],
            ]
        )
        rows.append(block.astype(np.float32))
        truth = (relevant or {}).get(query_id, set())
        labels.append(np.array([context.item_ids[item] in truth for item in items], dtype=int))
        query_index.append(np.full(items.size, position))
        item_index.append(items)

    return CandidateSet(
        query=np.concatenate(query_index),
        item=np.concatenate(item_index),
        # float32, а не float64: на донорском наборе это 3.4 млн строк на 26 признаков,
        # и разница между 350 и 700 МБ на слабой машине заметна, а точности хватает
        features=np.vstack(rows),
        label=np.concatenate(labels),
    )


def ranked_items(
    data: CandidateSet,
    scores: np.ndarray | None,
    n_queries: int,
    *,
    top_k: int,
) -> np.ndarray:
    """Матрица «запрос на top_k» с номерами строк корпуса, -1 на пустых местах

    scores=None означает «оставить порядок строк как есть»: collect кладёт кандидатов
    запроса подряд и уже по убыванию скора слияния, так что это выдача до переранжирования.
    Одна функция на все места, где нужен топ внутри запроса - раньше их было четыре,
    и каждая могла разъехаться со своей копией
    """
    order = np.full((n_queries, top_k), -1, dtype=np.int64)
    filled = np.zeros(n_queries, dtype=np.int64)
    rows = range(len(data.query)) if scores is None else np.lexsort((-scores, data.query))
    for row in rows:
        query = int(data.query[row])
        if filled[query] < top_k:
            order[query, filled[query]] = data.item[row]
            filled[query] += 1
    return order


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.empty(scores.size, dtype=np.float64)
    order[np.argsort(-scores)] = np.arange(scores.size)
    return order


def subset(data: CandidateSet, mask: np.ndarray) -> CandidateSet:
    return CandidateSet(
        query=data.query[mask],
        item=data.item[mask],
        features=data.features[mask],
        label=data.label[mask],
    )


def cross_validated_scores(
    data: CandidateSet,
    n_queries: int,
    *,
    folds: int = 5,
    seed: int = RANDOM_SEED,
    **kwargs: Any,
) -> np.ndarray:
    """Скор реранкера для каждого кандидата, полученный моделью, его не видевшей

    Разбиение по запросам, а не по строкам: иначе кандидаты одного запроса попали бы
    и в обучение, и в контроль, и оценка была бы завышена
    """
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, folds, size=n_queries)
    scores = np.zeros(len(data.label))

    for fold in range(folds):
        test_queries = np.flatnonzero(assignment == fold)
        test_mask = np.isin(data.query, test_queries)
        if test_mask.sum() == 0 or (~test_mask).sum() == 0:
            continue
        model = fit(subset(data, ~test_mask), seed=seed, **kwargs)
        scores[test_mask] = predict(model, data.features[test_mask])
    return scores


def fit(
    data: CandidateSet,
    *,
    iterations: int = 400,
    depth: int = 6,
    seed: int = RANDOM_SEED,
    objective: str = "logloss",
    features: Sequence[str] = tuple(FEATURES),
) -> CatBoostClassifier | CatBoostRanker:
    """Обучить переранжировщик

    Двоичная классификация оптимизирует калибровку вероятности, а нужен порядок внутри
    запроса, поэтому рядом лежит ранжирующая постановка с группировкой по запросу.
    Что из этого лучше на таких данных, рассуждением не решается, поэтому меряю обе
    """
    columns = _columns(features)
    if objective == "logloss":
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=0.05,
            loss_function="Logloss",
            # позитивов около половины процента от строк, без веса модель просто
            # научится всем отвечать «нет»
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(Pool(data.features[:, columns], data.label, feature_names=list(features)))
        return model

    # YetiRank требует, чтобы строки одной группы шли подряд
    order = np.argsort(data.query, kind="stable")
    model = CatBoostRanker(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="YetiRank",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(
            data.features[order][:, columns],
            data.label[order],
            group_id=data.query[order],
            feature_names=list(features),
        )
    )
    return model


def _columns(features: Sequence[str]) -> np.ndarray:
    return np.array([FEATURES.index(name) for name in features], dtype=int)


def predict(model: CatBoostClassifier | CatBoostRanker, features: np.ndarray) -> np.ndarray:
    columns = _columns(model.feature_names_)
    if isinstance(model, CatBoostRanker):
        return model.predict(features[:, columns])
    return model.predict_proba(features[:, columns])[:, 1]


def importance(model: CatBoostClassifier | CatBoostRanker) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {"признак": model.feature_names_, "важность": model.get_feature_importance().round(2)}
        )
        .sort_values("важность", ascending=False)
        .reset_index(drop=True)
    )
