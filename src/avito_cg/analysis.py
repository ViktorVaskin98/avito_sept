"""Расчёты для разбора данных

Каждая функция отвечает на один вопрос, от которого зависит архитектура решения,
и возвращает голые числа или DataFrame. Рисование живёт отдельно, в figures.py,
ноутбук только склеивает одно с другим
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

from avito_cg.data.io import QUERY_KEY_COLUMNS, query_key
from avito_cg.data.text import normalize, tokenize, truncate

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Расстояние по поверхности Земли в километрах"""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    inner = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))


def cliffs_delta(
    first: np.ndarray, second: np.ndarray, *, seed: int = 0, cap: int = 200_000
) -> float:
    """Размер эффекта для двух распределений, от -1 до 1

    На полумиллионе наблюдений p-value перестаёт что-либо значить: значимым
    становится любое различие, включая бессмысленное. Поэтому вместе с тестом
    всегда считаю размер эффекта, а решения принимаю по нему
    """
    rng = np.random.default_rng(seed)
    a = rng.choice(first, size=min(cap, first.size), replace=False)
    b = rng.choice(second, size=min(cap, second.size), replace=False)
    statistic = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * statistic / (a.size * b.size) - 1)


def query_item_counts(train: pd.DataFrame) -> pd.Series:
    """Сколько объявлений приходится на уникальный запрос"""
    return query_key(train).value_counts()


def overlap_report(
    train: pd.DataFrame,
    *,
    sample: int = 120_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Лексический разрыв: пересекается ли запрос с текстом выбранного объявления

    Главный вопрос всего решения. Если у заметной доли пар нет ни одного общего
    токена с заголовком, то BM25 по заголовку эти пары не вытащит никогда,
    и плотный поиск нужен не «для красоты», а по необходимости
    """
    rows = train.sample(min(sample, len(train)), random_state=seed)
    queries = [set(tokenize(text)) for text in rows["search_query"].astype(str)]
    fields = {
        "заголовок": rows["item_title_raw"].astype(str),
        "заголовок + параметры": (
            rows["item_title_raw"].astype(str) + " " + rows["item_infm_params_text"].astype(str)
        ),
    }
    if "item_description_raw" in rows.columns:
        fields["заголовок + параметры + описание"] = (
            fields["заголовок + параметры"] + " " + rows["item_description_raw"].astype(str)
        )

    records = []
    for name, column in fields.items():
        coverage = np.empty(len(rows))
        for index, (query_tokens, text) in enumerate(zip(queries, column, strict=True)):
            if not query_tokens:
                coverage[index] = np.nan
                continue
            item_tokens = set(tokenize(text))
            coverage[index] = len(query_tokens & item_tokens) / len(query_tokens)
        records.append(
            {
                "поле": name,
                "нет общих токенов": float(np.nanmean(coverage == 0)),
                "покрыт частично": float(np.nanmean((coverage > 0) & (coverage < 1))),
                "покрыт полностью": float(np.nanmean(coverage == 1)),
                "среднее покрытие": float(np.nanmean(coverage)),
            }
        )
    return pd.DataFrame(records)


def location_centroids(items: pd.DataFrame) -> pd.DataFrame:
    """Координаты локации как медиана координат её объявлений

    Медиана, а не среднее: в координатах встречаются явные выбросы,
    среднее от них уезжает в поле
    """
    grouped = items.groupby("item_location_id")
    return pd.DataFrame(
        {
            "latitude": grouped["item_latitude"].median(),
            "longitude": grouped["item_longitude"].median(),
            "items": grouped.size(),
        }
    )


def search_location_centroids(train: pd.DataFrame) -> pd.DataFrame:
    """Координаты поисковой локации, восстановленные по выбранным в ней объявлениям

    Нужны потому, что у 17.4% запросов бенчмарка в их search_location_id нет ни одного
    объявления корпуса: это родительский узел иерархии, а объявления лежат в дочерних.
    Через координаты иерархия обходится без знания её структуры
    """
    grouped = train.groupby("search_location_id")
    return pd.DataFrame(
        {
            "latitude": grouped["item_latitude"].median(),
            "longitude": grouped["item_longitude"].median(),
            "pairs": grouped.size(),
        }
    )


def distance_report(
    train: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    thresholds: Sequence[float] = (1, 5, 10, 25, 50, 100, 250, 500, 1000),
) -> pd.DataFrame:
    """Сколько выбранных объявлений лежит в пределах X км от центра поисковой локации

    Это потолок полноты для гео-фильтра: если резать кандидатов по расстоянию,
    выше этого числа Recall уже не поднимется
    """
    joined = train.join(centroids, on="search_location_id", rsuffix="_centroid")
    distance = haversine(
        joined["latitude"].to_numpy(dtype=float),
        joined["longitude"].to_numpy(dtype=float),
        joined["item_latitude"].to_numpy(dtype=float),
        joined["item_longitude"].to_numpy(dtype=float),
    )
    distance = distance[~np.isnan(distance)]
    return pd.DataFrame(
        {
            "порог, км": list(thresholds),
            "доля пар внутри": [float((distance <= t).mean()) for t in thresholds],
        }
    )


def location_match_report(train: pd.DataFrame) -> dict[str, float]:
    """Насколько плоское совпадение location_id вообще работает"""
    same = (train["search_location_id"] == train["item_location_id"]).mean()
    per_search = train.groupby("search_location_id")["item_location_id"].nunique()
    return {
        "совпадение id": float(same),
        "разных item_location на search_location, медиана": float(per_search.median()),
        "разных item_location на search_location, p90": float(per_search.quantile(0.9)),
    }


def facet_report(train: pd.DataFrame) -> dict[str, float]:
    """Насколько фильтр поиска жёстко ограничивает параметры объявления

    Дословного совпадения строк ждать не стоит: со стороны запроса ключ называется
    «Тип услуги автосервиса», со стороны объявления «Тип услуги». Поэтому сравниваю
    по токенам
    """
    mask = train["search_infm_params_text"].astype(str).str.len() > 0
    rows = train[mask]
    coverage = np.empty(len(rows))
    # рядом считаю дословное вхождение: оно показывает, почему сравнивать
    # надо по токенам, а не строками
    verbatim = np.zeros(len(rows), dtype=bool)
    for index, (filter_text, item_text) in enumerate(
        zip(
            rows["search_infm_params_text"].astype(str),
            rows["item_infm_params_text"].astype(str),
            strict=True,
        )
    ):
        normalized_filter = normalize(filter_text)
        normalized_item = normalize(item_text)
        filter_tokens = set(normalized_filter.split())
        if not filter_tokens:
            coverage[index] = np.nan
            continue
        coverage[index] = len(filter_tokens & set(normalized_item.split())) / len(filter_tokens)
        verbatim[index] = normalized_filter in normalized_item
    return {
        "пар с непустым фильтром": int(mask.sum()),
        "доля непустых фильтров": float(mask.mean()),
        "среднее покрытие фильтра": float(np.nanmean(coverage)),
        "полное покрытие": float(np.nanmean(coverage == 1)),
        "нулевое покрытие": float(np.nanmean(coverage == 0)),
        "дословное вхождение": float(verbatim.mean()),
    }


def microcat_concentration(train: pd.DataFrame, *, min_repeats: int = 3) -> dict[str, float]:
    """Предсказуема ли микрокатегория по тексту запроса

    Беру запросы, которые встречались хотя бы несколько раз, и смотрю,
    насколько сконцентрировано распределение микрокатегорий выбранных объявлений.
    Если оно узкое, то классификатор «запрос -> микрокатегория» будет работать,
    и это сразу сокращает корпус в сотни раз
    """
    grouped = train.groupby(train["search_query"].astype(str))["item_microcat_id"]
    sizes = grouped.size()
    frequent = sizes[sizes >= min_repeats].index
    subset = train[train["search_query"].astype(str).isin(set(frequent))]

    top_share = []
    entropies = []
    for _, group in subset.groupby(subset["search_query"].astype(str))["item_microcat_id"]:
        counts = group.value_counts(normalize=True).to_numpy()
        top_share.append(counts[0])
        entropies.append(-float((counts * np.log2(counts)).sum()))
    return {
        "запросов в выборке": len(top_share),
        "доля самой частой микрокатегории, среднее": float(np.mean(top_share)),
        "доля самой частой микрокатегории, медиана": float(np.median(top_share)),
        "энтропия распределения, среднее бит": float(np.mean(entropies)),
        "запросов с одной микрокатегорией": float(np.mean(np.array(top_share) == 1.0)),
    }


def selection_bias_report(train: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Отличаются ли выбранные объявления от корпуса по рейтингу, отзывам и цене

    Здесь важна оговорка: пулы разные, train это выбранные объявления за период,
    корпус это срез бенчмарка, так что часть различий объясняется составом пулов,
    а не поведением пользователей. Поэтому смотрю на размер эффекта, а не на p-value
    """
    records = []
    for column, label in (
        ("item_rating", "рейтинг"),
        ("item_rating_reviews_count", "число отзывов"),
        ("item_price", "цена"),
    ):
        chosen = train[column].to_numpy(dtype=float)
        pool = corpus[column].to_numpy(dtype=float)
        chosen = chosen[np.isfinite(chosen)]
        pool = pool[np.isfinite(pool)]
        if chosen.size == 0 or pool.size == 0:
            continue
        records.append(
            {
                "признак": label,
                "медиана у выбранных": float(np.median(chosen)),
                "медиана в корпусе": float(np.median(pool)),
                "дельта Клиффа": cliffs_delta(chosen, pool),
            }
        )
    return pd.DataFrame(records)


def shift_report(train: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Чем запросы бенчмарка отличаются от запросов обучения

    От этого зависит конструкция локальной валидации: если распределения разъехались,
    выборку отложенных запросов придётся перевзвешивать, иначе локальная метрика
    будет мерить не то
    """
    train_queries = train.drop_duplicates(subset=list(QUERY_KEY_COLUMNS))
    train_length = (
        train_queries["search_query"].astype(str).str.split().str.len().to_numpy(dtype=float)
    )
    bench_length = benchmark["search_query"].astype(str).str.split().str.len().to_numpy(dtype=float)
    ks = stats.ks_2samp(train_length, bench_length)

    train_empty = (train_queries["search_infm_params_text"].astype(str).str.len() == 0).mean()
    bench_empty = (benchmark["search_infm_params_text"].astype(str).str.len() == 0).mean()

    return pd.DataFrame(
        [
            {
                "величина": "слов в запросе, среднее",
                "обучение": float(train_length.mean()),
                "бенчмарк": float(bench_length.mean()),
                "примечание": f"KS={ks.statistic:.3f}",
            },
            {
                "величина": "доля пустых фильтров",
                "обучение": float(train_empty),
                "бенчмарк": float(bench_empty),
                "примечание": "перевзвешивать валидацию",
            },
            {
                "величина": "доля поиска с доставкой",
                "обучение": float(train_queries["search_is_delivery_search"].mean()),
                "бенчмарк": float(benchmark["search_is_delivery_search"].mean()),
                "примечание": "в бенчмарке признак вырожден",
            },
        ]
    )


def filter_ceiling_report(
    train: pd.DataFrame,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    centroids: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    model: ComplementNB,
    *,
    radius_km: float = 25.0,
    top_microcats: int = 3,
    chunk: int = 128,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Главная таблица разбора: чем платим за сокращение корпуса

    Слева потолок полноты, посчитанный на парах обучения: доля выбранных объявлений,
    которые фильтр пропускает. Справа число выживших кандидатов, посчитанное на реальном
    корпусе для реальных запросов бенчмарка. Метрика тут Recall@50, так что каждый
    процент потолка это процент, который уже не вернуть никаким ранжированием
    """
    item_lat = corpus["item_latitude"].to_numpy(dtype=float)
    item_lon = corpus["item_longitude"].to_numpy(dtype=float)
    item_microcat = corpus["item_microcat_id"].to_numpy()

    located = queries.join(centroids, on="search_location_id")
    query_lat = located["latitude"].to_numpy(dtype=float)
    query_lon = located["longitude"].to_numpy(dtype=float)

    binary = {"binary": True, "use_idf": False, "norm": None, "token_pattern": r"[0-9a-zа-я]+"}
    facet_vectorizer = TfidfVectorizer(analyzer="word", **binary)
    item_facets = facet_vectorizer.fit_transform(
        normalize(text) for text in corpus["item_infm_params_text"].astype(str)
    )
    query_facets = facet_vectorizer.transform(
        normalize(text) for text in queries["search_infm_params_text"].astype(str)
    )
    item_facets.data[:] = 1.0
    query_facets.data[:] = 1.0
    required = np.asarray(query_facets.sum(axis=1)).ravel()

    predicted = model.classes_[
        np.argsort(
            -model.predict_log_proba(
                vectorizer.transform(
                    normalize(text) for text in queries["search_query"].astype(str)
                )
            ),
            axis=1,
        )[:, :top_microcats]
    ]

    sizes: dict[str, list[int]] = {
        name: [] for name in ("гео", "фасет", "микрокат", "гео+фасет", "всё")
    }
    for start in range(0, len(query_lat), chunk):
        stop = min(start + chunk, len(query_lat))
        distance = haversine(
            query_lat[start:stop, None],
            query_lon[start:stop, None],
            item_lat[None, :],
            item_lon[None, :],
        )
        geo = distance <= radius_km
        hits = (query_facets[start:stop] @ item_facets.T).toarray()
        facet = np.where(
            required[start:stop][:, None] == 0, True, hits >= required[start:stop][:, None]
        )
        microcat = np.stack([np.isin(item_microcat, predicted[row]) for row in range(start, stop)])
        sizes["гео"].extend(geo.sum(axis=1).tolist())
        sizes["фасет"].extend(facet.sum(axis=1).tolist())
        sizes["микрокат"].extend(microcat.sum(axis=1).tolist())
        sizes["гео+фасет"].extend((geo & facet).sum(axis=1).tolist())
        sizes["всё"].extend((geo & facet & microcat).sum(axis=1).tolist())

    train_located = train.join(centroids, on="search_location_id", rsuffix="_centroid")
    train_distance = haversine(
        train_located["latitude"].to_numpy(dtype=float),
        train_located["longitude"].to_numpy(dtype=float),
        train_located["item_latitude"].to_numpy(dtype=float),
        train_located["item_longitude"].to_numpy(dtype=float),
    )
    geo_ok = train_distance <= radius_km
    facet_ok = np.array(
        [
            (not filter_tokens) or filter_tokens.issubset(item_tokens)
            for filter_tokens, item_tokens in zip(
                (set(normalize(t).split()) for t in train["search_infm_params_text"].astype(str)),
                (set(normalize(t).split()) for t in train["item_infm_params_text"].astype(str)),
                strict=True,
            )
        ]
    )
    train_predicted = model.classes_[
        np.argsort(
            -model.predict_log_proba(
                vectorizer.transform(normalize(text) for text in train["search_query"].astype(str))
            ),
            axis=1,
        )[:, :top_microcats]
    ]
    microcat_ok = (train_predicted == train["item_microcat_id"].to_numpy()[:, None]).any(axis=1)

    masks = {
        f"гео, радиус {radius_km:.0f} км": (geo_ok, "гео"),
        "фасетный фильтр": (facet_ok, "фасет"),
        f"топ-{top_microcats} микрокатегорий": (microcat_ok, "микрокат"),
        "гео + фасет": (geo_ok & facet_ok, "гео+фасет"),
        "гео + фасет + микрокатегория": (geo_ok & facet_ok & microcat_ok, "всё"),
    }
    records = [
        {
            "фильтр": "весь корпус",
            "потолок полноты": 1.0,
            "кандидатов, медиана": len(corpus),
            "кандидатов, p90": len(corpus),
        }
    ]
    for label, (mask, key) in masks.items():
        counts = np.array(sizes[key])
        records.append(
            {
                "фильтр": label,
                "потолок полноты": float(mask.mean()),
                "кандидатов, медиана": int(np.median(counts)),
                "кандидатов, p90": int(np.percentile(counts, 90)),
            }
        )
    return pd.DataFrame(records), {key: np.array(value) for key, value in sizes.items()}


def description_truncation_report(
    train: pd.DataFrame,
    *,
    limits: Sequence[int] = (0, 100, 200, 300, 500, 1000, 100_000),
    sample: int = 50_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Сколько описания реально нужно индексировать

    Описание закрывает лексический разрыв лучше всех остальных полей, но медиана
    его длины 916 символов, а p95 почти 3900. Смотрю, где кривая выходит на полку,
    чтобы не тащить в индекс хвосты про условия работы и контакты
    """
    rows = train.sample(min(sample, len(train)), random_state=seed)
    queries = [set(tokenize(text)) for text in rows["search_query"].astype(str)]
    head = (
        rows["item_title_raw"].astype(str) + " " + rows["item_infm_params_text"].astype(str)
    ).tolist()
    descriptions = rows["item_description_raw"].astype(str).tolist()

    records = []
    for limit in limits:
        coverage = np.array(
            [
                len(query_tokens & set(tokenize(f"{prefix} {truncate(body, limit)}")))
                / len(query_tokens)
                if query_tokens
                else np.nan
                for query_tokens, prefix, body in zip(queries, head, descriptions, strict=True)
            ]
        )
        records.append(
            {
                "символов описания": limit,
                "нет общих токенов": float(np.nanmean(coverage == 0)),
                "покрыт полностью": float(np.nanmean(coverage == 1)),
                "среднее покрытие": float(np.nanmean(coverage)),
            }
        )
    return pd.DataFrame(records)


def microcat_classifier_report(
    train: pd.DataFrame,
    *,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    seed: int = 0,
) -> tuple[pd.DataFrame, TfidfVectorizer, ComplementNB]:
    """Насколько хорошо микрокатегория предсказывается по одному тексту запроса

    Символьные n-граммы вместо слов: запросы короткие и с опечатками, а наивный байес
    поверх них учится за полминуты на всех 500 тысячах пар. Это оценка снизу,
    нормальная модель будет лучше, но для потолка по кандидатам её достаточно
    """
    pairs = train[["search_query", "item_microcat_id"]].copy()
    pairs["search_query"] = pairs["search_query"].astype(str).map(normalize)
    pairs = pairs[pairs["search_query"].str.len() > 0]

    rng = np.random.default_rng(seed)
    unique_queries = pairs["search_query"].unique()
    holdout = set(rng.choice(unique_queries, size=int(0.2 * len(unique_queries)), replace=False))
    is_test = pairs["search_query"].isin(holdout)
    fit, test = pairs[~is_test], pairs[is_test]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True
    )
    features = vectorizer.fit_transform(fit["search_query"])
    model = ComplementNB(alpha=0.2)
    model.fit(features, fit["item_microcat_id"].to_numpy())

    scores = model.predict_log_proba(vectorizer.transform(test["search_query"]))
    order = np.argsort(-scores, axis=1)
    truth = test["item_microcat_id"].to_numpy()
    report = pd.DataFrame(
        {
            "k": list(ks),
            "точность top-k": [
                float((model.classes_[order[:, :k]] == truth[:, None]).any(axis=1).mean())
                for k in ks
            ],
        }
    )
    return report, vectorizer, model


def duplicate_report(corpus: pd.DataFrame) -> dict[str, float]:
    """Сколько в корпусе повторяющихся заголовков

    Услуги часто размножают: одна компания публикует одно и то же объявление
    в разных локациях. Если такие дубли займут слоты в топ-50, это потерянная полнота
    """
    titles = [normalize(text) for text in corpus["item_title_raw"].astype(str)]
    counts = Counter(titles)
    repeated = np.array([count for count in counts.values() if count > 1])
    frame = pd.DataFrame(
        {
            "title": titles,
            "location": corpus["item_location_id"].to_numpy(),
            "microcat": corpus["item_microcat_id"].to_numpy(),
        }
    )
    identifiable = {
        label: float((frame.groupby(keys).size() == 1).sum() / len(frame))
        for keys, label in (
            (["title"], "по заголовку"),
            (["title", "location"], "по заголовку и локации"),
            (["title", "location", "microcat"], "по заголовку, локации и микрокатегории"),
        )
    }
    return {
        "уникальных заголовков": len(counts),
        "объявлений": len(titles),
        "доля объявлений с неуникальным заголовком": float(
            sum(count for count in counts.values() if count > 1) / len(titles)
        ),
        "максимальный тираж заголовка": int(repeated.max()) if repeated.size else 0,
        **{f"однозначно опознаётся {key}": value for key, value in identifiable.items()},
    }
