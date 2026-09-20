"""Сигналы для слияния

Каждый сигнал отвечает на один вопрос про пару «запрос - объявление» и возвращает
логарифм отношения правдоподобий: насколько наблюдаемое значение признака чаще
встречается у релевантных объявлений, чем у случайных. Выгода двойная.

Во-первых, все сигналы оказываются в одних единицах, и складывать их осмысленно,
а не на глазок с подобранным коэффициентом. Во-вторых, калибровка снимается с данных:
для расстояния не надо выбирать вид убывающей функции, для фасетов не надо решать,
сколько стоит совпадение на три четверти.

Веса при сложении всё равно остаются, потому что сигналы не независимы: близкое
объявление чаще оказывается и подходящим по фасетам. Наивный байес это игнорирует,
вес отчасти компенсирует.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

from avito_cg.analysis import haversine
from avito_cg.data.text import normalize
from avito_cg.index.geo import GeoIndex

SMOOTHING = 20.0
FACET_BINS = np.array([0.0, 0.25, 0.5, 0.75, 0.999, 1.01])


class Signal(Protocol):
    """Скор для кандидатов одного запроса

    name объявлен свойством, а не полем: сигналы это замороженные dataclass,
    у них атрибут только на чтение, и изменяемому полю протокола они не подходят
    """

    @property
    def name(self) -> str: ...

    def score(self, query: int, candidates: np.ndarray) -> np.ndarray: ...


def log_ratio(positive: np.ndarray, background: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """log P(значение | релевантно) - log P(значение | случайное объявление)

    Сглаживание нужно на редких корзинах: там бывает по десятку наблюдений,
    и без него отношение скачет на порядок от пары штук
    """
    positive_counts, _ = np.histogram(positive, bins=bins)
    background_counts, _ = np.histogram(background, bins=bins)
    positive_rate = (positive_counts + SMOOTHING) / (
        positive_counts.sum() + SMOOTHING * len(positive_counts)
    )
    background_rate = (background_counts + SMOOTHING) / (
        background_counts.sum() + SMOOTHING * len(background_counts)
    )
    return np.log(positive_rate / background_rate)


@dataclass(frozen=True, slots=True)
class GeoSignal:
    """Расстояние от центра поисковой локации до объявления"""

    geo: GeoIndex
    latitude: np.ndarray
    longitude: np.ndarray
    name: str = "гео"

    @classmethod
    def build(cls, geo: GeoIndex, queries: pd.DataFrame) -> GeoSignal:
        latitude, longitude = geo.query_coordinates(queries["search_location_id"].to_numpy())
        return cls(geo=geo, latitude=latitude, longitude=longitude)

    def score(self, query: int, candidates: np.ndarray) -> np.ndarray:
        if np.isnan(self.latitude[query]):
            return np.zeros(candidates.size)
        distance = haversine(
            np.full(candidates.size, self.latitude[query]),
            np.full(candidates.size, self.longitude[query]),
            self.geo.item_latitude[candidates],
            self.geo.item_longitude[candidates],
        )
        return self.geo.score(distance)


@dataclass(frozen=True, slots=True)
class FacetSignal:
    """Насколько параметры объявления покрывают фильтр поиска

    Из разбора данных: у 92% пар покрытие полное, среднее 0.971, нулевое почти не встречается.
    Как термы BM25 фасеты вредят (родовая лексика без различающей силы), а отдельным
    калиброванным признаком должны работать
    """

    masks: dict[int, np.ndarray]
    query_columns: list[np.ndarray]
    calibration: np.ndarray
    name: str = "фасеты"

    @classmethod
    def build(
        cls,
        queries: pd.DataFrame,
        corpus: pd.DataFrame,
        pairs: pd.DataFrame,
        *,
        sample: int = 100_000,
        seed: int = 0,
    ) -> FacetSignal:
        """Построить маски токенов фильтра по корпусу и откалибровать покрытие

        Хранится не матрица совпадений «запрос на объявление» - плотная она весила бы
        почти гигабайт, - а булева маска по каждому токену фасетов, который вообще
        встречается в запросах. Таких токенов сотни, а не сотни тысяч
        """
        vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=r"[0-9a-zа-я]+", binary=True, use_idf=False, norm=None
        )
        items = vectorizer.fit_transform(
            normalize(text) for text in corpus["item_infm_params_text"].astype(str)
        )
        items.data[:] = 1.0

        query_matrix = vectorizer.transform(
            normalize(text) for text in queries["search_infm_params_text"].astype(str)
        )
        query_matrix.data[:] = 1.0
        columns = [
            query_matrix.indices[query_matrix.indptr[row] : query_matrix.indptr[row + 1]]
            for row in range(query_matrix.shape[0])
        ]

        # плотная матрица совпадений «запрос на объявление» это 2452 на 189212, почти гигабайт.
        # вместо неё держу булеву маску по каждому токену фасетов, который вообще встречается
        # в запросах: таких токенов сотни, а не сотни тысяч, и выходит десятки мегабайт
        by_column = items.tocsc()
        used = {int(column) for row in columns for column in row}
        masks = {column: np.zeros(items.shape[0], dtype=bool) for column in used}
        for column in used:
            masks[column][by_column[:, column].indices] = True

        positive, background = cls._calibration_samples(
            vectorizer, pairs, items, sample=sample, seed=seed
        )
        return cls(
            masks=masks,
            query_columns=columns,
            calibration=log_ratio(positive, background, FACET_BINS),
        )

    @staticmethod
    def _calibration_samples(
        vectorizer: TfidfVectorizer,
        pairs: pd.DataFrame,
        items: sp.csr_matrix,
        *,
        sample: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Покрытие у выбранных объявлений против покрытия у случайных

        Фон беру по тем же фильтрам, что и позитивы: иначе отношение будет мерить
        не влияние совпадения, а разницу в составе запросов
        """
        rng = np.random.default_rng(seed)
        rows = pairs if len(pairs) <= sample else pairs.sample(sample, random_state=seed)
        filters = vectorizer.transform(
            normalize(text) for text in rows["search_infm_params_text"].astype(str)
        )
        filters.data[:] = 1.0
        chosen = vectorizer.transform(
            normalize(text) for text in rows["item_infm_params_text"].astype(str)
        )
        chosen.data[:] = 1.0

        needed = np.asarray(filters.sum(axis=1)).ravel()
        keep = needed > 0
        positive = np.asarray(filters.multiply(chosen).sum(axis=1)).ravel()[keep] / needed[keep]

        random_rows = rng.integers(0, items.shape[0], size=int(keep.sum()))
        background = (
            np.asarray(filters[keep].multiply(items[random_rows]).sum(axis=1)).ravel()
            / needed[keep]
        )
        return positive, background

    def score(self, query: int, candidates: np.ndarray) -> np.ndarray:
        columns = self.query_columns[query]
        if columns.size == 0:
            return np.zeros(candidates.size)
        matched = np.zeros(candidates.size)
        for column in columns:
            matched += self.masks[int(column)][candidates]
        coverage = matched / columns.size
        bins = np.clip(np.digitize(coverage, FACET_BINS) - 1, 0, len(self.calibration) - 1)
        return self.calibration[bins]


@dataclass(frozen=True, slots=True)
class MicrocatSignal:
    """Насколько микрокатегория объявления подходит тексту запроса

    Классификатор запроса даёт log P(микрокатегория | запрос), из него вычитается
    log P(микрокатегория) по корпусу. Получается тот же логарифм отношения: насколько
    запрос сдвигает вероятность категории относительно её базовой частоты
    """

    query_log_ratio: np.ndarray
    item_class: np.ndarray
    unknown_penalty: float
    name: str = "микрокатегория"

    @classmethod
    def build(
        cls, queries: pd.DataFrame, corpus: pd.DataFrame, pairs: pd.DataFrame
    ) -> MicrocatSignal:
        """Обучить классификатор микрокатегории по тексту запроса и снять приор

        Символьные n-граммы, а не слова: запросы короткие и с опечатками, а разница
        между «ремонт» и «капремонт» существенная. Из логарифма вероятности класса
        вычитается логарифм его частоты в корпусе, и получается то же отношение
        правдоподобий, что у остальных сигналов
        """
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=300_000,
            sublinear_tf=True,
        )
        features = vectorizer.fit_transform(
            normalize(text) for text in pairs["search_query"].astype(str)
        )
        model = ComplementNB(alpha=0.2)
        model.fit(features, pairs["item_microcat_id"].to_numpy())

        query_log_proba = model.predict_log_proba(
            vectorizer.transform(normalize(text) for text in queries["search_query"].astype(str))
        )
        microcats = corpus["item_microcat_id"].to_numpy()
        counts = pd.Series(microcats).value_counts()
        prior = np.array(
            [np.log(counts.get(label, 1) / len(microcats)) for label in model.classes_]
        )

        position = {label: index for index, label in enumerate(model.classes_)}
        item_class = np.array([position.get(label, -1) for label in microcats])
        ratio = query_log_proba - prior
        return cls(
            query_log_ratio=ratio,
            item_class=item_class,
            # незнакомых классификатору микрокатегорий в корпусе 0.9% по EDA,
            # так что опускать их вниз почти бесплатно, но и наказывать не за что
            unknown_penalty=float(np.percentile(ratio, 5)),
        )

    def score(self, query: int, candidates: np.ndarray) -> np.ndarray:
        classes = self.item_class[candidates]
        known = classes >= 0
        result = np.full(candidates.size, self.unknown_penalty)
        result[known] = self.query_log_ratio[query, classes[known]]
        return result


def load_embeddings(path: Path, item_ids: np.ndarray) -> np.ndarray:
    """Прочитать эмбеддинги и убедиться, что порядок строк совпадает с корпусом

    Без этой проверки перепутанный порядок не проявится никак: скоры останутся
    правдоподобными, метрика просто молча просядет, и искать причину придётся долго
    """
    vectors = np.load(path)
    saved = np.load(path.with_name(path.stem + "_item_ids.npy"), allow_pickle=True).astype(str)
    expected = item_ids.astype(str)
    if len(saved) != len(expected):
        raise ValueError(
            f"в {path.name} {len(saved)} строк, а в корпусе {len(expected)} объявлений"
        )
    if not np.array_equal(saved, expected):
        mismatched = int((saved != expected).sum())
        raise ValueError(
            f"порядок строк в {path.name} не совпадает с корпусом: расходится {mismatched} "
            f"из {len(expected)}, первое расхождение на позиции "
            f"{int(np.flatnonzero(saved != expected)[0])}"
        )
    return vectors


@dataclass(frozen=True, slots=True)
class DenseSignal:
    """Косинусная близость запроса и объявления по би-энкодеру

    В отличие от BM25 косинус нормированных векторов уже сравним между запросами:
    он лежит в [-1, 1] и не зависит ни от длины запроса, ни от редкости термов.
    Поэтому здесь, в отличие от остальных сигналов, калибровать нечего, хватает веса
    """

    query_vectors: np.ndarray
    item_vectors: np.ndarray
    name: str = "плотный"

    def score(self, query: int, candidates: np.ndarray) -> np.ndarray:
        return self.item_vectors[candidates].astype(np.float32) @ self.query_vectors[query].astype(
            np.float32
        )

    def top_candidates(self, top_k: int = 200, chunk: int = 64) -> np.ndarray:
        """Лучшие кандидаты по косинусу для каждого запроса, по всему корпусу

        Нужно потому, что как сигнал плотный поиск умеет только переупорядочивать
        найденное лексикой, а 3.0% пар по разбору данных не имеют с объявлением
        ни одного общего токена и в лексическое множество не попадают вовсе.

        Приближённый поиск не нужен: 2452 на 189212 на 768 это меньше терафлопа,
        BLAS считает это за минуту. Матрицу скоров держу блоками, целиком она заняла бы
        почти два гигабайта
        """
        items = self.item_vectors.astype(np.float32)
        result = np.full((len(self.query_vectors), top_k), -1, dtype=np.int64)
        for start in range(0, len(self.query_vectors), chunk):
            stop = min(start + chunk, len(self.query_vectors))
            scores = self.query_vectors[start:stop].astype(np.float32) @ items.T
            take = min(top_k, scores.shape[1])
            top = np.argpartition(-scores, take - 1, axis=1)[:, :take]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            result[start:stop, :take] = np.take_along_axis(top, order, axis=1)
        return result


def describe(signals: Sequence[tuple[Signal, float]]) -> str:
    return ", ".join(f"{signal.name}={weight}" for signal, weight in signals)
