"""География: расстояние вместо совпадения идентификатора локации

Из разбора данных: в 83.1% пар локация поиска совпадает с локацией объявления, но у 17.4%
запросов бенчмарка в их search_location_id нет ни одного объявления корпуса. location_id
иерархический, часть запросов приходит с родительского узла, и плоское сравнение там ломается.

Структуру иерархии нам не дали, зато есть координаты объявлений. Центр поисковой локации
восстанавливаю как медиану координат объявлений, выбранных в ней по обучающим парам.
Медиана, а не среднее: в координатах попадаются выбросы, среднее от них уезжает в поле.

Скор считаю не подобранной вручную экспонентой, а логарифмом отношения правдоподобий.
Из обучающих пар известно распределение расстояний до выбранного объявления, из корпуса
распределение до случайного. Отношение этих плотностей и есть то, насколько расстояние
сдвигает вероятность релевантности, и подбирать в нём нечего.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from avito_cg.analysis import haversine

BIN_EDGES = np.array(
    [0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 25, 40, 60, 100, 200, 500, 1000, 3000, np.inf]
)
SMOOTHING = 20.0


@dataclass(frozen=True, slots=True)
class GeoIndex:
    """Центры поисковых локаций, координаты корпуса и калиброванный скор расстояния"""

    latitude: dict[int, float]
    longitude: dict[int, float]
    item_latitude: np.ndarray
    item_longitude: np.ndarray
    log_ratio: np.ndarray

    @classmethod
    def fit(
        cls,
        pairs: pd.DataFrame,
        corpus: pd.DataFrame,
        *,
        sample: int = 200_000,
        seed: int = 0,
    ) -> GeoIndex:
        grouped = pairs.groupby("search_location_id")
        centroid_latitude = grouped["item_latitude"].median()
        centroid_longitude = grouped["item_longitude"].median()

        item_latitude = corpus["item_latitude"].to_numpy(dtype=float)
        item_longitude = corpus["item_longitude"].to_numpy(dtype=float)

        positive = cls._pair_distances(pairs, centroid_latitude, centroid_longitude)
        background = cls._background_distances(
            pairs,
            centroid_latitude,
            centroid_longitude,
            item_latitude,
            item_longitude,
            sample,
            seed,
        )
        return cls(
            latitude=centroid_latitude.to_dict(),
            longitude=centroid_longitude.to_dict(),
            item_latitude=item_latitude,
            item_longitude=item_longitude,
            log_ratio=cls._log_ratio(positive, background),
        )

    @staticmethod
    def _pair_distances(
        pairs: pd.DataFrame, latitude: pd.Series, longitude: pd.Series
    ) -> np.ndarray:
        locations = pairs["search_location_id"].to_numpy()
        distance = haversine(
            latitude.reindex(locations).to_numpy(dtype=float),
            longitude.reindex(locations).to_numpy(dtype=float),
            pairs["item_latitude"].to_numpy(dtype=float),
            pairs["item_longitude"].to_numpy(dtype=float),
        )
        return distance[np.isfinite(distance)]

    @staticmethod
    def _background_distances(
        pairs: pd.DataFrame,
        latitude: pd.Series,
        longitude: pd.Series,
        item_latitude: np.ndarray,
        item_longitude: np.ndarray,
        sample: int,
        seed: int,
    ) -> np.ndarray:
        """Расстояния до случайных объявлений корпуса

        Фон считаю по тем же поисковым локациям, что и позитивы, иначе отношение
        будет мерить не влияние расстояния, а разницу в составе локаций
        """
        rng = np.random.default_rng(seed)
        locations = pairs["search_location_id"].to_numpy()
        if len(locations) > sample:
            locations = rng.choice(locations, size=sample, replace=False)
        picked = rng.integers(0, len(item_latitude), size=len(locations))
        distance = haversine(
            latitude.reindex(locations).to_numpy(dtype=float),
            longitude.reindex(locations).to_numpy(dtype=float),
            item_latitude[picked],
            item_longitude[picked],
        )
        return distance[np.isfinite(distance)]

    @staticmethod
    def _log_ratio(positive: np.ndarray, background: np.ndarray) -> np.ndarray:
        """log P(расстояние | релевантно) - log P(расстояние | случайное объявление)

        Сглаживание нужно на дальних корзинах: туда попадает по десятку позитивов,
        и без него отношение там скачет на порядок от пары наблюдений
        """
        positive_counts, _ = np.histogram(positive, bins=BIN_EDGES)
        background_counts, _ = np.histogram(background, bins=BIN_EDGES)
        positive_rate = (positive_counts + SMOOTHING) / (
            positive_counts.sum() + SMOOTHING * len(positive_counts)
        )
        background_rate = (background_counts + SMOOTHING) / (
            background_counts.sum() + SMOOTHING * len(background_counts)
        )
        return np.log(positive_rate / background_rate)

    def query_coordinates(self, search_location_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latitude = np.array([self.latitude.get(int(key), np.nan) for key in search_location_ids])
        longitude = np.array([self.longitude.get(int(key), np.nan) for key in search_location_ids])
        return latitude, longitude

    def distance(self, search_location_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Расстояния от центра поисковой локации до каждого кандидата

        candidates это матрица индексов строк корпуса, -1 означает пустое место
        """
        latitude, longitude = self.query_coordinates(search_location_ids)
        safe = np.where(candidates >= 0, candidates, 0)
        distance = haversine(
            latitude[:, None],
            longitude[:, None],
            self.item_latitude[safe],
            self.item_longitude[safe],
        )
        return np.where(candidates >= 0, distance, np.inf)

    def score(self, distance: np.ndarray) -> np.ndarray:
        """Логарифм отношения правдоподобий для каждого расстояния

        У запросов без восстановленного центра расстояние получается nan,
        и скор для них нулевой: гео про такой запрос ничего не знает
        """
        bins = np.digitize(np.nan_to_num(distance, nan=np.inf), BIN_EDGES) - 1
        bins = np.clip(bins, 0, len(self.log_ratio) - 1)
        return np.where(np.isnan(distance), 0.0, self.log_ratio[bins])
