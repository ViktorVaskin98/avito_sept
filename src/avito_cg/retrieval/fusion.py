"""Слияние лексического скора с остальными сигналами

Разбор промахов на шаге с BM25F показал, где именно теряется метрика. Из 2 913 релевантных
объявлений 1 479 находятся текстом, но стоят ниже пятидесятого места, и ещё 395 не попадают
даже в топ-1000. Причина у обеих групп одна: под короткий родовой запрос вроде «кран»
или «грузчики» подходят сотни объявлений с почти одинаковым скором, а 48.6% корпуса
вдобавок делит заголовок с другими. Текст внутри такой группы не различает ничего.

Отсюда два требования к конструкции.

Первое: остальные сигналы подмешиваются ко **всем** ненулевым позициям разреженной строки
BM25, а не к усечённому топу. Схема «взять топ-1000 и переранжировать» отрезала бы
вторую группу до всякого ранжирования, и вернуть её было бы уже нечем.

Второе: масштабы надо согласовать. Скор BM25 зависит от длины запроса и редкости термов,
поэтому один и тот же вес сигнала означает разный вклад для разных запросов. Какой способ
согласования лучше, рассуждением не решить, поэтому их тут три и все измеряются.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from avito_cg.analysis import haversine
from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import BM25FIndex, top_k_from_row
from avito_cg.retrieval.signals import Signal

Mode = Literal["raw", "normalized", "rrf"]
RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Как складывать лексику с остальными сигналами

    raw        - сырой скор BM25 плюс взвешенные сигналы. Масштаб BM25 гуляет по запросам,
                 так что вклад сигналов получается больше на коротких запросах. Это как раз
                 там, где он и нужен, но управлять этим нельзя
    normalized - скор BM25, поделённый на максимум внутри запроса. Масштаб фиксирован,
                 вес означает одно и то же везде
    rrf        - обратные ранги. Масштабов нет вовсе, но и величина разрыва между первым
                 и вторым местом теряется, а для расстояния она решает
    """

    mode: Mode = "normalized"
    geo_weight: float = 0.10
    rrf_k: int = RRF_K


DEFAULT_CONFIG = FusionConfig()


def _lexical_component(lexical: np.ndarray, config: FusionConfig) -> np.ndarray:
    if config.mode == "raw":
        return lexical
    if config.mode == "normalized":
        top = lexical.max() if lexical.size else 1.0
        return lexical / (top or 1.0)
    ranks = np.empty(lexical.size, dtype=np.float64)
    ranks[np.argsort(-lexical)] = np.arange(lexical.size)
    return 1 / (config.rrf_k + ranks)


def _signal_component(values: np.ndarray, weight: float, config: FusionConfig) -> np.ndarray:
    if config.mode != "rrf":
        return weight * values
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[np.argsort(-values)] = np.arange(values.size)
    return weight / (config.rrf_k + ranks)


def retrieve(
    index: BM25FIndex,
    signals: Sequence[tuple[Signal, float]],
    texts: Sequence[str],
    *,
    top_k: int = 50,
    config: FusionConfig = DEFAULT_CONFIG,
    chunk: int = 128,
    padding: GeoIndex | None = None,
    query_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    extra_candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Индексы строк корпуса, по убыванию итогового скора, -1 на пустых местах

    extra_candidates добавляет к лексическим кандидатам ещё по строке индексов на запрос.
    Это нужно плотному поиску: как сигнал он умеет только переупорядочивать найденное
    лексикой, а 3.0% пар по разбору данных не имеют с объявлением ни одного общего токена
    и в лексическое множество не попадают вовсе. Достать их можно, только если плотный
    поиск ещё и порождает кандидатов
    """
    result = np.full((len(texts), top_k), -1, dtype=np.int64)

    for start, block in index.iter_scores(texts, chunk=chunk):
        for row in range(block.shape[0]):
            query = start + row
            begin, end = block.indptr[row], block.indptr[row + 1]
            candidates = block.indices[begin:end]
            lexical = block.data[begin:end].astype(np.float64)

            if extra_candidates is not None:
                extra = extra_candidates[query]
                extra = extra[extra >= 0]
                # у пришедших только из плотного поиска лексического скора нет,
                # и это честный ноль: текст запроса с ними действительно не пересёкся
                fresh = np.setdiff1d(extra, candidates, assume_unique=False)
                candidates = np.concatenate([candidates, fresh])
                lexical = np.concatenate([lexical, np.zeros(fresh.size)])

            if candidates.size == 0:
                continue

            combined = _lexical_component(lexical, config)
            for signal, weight in signals:
                if weight:
                    combined = combined + _signal_component(
                        signal.score(query, candidates), weight, config
                    )
            chosen, _ = top_k_from_row(candidates, combined, top_k)
            result[query, : chosen.size] = chosen

    if padding is not None and query_coordinates is not None:
        _pad_with_nearest(result, padding, *query_coordinates, top_k)
    return result


def _pad_with_nearest(
    result: np.ndarray,
    geo: GeoIndex,
    latitude: np.ndarray,
    longitude: np.ndarray,
    top_k: int,
) -> None:
    """Добить неполные ответы ближайшими по расстоянию

    Метрика не штрафует за лишних кандидатов, поэтому пустое место в ответе это чистая
    потеря. После слияния таких запросов почти не остаётся, вклад в метрику нулевой,
    но как защита от краевого случая пусть будет
    """
    incomplete = np.flatnonzero((result < 0).any(axis=1))
    for query in incomplete:
        if np.isnan(latitude[query]):
            continue
        filled = result[query][result[query] >= 0]
        missing = top_k - filled.size
        distance = haversine(
            np.full(geo.item_latitude.size, latitude[query]),
            np.full(geo.item_longitude.size, longitude[query]),
            geo.item_latitude,
            geo.item_longitude,
        )
        distance[filled] = np.inf
        extra = np.argpartition(distance, missing)[:missing]
        result[query, filled.size :] = extra[np.argsort(distance[extra])]
