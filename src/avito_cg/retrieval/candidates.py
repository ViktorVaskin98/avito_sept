"""Сбор кандидатов вместе с разложением скора по сигналам

Отдельно от retrieve, потому что нужно другое. retrieve отдаёт готовую выдачу и складывает
сигналы в один скор, а переранжировщику нужна каждая составляющая по отдельности: он учится
именно на том, как они соотносятся между собой.

Считать их дважды незачем, поэтому здесь один проход, и результат годится и для того,
чтобы просто взять топ, и для того, чтобы построить таблицу признаков.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from avito_cg.index.lexical import BM25FIndex
from avito_cg.retrieval.fusion import (
    DEFAULT_CONFIG,
    FusionConfig,
    lexical_component,
    signal_component,
)
from avito_cg.retrieval.signals import Signal

FINAL = "итог"


@dataclass(frozen=True, slots=True)
class Candidates:
    """Кандидаты каждого запроса со скорами каждого сигнала

    Списки по запросам, а не матрица: число кандидатов у запросов разное, от нуля
    до десятков тысяч, и прямоугольная матрица под это не годится
    """

    items: list[np.ndarray]
    lexical: list[np.ndarray]
    scores: dict[str, list[np.ndarray]]

    def order(self, top_k: int) -> np.ndarray:
        """Выдача в том же формате, что у retrieve"""
        result = np.full((len(self.items), top_k), -1, dtype=np.int64)
        for query, items in enumerate(self.items):
            take = min(top_k, items.size)
            result[query, :take] = items[:take]
        return result


def collect(
    index: BM25FIndex,
    signals: Sequence[tuple[Signal, float]],
    texts: Sequence[str],
    *,
    depth: int = 200,
    config: FusionConfig = DEFAULT_CONFIG,
    chunk: int = 128,
    extra_candidates: np.ndarray | None = None,
) -> Candidates:
    """Топ-depth кандидатов на запрос вместе со всеми составляющими скора"""
    names = [signal.name for signal, _ in signals]
    items: list[np.ndarray] = [np.array([], dtype=np.int64)] * len(texts)
    lexical: list[np.ndarray] = [np.array([])] * len(texts)
    scores: dict[str, list[np.ndarray]] = {
        name: [np.array([])] * len(texts) for name in (*names, FINAL)
    }

    for start, block in index.iter_scores(texts, chunk=chunk):
        for row in range(block.shape[0]):
            query = start + row
            begin, end = block.indptr[row], block.indptr[row + 1]
            candidates = block.indices[begin:end]
            values = block.data[begin:end].astype(np.float64)

            if extra_candidates is not None:
                extra = extra_candidates[query]
                fresh = np.setdiff1d(extra[extra >= 0], candidates)
                candidates = np.concatenate([candidates, fresh])
                values = np.concatenate([values, np.zeros(fresh.size)])
            if candidates.size == 0:
                continue

            total = lexical_component(values, config)
            per_signal = {}
            for signal, weight in signals:
                # сохраняю сырой скор сигнала: он идёт признаком в переранжировщик.
                # В сумму он складывается ровно тем же преобразованием, что и в retrieve,
                # иначе пул кандидатов и выдача считались бы по разным формулам
                per_signal[signal.name] = signal.score(query, candidates)
                total = total + signal_component(per_signal[signal.name], weight, config)

            keep = np.argsort(-total)[:depth]
            items[query] = candidates[keep]
            lexical[query] = values[keep]
            for name in names:
                scores[name][query] = per_signal[name][keep]
            scores[FINAL][query] = total[keep]

    return Candidates(items=items, lexical=lexical, scores=scores)
