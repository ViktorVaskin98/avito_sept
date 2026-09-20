"""Лексический поиск: BM25F

Обычный BM25 по склейке всех полей здесь работает плохо, и видно почему. Длины полей
отличаются на два порядка: заголовок это примерно 4 токена, параметры 140, описание 130.
В склейке нормировка по длине считается по сумме, и вклад заголовка растворяется,
хотя именно заголовок чаще всего и отвечает на запрос.

BM25F решает ровно эту задачу: каждое поле нормируется на свою среднюю длину, взвешенные
частоты складываются, и только потом применяется насыщение. То есть терм, встреченный
в заголовке, и терм, встреченный в трёхтысячном описании, попадают в общую сумму
с разным весом, а не с одинаковым.

Формулы:

    x[f,d,t] = tf[f,d,t] / (1 - b[f] + b[f] * len[f,d] / avglen[f])
    W[d,t]   = sum_f boost[f] * x[f,d,t]
    score    = sum_{t in q} idf[t] * W[d,t] / (k1 + W[d,t])

Насыщение применяется один раз к общему весу, а не отдельно по полям. Если применить
его к каждому полю, получится не BM25F, а сумма независимых BM25, и повторы терма
в разных полях будут засчитываться дважды.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from avito_cg.data.text import tokenize

K1 = 1.2


@dataclass(frozen=True, slots=True)
class Field:
    """Одно поле документа

    boost это вклад поля в общий вес терма, b это сила нормировки по длине:
    при b=0 длина поля игнорируется, при b=1 учитывается полностью
    """

    name: str
    boost: float = 1.0
    b: float = 0.75


# веса подобраны на локальном бенчмарке, подробности в docs/EXPERIMENTS.md.
# вес заголовка взят 20, а не найденные сеткой 100: метрика на отрезке от 8 до 100
# растёт всего на 0.008 при её собственной ошибке 0.01, то есть плоская, и брать
# argmax значит подгоняться под шум. 20 это внутренняя точка этого плато.
# b подобраны каждый со своим внутренним максимумом, там подгонки нет
DEFAULT_FIELDS: tuple[Field, ...] = (
    Field("title", boost=20.0, b=0.6),
    Field("params", boost=0.5, b=0.75),
    Field("description", boost=1.0, b=0.75),
)


def _build_vocabulary(
    token_lists: Sequence[Sequence[Sequence[str]]], min_df: int
) -> dict[str, int]:
    """Общий словарь по всем полям

    Словарь обязан быть общим, иначе матрицы полей не сложить: один и тот же терм
    должен стоять в одном и том же столбце
    """
    document_frequency: dict[str, int] = {}
    for field_tokens in token_lists:
        for tokens in field_tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: index
        for index, token in enumerate(
            sorted(token for token, count in document_frequency.items() if count >= min_df)
        )
    }


def _counts(
    field_tokens: Sequence[Sequence[str]], vocabulary: Mapping[str, int]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Матрица частот термов и длины документов по этому полю

    Длину считаю по всем токенам, включая те, что не попали в словарь: иначе
    нормировка соврёт в пользу документов с редкими словами
    """
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    lengths = np.zeros(len(field_tokens), dtype=np.float64)

    for row, tokens in enumerate(field_tokens):
        lengths[row] = len(tokens)
        counts: dict[int, int] = {}
        for token in tokens:
            column = vocabulary.get(token)
            if column is not None:
                counts[column] = counts.get(column, 0) + 1
        indices.extend(counts)
        data.extend(counts.values())
        indptr.append(len(indices))

    matrix = sp.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr)),
        shape=(len(field_tokens), len(vocabulary)),
    )
    return matrix, lengths


def top_k_from_row(
    indices: np.ndarray, scores: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Топ-k по убыванию скора из ненулевых элементов одной строки

    Плотная матрица 2452 на 189212 это 1.85 ГБ, а нулевые скоры в топ всё равно
    не попадут, так что работаю прямо с ненулевыми
    """
    if scores.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    take = min(top_k, scores.size)
    top = np.argpartition(-scores, take - 1)[:take]
    top = top[np.argsort(-scores[top])]
    return indices[top], scores[top]


class BM25FIndex:
    """Индекс BM25F по нескольким полям с общим словарём"""

    def __init__(
        self,
        fields: Sequence[Field] = DEFAULT_FIELDS,
        *,
        k1: float = K1,
        min_df: int = 1,
        tokenizer: Callable[[str | None], list[str]] = tokenize,
    ) -> None:
        self.fields = tuple(fields)
        self.k1 = k1
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.vocabulary: dict[str, int] = {}
        self.idf = np.zeros(0)
        self._matrix: sp.csr_matrix | None = None
        self._field_counts: dict[str, sp.csr_matrix] = {}
        self._field_lengths: dict[str, np.ndarray] = {}

    @property
    def n_documents(self) -> int:
        return 0 if self._matrix is None else self._matrix.shape[0]

    def fit(self, documents: Mapping[str, Sequence[str]]) -> BM25FIndex:
        """Построить индекс по словарю «поле -> тексты документов»

        Токенизация тут самая дорогая часть, поэтому делаю её ровно один раз
        и держу результат, чтобы перевзвешивание полей не требовало пересчёта
        """
        names = [field.name for field in self.fields]
        missing = [name for name in names if name not in documents]
        if missing:
            raise KeyError(f"нет текстов для полей: {missing}")

        sizes = {len(documents[name]) for name in names}
        if len(sizes) != 1:
            raise ValueError(f"поля разной длины: {sizes}")

        tokenized = {}
        for name in names:
            started = time.time()
            tokenized[name] = [self.tokenizer(text) for text in documents[name]]
            print(f"  токенизация {name}: {time.time() - started:.0f} c", flush=True)

        self.vocabulary = _build_vocabulary(list(tokenized.values()), self.min_df)
        print(f"  словарь: {len(self.vocabulary)} термов", flush=True)

        for name in names:
            counts, lengths = _counts(tokenized[name], self.vocabulary)
            self._field_counts[name] = counts
            self._field_lengths[name] = lengths

        self._compute_idf()
        self.reweight()
        return self

    def save(self, directory: Path) -> None:
        """Сохранить построенный индекс

        Токенизация корпуса это несколько минут, а всё остальное секунды, так что
        разумно один раз переварить тексты и дальше работать с матрицами
        """
        directory.mkdir(parents=True, exist_ok=True)
        for name, counts in self._field_counts.items():
            sp.save_npz(directory / f"counts_{name}.npz", counts)
        np.savez(
            directory / "vectors.npz",
            idf=self.idf,
            **{f"length_{name}": values for name, values in self._field_lengths.items()},
        )
        (directory / "vocabulary.json").write_text(
            json.dumps(self.vocabulary, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        fields: Sequence[Field] = DEFAULT_FIELDS,
        *,
        k1: float = K1,
        tokenizer: Callable[[str | None], list[str]] = tokenize,
    ) -> BM25FIndex:
        index = cls(fields, k1=k1, tokenizer=tokenizer)
        index.vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npz")
        index.idf = vectors["idf"]
        for field in fields:
            index._field_counts[field.name] = sp.load_npz(directory / f"counts_{field.name}.npz")
            index._field_lengths[field.name] = vectors[f"length_{field.name}"]
        index.reweight()
        return index

    @staticmethod
    def exists(directory: Path, fields: Sequence[Field] = DEFAULT_FIELDS) -> bool:
        needed = [directory / "vocabulary.json", directory / "vectors.npz"]
        needed += [directory / f"counts_{field.name}.npz" for field in fields]
        return all(path.exists() for path in needed)

    def _compute_idf(self) -> None:
        """Документная частота считается по объединению полей

        Терм, стоящий и в заголовке, и в описании одного объявления, встречается
        в одном документе, а не в двух
        """
        n_documents = next(iter(self._field_counts.values())).shape[0]
        presence = sum((counts > 0).astype(np.int32) for counts in self._field_counts.values())
        document_frequency = np.asarray((presence > 0).sum(axis=0)).ravel()
        self.idf = np.log(
            1 + (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)

    def reweight(self, fields: Sequence[Field] | None = None, *, k1: float | None = None) -> None:
        """Пересобрать веса без повторной токенизации

        Нужно для подбора boost и b: токенизация корпуса занимает минуты,
        а пересборка матрицы секунды
        """
        if fields is not None:
            self.fields = tuple(fields)
        if k1 is not None:
            self.k1 = k1

        combined: sp.csr_matrix | None = None
        for field in self.fields:
            counts = self._field_counts[field.name]
            lengths = self._field_lengths[field.name]
            average = lengths[lengths > 0].mean() if (lengths > 0).any() else 1.0
            norm = 1 - field.b + field.b * lengths / average
            norm[lengths == 0] = 1.0
            scaled = sp.diags(field.boost / norm).dot(counts)
            combined = scaled if combined is None else combined + scaled

        assert combined is not None
        combined = combined.tocsr()
        combined.data = combined.data / (self.k1 + combined.data)
        self._matrix = combined.dot(sp.diags(self.idf)).tocsr().astype(np.float32)

    def iter_scores(self, queries: Sequence[str], *, chunk: int = 256):
        """Отдавать разреженные строки скоров блоками запросов

        Нужно для слияния с другими сигналами: усечённый топ для этого не годится,
        из 2 913 релевантных объявлений 395 не попадают даже в топ-1000 по одному
        только тексту, и отрезав их на этом шаге, вернуть их уже нечем
        """
        if self._matrix is None:
            raise RuntimeError("индекс не построен")
        query_matrix = self._encode(queries)
        for start in range(0, len(queries), chunk):
            stop = min(start + chunk, len(queries))
            yield start, (query_matrix[start:stop] @ self._matrix.T).tocsr()

    def search(
        self, queries: Sequence[str], *, top_k: int = 50, chunk: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть индексы документов и скоры, по убыванию скора

        Произведение считаю разреженным и топ беру прямо из ненулевых элементов строки:
        плотная матрица 2452 на 189212 это 1.85 ГБ, а нулевые скоры в топ всё равно
        не попадут
        """
        indices = np.full((len(queries), top_k), -1, dtype=np.int64)
        scores = np.zeros((len(queries), top_k), dtype=np.float32)

        for start, block in self.iter_scores(queries, chunk=chunk):
            for row in range(block.shape[0]):
                begin, end = block.indptr[row], block.indptr[row + 1]
                top_indices, top_scores = top_k_from_row(
                    block.indices[begin:end], block.data[begin:end], top_k
                )
                indices[start + row, : top_indices.size] = top_indices
                scores[start + row, : top_scores.size] = top_scores
        return indices, scores

    def _encode(self, queries: Sequence[str]) -> sp.csr_matrix:
        """Запрос это бинарный индикатор термов

        Частоту термов в запросе не учитываю: запросы короткие, в среднем 3.2 слова,
        и повтор слова там означает опечатку, а не усиление
        """
        indptr = [0]
        indices: list[int] = []
        for text in queries:
            columns = {
                self.vocabulary[token] for token in self.tokenizer(text) if token in self.vocabulary
            }
            indices.extend(sorted(columns))
            indptr.append(len(indices))
        return sp.csr_matrix(
            (
                np.ones(len(indices), dtype=np.float32),
                np.array(indices, dtype=np.int32),
                np.array(indptr),
            ),
            shape=(len(queries), len(self.vocabulary)),
        )

    def query_terms(self, text: str) -> np.ndarray:
        """Идентификаторы термов запроса, попавших в словарь"""
        columns = {
            self.vocabulary[token] for token in self.tokenizer(text) if token in self.vocabulary
        }
        return np.array(sorted(columns), dtype=np.int64)

    def presence(self, field_name: str) -> FieldPresence:
        """Быстрая проверка «стоит ли терм в этом поле этого объявления»"""
        field = next(item for item in self.fields if item.name == field_name)
        lengths = self._field_lengths[field_name]
        average = lengths[lengths > 0].mean() if (lengths > 0).any() else 1.0
        norm = 1 - field.b + field.b * lengths / average
        norm[lengths == 0] = 1.0
        return FieldPresence(
            self._field_counts[field_name].tocsc(),
            lengths=lengths,
            norm=norm,
            boost=field.boost,
            k1=self.k1,
        )


class FieldPresence:
    """Обратный список одного поля: терм -> отсортированные номера объявлений

    Нужен для признаков переранжировщика. Токенизировать заголовки и описания заново
    в питоне это миллионы вызовов, а индекс их уже переварил: достаточно взять столбец
    матрицы частот и пересечь его с кандидатами
    """

    __slots__ = ("_boost", "_data", "_indices", "_indptr", "_k1", "length", "norm")

    def __init__(
        self,
        counts: sp.csc_matrix,
        *,
        lengths: np.ndarray,
        norm: np.ndarray,
        boost: float,
        k1: float,
    ) -> None:
        counts.sort_indices()
        self._indices = counts.indices
        self._indptr = counts.indptr
        self._data = counts.data
        self._boost = boost
        self._k1 = k1
        self.length = lengths
        self.norm = norm

    def counts(self, term: int, items: np.ndarray) -> np.ndarray:
        """Частота терма в поле каждого объявления, ноль там, где терма нет"""
        begin, end = self._indptr[term], self._indptr[term + 1]
        column = self._indices[begin:end]
        if column.size == 0:
            return np.zeros(items.size)
        position = np.searchsorted(column, items).clip(0, column.size - 1)
        hit = column[position] == items
        return np.where(hit, self._data[begin:end][position], 0.0)

    def score(self, terms: np.ndarray, idf: np.ndarray, items: np.ndarray) -> np.ndarray:
        """BM25 по одному этому полю

        Индекс считает насыщение один раз по сумме всех полей, и это правильно для отбора.
        Но переранжировщику полезно знать, где именно совпало: «нашлось в заголовке»
        и «нашлось только в описании» это разные ситуации, а в общем скоре они слиты
        """
        total = np.zeros(items.size)
        if terms.size == 0 or items.size == 0:
            return total
        weights = self._boost * (1.0 / self.norm[items])
        for position, term in enumerate(terms):
            value = weights * self.counts(int(term), items)
            total += idf[position] * value / (self._k1 + value)
        return total

    def contains(self, term: int, items: np.ndarray) -> np.ndarray:
        # столбец отсортирован, поэтому не isin, а двоичный поиск: столбец частого
        # терма это сотня тысяч номеров, и линейный проход по нему на каждый запрос
        # обходится дороже всего остального вместе взятого
        begin, end = self._indptr[term], self._indptr[term + 1]
        column = self._indices[begin:end]
        if column.size == 0:
            return np.zeros(items.size, dtype=bool)
        position = np.searchsorted(column, items).clip(0, column.size - 1)
        return column[position] == items

    def coverage(
        self, terms: np.ndarray, items: np.ndarray, weights: np.ndarray | None
    ) -> np.ndarray:
        """Доля термов запроса, встреченных в поле, при желании взвешенная по idf"""
        if terms.size == 0 or items.size == 0:
            return np.zeros(items.size)
        hits = np.zeros(items.size)
        total = 0.0
        for position, term in enumerate(terms):
            weight = 1.0 if weights is None else float(weights[position])
            hits += weight * self.contains(int(term), items)
            total += weight
        return hits / total if total else hits
