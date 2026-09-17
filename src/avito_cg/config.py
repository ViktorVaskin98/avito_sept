"""Пути и константы

Пути считаю от корня репозитория, а не от рабочей директории: иначе ноутбук из notebooks/
и скрипт из src/ видят разные data/ и я на этом уже попадался
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# в answer.csv разрешено не больше 50 item_id на запрос
TOP_K = 50

# и query_id, и item_id всегда ровно 16 символов
ID_LENGTH = 16

RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        # исходные parquet как есть с платформы
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        # нормализованные тексты, распарсенные параметры, сплиты
        return self.data / "interim"

    @property
    def artifacts(self) -> Path:
        # тяжёлое и пересоздаваемое: индексы, эмбеддинги, веса
        return self.data / "artifacts"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def train(self) -> Path:
        return self.raw / "train.parquet"

    @property
    def benchmark_items(self) -> Path:
        return self.raw / "benchmark_items.parquet"

    @property
    def benchmark_queries(self) -> Path:
        return self.raw / "benchmark_queries.parquet"

    def ensure(self) -> None:
        for path in (self.interim, self.artifacts, self.submissions, self.figures):
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()
