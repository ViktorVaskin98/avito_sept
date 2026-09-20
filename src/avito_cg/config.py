"""Пути и константы

Пути считаю от корня репозитория, а не от рабочей директории: иначе ноутбук из notebooks/
и скрипт из src/ видят разные data/ и я на этом уже попадался
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# по умолчанию корень это репозиторий, в котором лежит пакет, и при установке через
# pip install -e так и получается. Но при обычной установке пакет уезжает в site-packages,
# и корень вместе с ним, поэтому его можно переопределить снаружи. Нужно для Kaggle
REPO_ROOT = Path(os.environ.get("AVITO_CG_ROOT") or Path(__file__).resolve().parents[2])

# в answer.csv разрешено не больше 50 item_id на запрос
TOP_K = 50

# и query_id, и item_id всегда ровно 16 символов
ID_LENGTH = 16

# сколько кандидатов приводит плотный поиск помимо лексических. Он нужен не только
# чтобы переупорядочивать найденное лексикой: у 3.0% пар нет ни одного общего токена
# с объявлением, и в лексическое множество они не попадают вовсе.
#
# Значение подобрано замером, см. docs/EXPERIMENTS.md: 200 -> 1000 поднимает потолок
# пула кандидатов с 0.9608 до 0.9679, парный прирост по Recall@50 +0.0041
# [+0.0016, +0.0069]. Константа лежит здесь, а не в трёх модулях по копии, именно
# потому, что тремя копиями её ни разу и не переподобрали
DENSE_DEPTH = 1000

RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class Paths:
    """Все пути проекта в одном месте, считаются от корня репозитория

    Свойствами, а не полями: корень можно переопределить через AVITO_CG_ROOT,
    и производные пути обязаны поехать за ним
    """

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
