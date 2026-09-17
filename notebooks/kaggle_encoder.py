"""Обёртка для Kaggle: дообучение би-энкодера и кодирование корпусов на GPU

Это не часть решения, а способ запустить единственный тяжёлый шаг там, где есть видеокарта.
Содержимое копируется в ячейки Kaggle-ноутбука по разделителям. Сам код решения живёт
в пакете и от Kaggle не зависит: те же команды работают локально с --device cpu, просто дольше.

Инструкция по запуску: docs/KAGGLE.md
"""

# %% [markdown]
# # Дообучение би-энкодера
#
# Нужен GPU (T4) и включённый Internet. Датасет с тремя parquet подключается
# как `/kaggle/input/avito-sept`.

# %% ячейка 1: установка пакета
# !pip install -q git+https://github.com/ViktorVaskin98/avito_sept.git

# %% ячейка 2: подключение данных
import os
import shutil
from pathlib import Path

WORK = Path("/kaggle/working/avito")
RAW = WORK / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCE = Path("/kaggle/input/avito-sept")
for name in ("train.parquet", "benchmark_items.parquet", "benchmark_queries.parquet"):
    target = RAW / name
    if not target.exists():
        shutil.copy(SOURCE / name, target)
os.chdir(WORK)
print(sorted(path.name for path in RAW.iterdir()))

# %% ячейка 3: проверка, что видеокарта на месте
import torch

print("cuda:", torch.cuda.is_available())
print("устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет")

# %% ячейка 4: локальный бенчмарк
# Сплит детерминирован: те же данные плюс тот же seed дают ту же выборку,
# что и на ноутбуке, поэтому его не надо перетаскивать файлом
# !avito-cg split --no-sanity

# %% ячейка 5: дообучение, около часа на T4
# !avito-cg train-encoder --batch-size 96 --epochs 2

# %% ячейка 6: кодирование обоих корпусов
# Одна и та же модель кодирует и локальный корпус, и настоящий:
# так локальный замер описывает ровно то, что уйдёт на платформу
# !avito-cg encode-corpus local
# !avito-cg encode-corpus benchmark

# %% ячейка 7: сложить результаты туда, откуда Kaggle даст их скачать
import shutil
from pathlib import Path

OUT = Path("/kaggle/working/out")
OUT.mkdir(exist_ok=True)
ARTIFACTS = Path("/kaggle/working/avito/data/artifacts")

shutil.make_archive(str(OUT / "encoder"), "zip", ARTIFACTS / "encoder")
for name in (
    "embeddings_local.npy",
    "embeddings_local_item_ids.npy",
    "embeddings_benchmark.npy",
    "embeddings_benchmark_item_ids.npy",
):
    shutil.copy(ARTIFACTS / name, OUT / name)

for path in sorted(OUT.iterdir()):
    print(f"{path.name:36s} {path.stat().st_size / 1e6:8.1f} МБ")
