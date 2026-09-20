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

# %% ячейка 1: репозиторий
# GIT_TERMINAL_PROMPT=0 обязателен: у приватного репозитория git спросит логин,
# отвечать ему в ноутбуке некому, и ячейка зависнет навсегда без сообщений
# !GIT_TERMINAL_PROMPT=0 git clone -q https://github.com/ViktorVaskin98/avito_sept.git /kaggle/working/avito
# !pip install -q -e /kaggle/working/avito

# %% ячейка 2: подключение данных
import os
import sys
from pathlib import Path

WORK = Path("/kaggle/working/avito")
(WORK / "data").mkdir(parents=True, exist_ok=True)

# редактируемая установка кладёт .pth в site-packages, а он читается только при старте
# интерпретатора, поэтому работающее ядро пакета не видит
sys.path.insert(0, str(WORK / "src"))

# путь монтирования зависит от слага датасета, а не от названия в панели
found = {path.name: path for path in Path("/kaggle/input").rglob("*.parquet")}
print("нашёл:", {name: str(path) for name, path in found.items()})

RAW = WORK / "data" / "raw"
if not RAW.exists():
    # симлинк вместо копии: 686 МБ незачем таскать, /kaggle/input нужен только на чтение
    RAW.symlink_to(found["train.parquet"].parent, target_is_directory=True)

# корень проекта считается от расположения пакета, и при установке не в editable-режиме
# он уехал бы в site-packages вместе с ним
os.environ["AVITO_CG_ROOT"] = str(WORK)
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
