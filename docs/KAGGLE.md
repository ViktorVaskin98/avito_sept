# Как прогнать тяжёлый шаг на Kaggle

Единственная часть решения, которой нужен GPU, это дообучение би-энкодера и кодирование
корпусов. На ноутбуке без видеокарты это сутки с лишним, на бесплатной T4 полтора часа.
Всё остальное — разбор данных, лексический индекс, гео, фасеты, переранжирование, сборка
ответа — считается локально за минуты.

Код от Kaggle не зависит. Те же самые команды работают локально с `--device cpu`, просто дольше,
и в README это указано. Kaggle тут арендованная видеокарта, а не часть решения.

## Что понадобится

- аккаунт Kaggle с подтверждённым телефоном (без подтверждения не включается Internet,
  а без него не скачаются веса модели с HuggingFace)
- репозиторий, доступный по ссылке: на Kaggle он ставится через `pip install`
- три файла данных: `train.parquet`, `benchmark_items.parquet`, `benchmark_queries.parquet`

## Шаг 1. Загрузить данные приватным датасетом

**Датасет обязан быть приватным.** Это данные отборочного задания Авито, выкладывать их
в открытый доступ нельзя.

Через браузер:

1. `kaggle.com` → в левом меню **Datasets** → кнопка **New Dataset**
2. перетащить три файла из `data/raw/`
3. в поле Title написать `avito-sept`
4. справа в блоке visibility выбрать **Private**
5. **Create**

Заливается 686 МБ, это несколько минут.

Через консоль, если удобнее (нужен `pip install kaggle` и токен в `~/.kaggle/kaggle.json`):

```bash
kaggle datasets init -p data/raw
```

В появившемся `data/raw/dataset-metadata.json` поставить `"title": "avito-sept"`
и `"id": "<твой-логин>/avito-sept"`, затем:

```bash
kaggle datasets create -p data/raw --dir-mode zip
```

Датасет смонтируется в `/kaggle/input/<слаг>`. Слаг не обязан совпадать с названием,
поэтому в ячейке 2 путь не пишется руками, а ищется.

## Шаг 2. Создать ноутбук

1. `kaggle.com` → **Code** → **New Notebook**
2. справа панель настроек, раздел **Input** → **Add Input** → вкладка **Datasets** →
   найти свой `avito-sept` → **Add**
3. там же раздел **Accelerator** → выбрать **GPU T4 x2** (или **GPU P100**, разницы
   для нас почти нет)
4. там же **Internet** → **On**

Если тумблер Internet серый, значит телефон не подтверждён: Settings → Phone Verification.

## Шаг 3. Ячейки

Содержимое лежит в [`notebooks/kaggle_encoder.py`](../notebooks/kaggle_encoder.py),
разделители `# %%` соответствуют ячейкам. Ниже то же самое в готовом виде.

**Ячейка 1. Репозиторий.**

```python
!GIT_TERMINAL_PROMPT=0 git clone -q https://github.com/ViktorVaskin98/avito_sept.git /kaggle/working/avito
!pip install -q -e /kaggle/working/avito
```

Клонирую, а не ставлю пакет напрямую, чтобы сохранилась структура папок: пути в проекте
считаются от корня репозитория.

`GIT_TERMINAL_PROMPT=0` тут не для красоты. Если репозиторий приватный, git попросит логин,
а отвечать ему в ноутбуке некому, и ячейка просто зависнет навсегда без единого сообщения.
С этой переменной она падает сразу и понятно.

**Если репозиторий приватный**, есть два выхода. Первый: сделать его публичным, всё равно
по условию задания ссылку надо открыть проверяющему, а данные в репозиторий не входят.
Второй: залить код вторым приватным датасетом. Локально собрать архив без данных и окружения,

```bash
tar --exclude=avito_sept/data --exclude=avito_sept/.venv --exclude=avito_sept/.git --exclude=avito_sept/task --exclude='*__pycache__*' -czf avito_code.tar.gz avito_sept
```

залить его датасетом `avito-sept-code` и заменить ячейку на

```python
!tar -xzf /kaggle/input/avito-sept-code/avito_code.tar.gz -C /kaggle/working
!mv /kaggle/working/avito_sept /kaggle/working/avito
!pip install -q -e /kaggle/working/avito
```

**Ячейка 2. Данные.**

```python
import os, shutil, sys
from pathlib import Path

WORK = Path("/kaggle/working/avito")
(WORK / "data").mkdir(parents=True, exist_ok=True)

# редактируемая установка кладёт .pth в site-packages, а он читается только при старте
# интерпретатора, поэтому работающее ядро пакета не видит
sys.path.insert(0, str(WORK / "src"))

# путь монтирования зависит от слага датасета, а не от его названия: ищу файлы
found = {p.name: p for p in Path("/kaggle/input").rglob("*.parquet")}
print("нашёл:", {k: str(v) for k, v in found.items()})

RAW = WORK / "data" / "raw"
dataset = found["train.parquet"].parent
if not RAW.exists():
    RAW.symlink_to(dataset, target_is_directory=True)

os.environ["AVITO_CG_ROOT"] = str(WORK)
os.chdir(WORK)
print(sorted(p.name for p in RAW.iterdir()))
```

Три вещи в этой ячейке неочевидны, и на каждой я уже наступил.

`sys.path.insert` нужен потому, что `pip install -e` кладёт в `site-packages` файл `.pth`,
а он читается только при старте интерпретатора. Команды `!avito-cg ...` работают и без этого,
они идут отдельным процессом, а вот `import avito_cg` в самом ноутбуке упадёт.

Путь к датасету ищется, а не пишется руками: Kaggle монтирует его по слагу, а слаг
не обязан совпадать с названием, которое видно в панели.

Симлинк вместо копирования: 686 МБ незачем таскать внутрь рабочей папки, `/kaggle/input`
доступен на чтение, а писать нам туда и не надо.

Переменная `AVITO_CG_ROOT` нужна на случай, если пакет встанет не в editable-режиме:
тогда без неё он будет искать данные внутри `site-packages`.

**Ячейка 3. Проверка видеокарты.**

```python
import numpy, torch
print("numpy", numpy.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
from avito_cg.config import PATHS
print("корень:", PATHS.root)
```

```python
!avito-cg check-data
```

Если `cuda` это `False`, дальше идти бессмысленно: вернись в настройки и включи Accelerator.
Если `check-data` напечатал три файла с размерами, данные на месте.

**Ячейка 4. Локальный бенчмарк.**

```python
!avito-cg split --no-sanity
```

Сплит детерминирован: те же данные и тот же seed дают ровно ту же выборку, что на ноутбуке,
поэтому его не надо перетаскивать файлом. Занимает около минуты.

**Ячейка 5. Дообучение.**

```python
!avito-cg train-encoder --batch-size 96 --epochs 2
```

Около часа на T4. В логе видно скорость и оставшееся время. Если по логу выходит сильно
больше полутора часов, уменьши `--epochs` до 1.

**Ячейка 6. Кодирование корпусов.**

```python
!avito-cg encode-corpus local
!avito-cg encode-corpus benchmark
```

По 15-20 минут каждый. Одна и та же модель кодирует оба корпуса: так локальный замер
описывает ровно ту конфигурацию, которая уйдёт на платформу.

**Ячейка 7. Собрать результаты для скачивания.**

```python
import shutil
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(exist_ok=True)
ARTIFACTS = Path("/kaggle/working/avito/data/artifacts")

shutil.make_archive(str(OUT / "encoder"), "zip", ARTIFACTS / "encoder")
for name in ("embeddings_local.npy", "embeddings_local_item_ids.npy",
             "embeddings_benchmark.npy", "embeddings_benchmark_item_ids.npy"):
    shutil.copy(ARTIFACTS / name, OUT / name)

for path in sorted(OUT.iterdir()):
    print(f"{path.name:36s} {path.stat().st_size / 1e6:8.1f} МБ")
```

## Шаг 4. Запустить

Кнопка **Save Version** справа вверху → **Save & Run All (Commit)** → **Save**.

Ноутбук уедет считаться на сервер. Вкладку можно закрыть, компьютер выключить. Прогресс
виден в списке версий, туда же придёт уведомление о завершении.

Не запускай ячейки по одной в интерактивном режиме: там сессия отваливается по таймауту
бездействия, и часовое обучение до конца не доживёт.

## Шаг 5. Скачать результаты

Когда версия досчитается: открыть её → вкладка **Output** → папка `out` → **Download All**.

Ожидаемые размеры:

| файл | размер |
|---|---|
| `encoder.zip` | ~540 МБ |
| `embeddings_local.npy` | ~290 МБ |
| `embeddings_benchmark.npy` | ~290 МБ |
| файлы с `item_ids` | по паре мегабайт |

Итого около 1.1 ГБ.

## Шаг 6. Что делать локально

Распаковать и разложить:

```bash
unzip -q encoder.zip -d data/artifacts/encoder
```

```bash
cp embeddings_*.npy data/artifacts/
```

Дальше всё как обычно, плотный поиск становится ещё одним сигналом в том же слиянии:

```bash
uv run avito-cg fuse
```

Запросов всего 2 452, они кодируются локально на CPU за пару минут, поэтому качать
эмбеддинги запросов не нужно. Именно ради этого веса модели и скачиваются целиком,
а не только матрица эмбеддингов: решение остаётся работающим на произвольном новом запросе,
как того требует условие.

## Если что-то пошло не так

**Первая ячейка висит, и в выводе `Username for https://github.com`.** Репозиторий приватный,
git ждёт логин, а ввести его в ноутбуке некуда. Останови ячейку и смотри выше, в разделе
про ячейку 1: либо открыть репозиторий, либо залить код датасетом.

**`ModuleNotFoundError: No module named 'avito_cg'` сразу после установки.** Редактируемая
установка кладёт `.pth` в `site-packages`, а он читается только при старте интерпретатора.
Либо `sys.path.insert(0, "/kaggle/working/avito/src")`, как в ячейке 2, либо перезапуск ядра.

**`FileNotFoundError` на файле из `/kaggle/input`.** Датасет монтируется по слагу, а слаг
не обязан совпадать с названием в панели. Ячейка 2 ищет файлы через `rglob`, а не строит путь.

**pip ругается на конфликты версий после установки.** Это жалобы чужих пакетов образа
на то, что мы подняли им зависимости. Нас это не ломает, пока `import torch` работает
и `cuda` возвращает `True`. Если всё-таки сломало, поставь пакет без перетягивания
зависимостей: `pip install -q -e /kaggle/working/avito --no-deps` плюс `pip install -q pystemmer`.

**Тумблер Internet не включается.** Не подтверждён телефон: Settings → Phone Verification.

**`cuda: False`.** Accelerator не выбран или квота на неделю исчерпана. Квота видна
на странице Notebooks, её хватает с большим запасом: нам нужен примерно час из тридцати.

**Кончилось место.** В `/kaggle/working` лимит 20 ГБ. Мы занимаем около 3 ГБ,
так что это означает, что что-то дублируется, скорее всего данные скопированы дважды.

**Обучение идёт заметно медленнее, чем в логе обещано.** Проверь, что выбрана GPU,
а не CPU: на CPU скорость упадёт примерно в тридцать раз. Для быстрой проверки
кода без ожидания есть `--max-pairs 5000 --epochs 1`.

**Хочется сначала убедиться, что всё работает.** Прогони сокращённый вариант,
он занимает пару минут:

```bash
avito-cg train-encoder --model cointegrated/rubert-tiny2 --max-pairs 2000 --epochs 1 --batch-size 32
```
