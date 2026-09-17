"""Би-энкодер: дообучение и кодирование корпуса

Зачем он нужен, видно из разбора данных: 3.0% пар «запрос - выбранное объявление» не имеют
ни одного общего токена даже с полным текстом объявления. Запрос «маникюр» против заголовка
«Наращивание ногтей», «работа» против «Копка траншей вручную». Лексический поиск такие пары
не достанет никогда, сколько ни настраивай веса полей.

**Почему in-batch negatives, а не намайненные тяжёлые.** Обычно для ретриверов всё наоборот:
hard negatives дают основной прирост. Здесь они вредны, и причина в данных. Самые тяжёлые
лексические негативы у нас это объявления с **тем же самым заголовком** в другой локации,
а таких полкорпуса: 48.6% объявлений делят заголовок с другими, у «наращивание ресниц»
873 близнеца. Текстовый энкодер их различить не может в принципе, потому что различать нечего,
и обучение на таких парах это противоречивый сигнал. Разводит их гео, оно для этого и стоит
в слиянии. Задача энкодера другая - закрыть семантический разрыв, и для неё случайных
негативов из батча достаточно.

Дообучение идёт на разрешённой части обучающих пар, отложенные запросы туда не попадают.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

BASE_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass(slots=True)
class TrainingConfig:
    """Гиперпараметры дообучения

    Температура 0.05 это стандарт для InfoNCE в ретриверах: при ней softmax по батчу
    достаточно резкий, чтобы градиент шёл от ближайших негативов, а не размазывался.
    Батч важнее скорости обучения: каждый лишний пример в батче это лишний негатив
    """

    model_name: str = BASE_MODEL
    query_max_length: int = 32
    passage_max_length: int = 128
    batch_size: int = 96
    epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    log_every: int = 100
    max_pairs: int | None = None
    query_prefix: str = QUERY_PREFIX
    passage_prefix: str = PASSAGE_PREFIX
    frozen_embeddings: bool = False
    tags: list[str] = field(default_factory=list)

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairDataset(Dataset):
    """Пары «запрос - выбранное объявление»"""

    def __init__(self, queries: Sequence[str], passages: Sequence[str]) -> None:
        if len(queries) != len(passages):
            raise ValueError("число запросов и объявлений не совпадает")
        self.queries = list(queries)
        self.passages = list(passages)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.queries[index], self.passages[index]


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Усреднение по токенам с учётом маски

    CLS-токен у e5 не обучен под представление последовательности, авторы модели
    используют именно среднее
    """
    expanded = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)


def embed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    batch = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    hidden = model(**batch).last_hidden_state
    return functional.normalize(mean_pooling(hidden, batch["attention_mask"]), dim=-1)


def info_nce(queries: torch.Tensor, passages: torch.Tensor, temperature: float) -> torch.Tensor:
    """Симметричный InfoNCE по батчу

    Каждый запрос должен выбрать своё объявление среди всех объявлений батча,
    и наоборот. Симметричная форма сходится заметно ровнее односторонней
    """
    logits = queries @ passages.T / temperature
    target = torch.arange(len(queries), device=queries.device)
    return 0.5 * (
        functional.cross_entropy(logits, target) + functional.cross_entropy(logits.T, target)
    )


def train(
    queries: Sequence[str],
    passages: Sequence[str],
    config: TrainingConfig,
    output: Path,
) -> Path:
    """Дообучить би-энкодер и сохранить веса"""
    torch.manual_seed(config.seed)
    device = config.resolve_device()
    amp = config.use_amp and device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name).to(device)
    model.train()

    if config.frozen_embeddings:
        # таблица эмбеддингов это больше половины параметров e5-base, а её градиент
        # плотный и на CPU съедает больше времени, чем все слои вместе
        for parameter in model.get_input_embeddings().parameters():
            parameter.requires_grad = False

    dataset = PairDataset(queries, passages)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda rows: ([row[0] for row in rows], [row[1] for row in rows]),
    )
    steps = len(loader) * config.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup = max(1, int(steps * config.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print(
        f"устройство {device}, пар {len(dataset)}, батч {config.batch_size}, "
        f"шагов {steps}, amp {amp}",
        flush=True,
    )
    started = time.time()
    step = 0
    for epoch in range(config.epochs):
        running = 0.0
        for query_batch, passage_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                query_vectors = embed(
                    model,
                    tokenizer,
                    [config.query_prefix + text for text in query_batch],
                    max_length=config.query_max_length,
                    device=device,
                )
                passage_vectors = embed(
                    model,
                    tokenizer,
                    [config.passage_prefix + text for text in passage_batch],
                    max_length=config.passage_max_length,
                    device=device,
                )
                loss = info_nce(query_vectors, passage_vectors, config.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            step += 1
            if step % config.log_every == 0:
                speed = step / (time.time() - started)
                left = (steps - step) / max(speed, 1e-9)
                print(
                    f"  эпоха {epoch + 1} шаг {step}/{steps} "
                    f"loss {running / config.log_every:.4f} "
                    f"{speed:.2f} шаг/с осталось {left / 60:.0f} мин",
                    flush=True,
                )
                running = 0.0

    output.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"веса сохранены в {output}, всего {(time.time() - started) / 60:.0f} мин", flush=True)
    return output


@torch.no_grad()
def encode(
    texts: Sequence[str],
    model_path: str | Path,
    *,
    prefix: str = PASSAGE_PREFIX,
    max_length: int = 128,
    batch_size: int = 256,
    device: str = "auto",
    log_every: int = 200,
) -> np.ndarray:
    """Закодировать тексты в нормированные векторы

    Результат во float16: 189 тысяч векторов по 768 измерений это 290 МБ вместо 580,
    а на косинусную близость потеря точности не влияет
    """
    resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    if device == "auto" and resolved.type != "cuda":
        resolved = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path)).to(resolved).eval()
    if resolved.type == "cuda":
        model = model.half()

    vectors = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    started = time.time()
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        chunk = [prefix + text for text in texts[start:stop]]
        vectors[start:stop] = (
            embed(model, tokenizer, chunk, max_length=max_length, device=resolved)
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        if (start // batch_size) % log_every == 0 and start:
            done = stop / len(texts)
            elapsed = time.time() - started
            print(
                f"  закодировано {done:.0%}, осталось {elapsed * (1 - done) / done / 60:.0f} мин",
                flush=True,
            )
    print(f"кодирование заняло {(time.time() - started) / 60:.1f} мин", flush=True)
    return vectors
