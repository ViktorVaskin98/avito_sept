"""Тяжёлый шаг: дообучение би-энкодера и кодирование корпусов

Единственная часть решения, которой нужен GPU. Всё остальное считается на ноутбуке
за минуты, а здесь на CPU получилось бы часов десять, поэтому шаг вынесен в Kaggle.
Скрипт при этом от Kaggle не зависит: с `--device cpu` он работает там же, просто дольше.

Модель обучается **одна**, на разрешённой части обучающих пар, и ей же кодируются оба
корпуса - локальный и настоящий. Так локальный замер описывает ровно ту конфигурацию,
которая уходит на платформу. Цена в том, что 57% пар в обучении не участвуют;
если выигрыш окажется заметным, есть смысл потом обучить вторую модель на всех парах
и сравнить, но сравнивать тогда придётся уже вслепую.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from avito_cg.config import PATHS
from avito_cg.data.io import load_benchmark_items, load_train
from avito_cg.eval.benchmark import LocalBenchmark
from avito_cg.index.fields import (
    encoder_texts,
    frequent_values,
    load_parser,
    params_values,
)
from avito_cg.train.biencoder import TrainingConfig, encode, train

ENCODER_DIR = "encoder"
ITEM_COLUMNS = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
TRAIN_COLUMNS = ["search_query", *ITEM_COLUMNS]
SPLIT_DIR = "local_benchmark"


def run_training(config: TrainingConfig, *, output: Path | None = None) -> Path:
    """Дообучить энкодер на парах, которыми разрешено пользоваться"""
    PATHS.ensure()
    destination = output or PATHS.artifacts / ENCODER_DIR

    started = time.time()
    train_frame = load_train(columns=TRAIN_COLUMNS)
    local = LocalBenchmark.load(PATHS.interim / SPLIT_DIR)
    pairs = local.fit_pairs(train_frame)
    if config.max_pairs:
        pairs = pairs.sample(min(config.max_pairs, len(pairs)), random_state=config.seed)

    # частые значения параметров считаю по уникальным объявлениям, а не по парам:
    # в парах популярное объявление лежит десятки раз, и его значения выглядели бы
    # частыми там, где по корпусу они редкие. Иначе текст одного и того же объявления
    # при обучении и при кодировании корпуса получается разным
    parser = load_parser()
    unique_items = pairs.drop_duplicates(subset="item_id")
    common = frequent_values(
        params_values(
            parser,
            unique_items["item_infm_params_text"].fillna("").astype(str).tolist(),
            include_address=False,
        )
    )
    queries = pairs["search_query"].fillna("").astype(str).tolist()
    passages = encoder_texts(pairs, parser, common=common)
    del train_frame, pairs
    print(f"пары готовы за {time.time() - started:.0f} c: {len(queries)}", flush=True)

    return train(queries, passages, config, destination)


def run_encoding(
    scope: str,
    model_path: Path | None = None,
    *,
    output: Path | None = None,
    device: str = "auto",
    batch_size: int = 256,
) -> Path:
    """Закодировать корпус: локальный для замеров или настоящий для ответа"""
    PATHS.ensure()
    model = model_path or PATHS.artifacts / ENCODER_DIR
    destination = output or PATHS.artifacts / f"embeddings_{scope}.npy"

    started = time.time()
    if scope == "benchmark":
        items = load_benchmark_items(columns=ITEM_COLUMNS)
    elif scope == "local":
        train_frame = load_train(columns=ITEM_COLUMNS)
        local = LocalBenchmark.load(PATHS.interim / SPLIT_DIR)
        items = local.corpus(train_frame, columns=ITEM_COLUMNS)
        del train_frame
    else:
        raise ValueError(f"неизвестный корпус: {scope}")

    parser = load_parser()
    texts = encoder_texts(items, parser)
    print(f"тексты готовы за {time.time() - started:.0f} c: {len(texts)}", flush=True)

    vectors = encode(texts, model, device=device, batch_size=batch_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, vectors)
    # порядок строк обязан совпадать с порядком корпуса, иначе всё поедет молча,
    # поэтому рядом кладу item_id и сверяю их при загрузке
    np.save(
        destination.with_name(destination.stem + "_item_ids.npy"),
        items["item_id"].astype(str).to_numpy(),
    )
    print(f"эмбеддинги сохранены в {destination}, форма {vectors.shape}", flush=True)
    return destination
