"""Переранжирование кандидатов на локальном бенчмарке

Меряю четыре вещи по очереди, каждая отвечает на отдельный вопрос:

1. сколько вообще можно выиграть - идеальное переупорядочивание топ-200
2. хватает ли отложенных запросов для обучения, кросс-валидация по фолдам
3. помогают ли донорские запросы, которых в девять раз больше
4. нужна ли ранжирующая постановка вместо двоичной

Сравнения парные: конфигурации гоняются на одних и тех же запросах, и в абсолютных
интервалах их разница тонет.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from avito_cg.config import PATHS, TOP_K
from avito_cg.data.io import load_train
from avito_cg.eval.benchmark import LocalBenchmark
from avito_cg.eval.metrics import paired_bootstrap, recall_at_k, recall_per_query
from avito_cg.index.fields import query_texts
from avito_cg.index.geo import GeoIndex
from avito_cg.index.lexical import BM25FIndex
from avito_cg.retrieval.candidates import collect
from avito_cg.retrieval.signals import (
    DenseSignal,
    FacetSignal,
    GeoSignal,
    MicrocatSignal,
    load_embeddings,
)
from avito_cg.train.reranker import (
    FEATURES,
    CandidateSet,
    ItemContext,
    build_features,
    cross_validated_scores,
    fit,
    importance,
    predict,
)

COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_latitude",
    "item_longitude",
    "item_location_id",
    "item_microcat_id",
    "item_infm_params_text",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]
TRAIN_COLUMNS = ["search_location_id", "search_query", "search_infm_params_text", *COLUMNS]
WEIGHTS = {"гео": 0.10, "фасеты": 0.10, "микрокатегория": 0.02, "плотный": 0.5}
DEPTH = 200
DENSE_DEPTH = 200

# признаки уровня объявления. Выношу отдельно, чтобы измерить их вклад отдельным замером:
# на донорских запросах популярность смещена по построению, и это надо видеть
PRIOR_FEATURES = ("популярность", "знакомое", "память", "близнецов", "цена", "рейтинг", "отзывов")


DONOR_DIR = "donor_queries"
DONOR_VECTORS = "query_vectors_donor.npy"


def build_donor(*, n_queries: int = 40_000, device: str = "auto") -> None:
    """Набрать донорские запросы, сохранить их разметку и закодировать тексты

    Кодирование тут самое дорогое: на процессоре семнадцать тысяч запросов это около часа.
    Поэтому векторы кладутся в артефакты и переиспользуются всеми последующими замерами
    """
    from avito_cg.train.biencoder import QUERY_PREFIX, encode
    from avito_cg.train.extra import build_donor_set

    PATHS.ensure()
    train = load_train(
        columns=[
            "search_query",
            "search_location_id",
            "search_is_delivery_search",
            "search_infm_params_text",
            "search_category",
            "item_id",
        ]
    )
    local = LocalBenchmark.load(PATHS.interim / "local_benchmark")
    queries, relevant, report = build_donor_set(train, local, n_queries=n_queries)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    directory = PATHS.interim / DONOR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(directory / "queries.parquet", index=False)
    pd.DataFrame(
        [
            {"query_id": query, "item_id": item}
            for query, items in relevant.items()
            for item in sorted(items)
        ]
    ).to_parquet(directory / "relevant.parquet", index=False)
    (directory / "meta.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    started = time.time()
    vectors = encode(
        queries["search_query"].fillna("").astype(str).tolist(),
        PATHS.artifacts / "encoder",
        prefix=QUERY_PREFIX,
        max_length=32,
        device=device,
    )
    np.save(PATHS.artifacts / DONOR_VECTORS, vectors)
    print(f"векторы запросов за {time.time() - started:.0f} c: {vectors.shape}")


def _signals(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    pairs: pd.DataFrame,
    geo: GeoIndex,
    item_vectors: np.ndarray,
    query_vectors: np.ndarray,
) -> tuple[list, DenseSignal]:
    dense = DenseSignal(query_vectors=query_vectors, item_vectors=item_vectors)
    return [
        (GeoSignal.build(geo, queries), WEIGHTS["гео"]),
        (FacetSignal.build(queries, corpus, pairs), WEIGHTS["фасеты"]),
        (MicrocatSignal.build(queries, corpus, pairs), WEIGHTS["микрокатегория"]),
        (dense, WEIGHTS["плотный"]),
    ], dense


def _prepare(
    queries: pd.DataFrame,
    relevant: dict[str, set[str]] | None,
    corpus: pd.DataFrame,
    pairs: pd.DataFrame,
    geo: GeoIndex,
    index: BM25FIndex,
    context: ItemContext,
    item_vectors: np.ndarray,
    query_vectors: np.ndarray,
    *,
    depth: int = DEPTH,
    cache: Path | None = None,
) -> tuple[np.ndarray, CandidateSet, list[str]]:
    """Выдача до переранжирования и таблица признаков по тем же кандидатам

    Сбор кандидатов это самая дорогая часть прогона, семь минут на две с половиной тысячи
    запросов, поэтому результат кладётся рядом. Кэш привязан к набору признаков:
    поменял FEATURES - удали файл
    """
    query_ids = queries["query_id"].astype(str).tolist()
    if cache is not None and cache.exists():
        stored = np.load(cache)
        rows = len(stored["label"])
        print(f"  таблица признаков из кэша {cache.name}, строк {rows}", flush=True)
        data = CandidateSet(
            query=stored["query"],
            item=stored["item"],
            features=stored["features"],
            label=stored["label"],
        )
        return stored["order"], data, query_ids

    signals, dense = _signals(queries, corpus, pairs, geo, item_vectors, query_vectors)
    started = time.time()
    extra = dense.top_candidates(top_k=DENSE_DEPTH)
    collected = collect(
        index,
        signals,
        query_texts(queries),
        depth=depth,
        extra_candidates=extra,
    )
    print(f"  кандидаты за {time.time() - started:.0f} c", flush=True)

    latitude, longitude = geo.query_coordinates(queries["search_location_id"].to_numpy())
    started = time.time()
    data = build_features(
        collected.items,
        collected.lexical,
        collected.scores,
        queries,
        context,
        index,
        query_latitude=latitude,
        query_longitude=longitude,
        query_ids=query_ids,
        relevant=relevant,
    )
    print(f"  признаки за {time.time() - started:.0f} c, строк {len(data.label)}", flush=True)

    order = collected.order(TOP_K)
    if cache is not None:
        np.savez_compressed(
            cache,
            features=data.features,
            label=data.label,
            query=data.query,
            item=data.item,
            order=order,
        )
    return order, data, query_ids


def _reorder(
    data: CandidateSet,
    scores: np.ndarray,
    item_ids: np.ndarray,
    query_ids: list[str],
    *,
    top_k: int = TOP_K,
) -> dict[str, list[str]]:
    """Выдача после переранжирования: топ-k по скору внутри каждого запроса"""
    order = np.lexsort((-scores, data.query))
    predictions: dict[str, list[str]] = {query: [] for query in query_ids}
    for position in order:
        query = query_ids[data.query[position]]
        if len(predictions[query]) < top_k:
            predictions[query].append(str(item_ids[data.item[position]]))
    return predictions


def run(*, donor: bool = True, top_k: int = TOP_K, save: str | None = None) -> None:
    """Прогнать замеры и, если попросили, сохранить выбранную модель для сборки ответа

    save это имя конфигурации из сводки. Выбираю её руками по результатам замера,
    а не автоматическим максимумом: разница между соседними конфигурациями бывает
    в пределах шума, и брать по ней argmax значит подбирать шум
    """
    PATHS.ensure()
    started = time.time()
    train = load_train(columns=TRAIN_COLUMNS)
    local = LocalBenchmark.load(PATHS.interim / "local_benchmark")
    corpus = local.corpus(train, columns=COLUMNS)
    pairs = local.fit_pairs(train)
    geo = GeoIndex.fit(pairs, corpus)
    index = BM25FIndex.load(PATHS.artifacts / "lexical_local")
    context = ItemContext.build(corpus, index, pairs)
    item_ids = context.item_ids
    item_vectors = load_embeddings(PATHS.artifacts / "embeddings_local.npy", item_ids)
    print(f"подготовка за {time.time() - started:.0f} c", flush=True)

    print("\nотложенные запросы")
    base_order, data, query_ids = _prepare(
        local.queries,
        local.relevant,
        corpus,
        pairs,
        geo,
        index,
        context,
        item_vectors,
        np.load(PATHS.artifacts / "query_vectors_local.npy"),
        cache=PATHS.artifacts / "rerank_local.npz",
    )

    base = {
        query: [str(item_ids[position]) for position in row if position >= 0]
        for query, row in zip(query_ids, base_order, strict=True)
    }
    base_scores = recall_per_query(local.relevant, base)
    print(f"\nбез переранжирования: {recall_at_k(local.relevant, base):.4f}")

    ceiling = _reorder(data, data.label.astype(float), item_ids, query_ids, top_k=top_k)
    print(f"потолок топ-{DEPTH}, идеальный порядок: {recall_at_k(local.relevant, ceiling):.4f}")

    results: list[tuple[str, np.ndarray]] = []
    # обучение финальной модели откладываю до конца: если сохранять не просили,
    # лишние несколько минут ни к чему
    models: dict[str, object] = {}

    print("\nобучение на отложенных, кросс-валидация по запросам", flush=True)
    for name, features in (("все признаки", FEATURES), ("без приоров", _without_priors())):
        scores = cross_validated_scores(data, len(query_ids), features=features)
        per_query = recall_per_query(
            local.relevant, _reorder(data, scores, item_ids, query_ids, top_k=top_k)
        )
        results.append((f"отложенные, {name}", per_query))
        models[f"отложенные, {name}"] = partial(fit, data, features=features)
        _report(f"  {name}", base_scores, per_query)

    if donor:
        donor_dir = PATHS.interim / DONOR_DIR
        vectors = PATHS.artifacts / DONOR_VECTORS
        if not (donor_dir / "queries.parquet").exists() or not vectors.exists():
            print(f"\nдонорского набора нет в {donor_dir}, пропускаю", flush=True)
        else:
            print("\nдонорские запросы", flush=True)
            donor_queries = pd.read_parquet(donor_dir / "queries.parquet")
            donor_relevant: dict[str, set[str]] = {}
            for query, item in pd.read_parquet(donor_dir / "relevant.parquet").itertuples(
                index=False
            ):
                donor_relevant.setdefault(str(query), set()).add(str(item))
            _, donor_data, _ = _prepare(
                donor_queries,
                donor_relevant,
                corpus,
                pairs,
                geo,
                index,
                context,
                item_vectors,
                np.load(vectors),
                cache=PATHS.artifacts / "rerank_donor.npz",
            )

            for objective in ("logloss", "yetirank"):
                started = time.time()
                model = fit(donor_data, objective=objective)
                scores = predict(model, data.features)
                per_query = recall_per_query(
                    local.relevant, _reorder(data, scores, item_ids, query_ids, top_k=top_k)
                )
                results.append((f"донор, {objective}", per_query))
                models[f"донор, {objective}"] = partial(fit, donor_data, objective=objective)
                _report(f"  {objective} за {time.time() - started:.0f} c", base_scores, per_query)
                if objective == "logloss":
                    print(importance(model).to_string(index=False))

    print("\nсводка")
    print(f"  {'без переранжирования':32s} {base_scores.mean():.4f}")
    for name, per_query in results:
        print(f"  {name:32s} {per_query.mean():.4f}")

    if save is not None:
        if save not in models:
            raise KeyError(f"нет такой конфигурации: {save}, есть {sorted(models)}")
        destination = PATHS.artifacts / "reranker.cbm"
        models[save]().save_model(str(destination))
        # постановку задачи CatBoost держит внутри модели, но класс для загрузки надо знать
        # заранее, поэтому кладу её рядом обычным json, а не выковыриваю из бинарника
        destination.with_suffix(".json").write_text(
            json.dumps(
                {"имя": save, "objective": "yetirank" if "yetirank" in save else "logloss"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print()
        print(f"модель «{save}» обучена на всех строках и сохранена в {destination}")


def _without_priors() -> list[str]:
    return [name for name in FEATURES if name not in PRIOR_FEATURES]


def _report(label: str, base: np.ndarray, per_query: np.ndarray) -> None:
    delta, low, high = paired_bootstrap(base, per_query)
    verdict = "значимо" if low > 0 else "в пределах шума"
    print(
        f"{label}: {per_query.mean():.4f}, "
        f"прирост {delta:+.4f} [{low:+.4f}, {high:+.4f}], {verdict}"
    )


def save_model(model: object, path: Path) -> None:
    model.save_model(str(path))  # type: ignore[attr-defined]


if __name__ == "__main__":
    run()
