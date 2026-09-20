"""Единая точка входа: avito-cg <команда>

Команды добавляю по мере появления модулей. Хочу, чтобы весь путь от сырых parquet
до answer.csv воспроизводился набором вызовов из README, без запуска ноутбуков
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from avito_cg.config import PATHS


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PATHS.root / path


def _cmd_check_data(_: argparse.Namespace) -> int:
    """Проверить, что исходные файлы на месте и читаются"""
    from avito_cg.data.io import available_columns

    sources = (PATHS.train, PATHS.benchmark_items, PATHS.benchmark_queries)
    missing = [path for path in sources if not path.exists()]
    if missing:
        print(f"корень проекта: {PATHS.root}", file=sys.stderr)
        print("не найдены файлы:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        if not (PATHS.root / "pyproject.toml").exists():
            # пакет поставлен не в editable-режиме: корень уехал в site-packages
            # вместе с ним, и data/raw ищется совсем не там, где лежит
            print(
                "\nкорень не похож на репозиторий, в нём нет pyproject.toml. "
                "Похоже, пакет установлен не через `pip install -e`.\n"
                "Укажи корень явно: AVITO_CG_ROOT=/путь/к/репозиторию",
                file=sys.stderr,
            )
        else:
            print(
                "\nскачай архив по ссылке из docs/TASK.md и распакуй содержимое в data/raw/",
                file=sys.stderr,
            )
        return 1

    for path in sources:
        size_mb = path.stat().st_size / 1e6
        print(f"{path.name:28s} {size_mb:8.1f} МБ  колонок: {len(available_columns(path))}")
    return 0


def _cmd_validate_answer(args: argparse.Namespace) -> int:
    """Проверить готовый answer.csv перед отправкой на платформу"""
    from avito_cg.data.io import load_benchmark_items, load_benchmark_queries
    from avito_cg.eval.submission import read_submission, split_problems, validate_submission

    frame = read_submission(args.path)
    queries = load_benchmark_queries(columns=["query_id"])
    items = load_benchmark_items(columns=["item_id"])
    problems = validate_submission(
        frame,
        expected_query_ids=queries["query_id"].astype(str).tolist(),
        corpus_item_ids=set(items["item_id"].astype(str)),
    )
    errors, warnings = split_problems(problems)
    for warning in warnings:
        print(f"  внимание: {warning}", file=sys.stderr)
    if not errors:
        print(f"{args.path}: формат в порядке, {len(frame)} строк")
        return 0
    print(f"{args.path}: формат нарушен", file=sys.stderr)
    for problem in errors:
        print(f"  - {problem}", file=sys.stderr)
    return 1


def _cmd_eda(_: argparse.Namespace) -> int:
    """Прогнать разбор данных: таблицы, графики и reports/eda.json"""
    from avito_cg.cli.eda import run

    run()
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    """Собрать локальный бенчмарк из train и сохранить его"""
    from avito_cg.cli.split import run

    run(sanity=not args.no_sanity, seen_share=args.seen_share, seed=args.seed)
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Прогнать BM25F на локальном бенчмарке"""
    from avito_cg.cli.baseline import run

    run(grid=not args.no_grid)
    return 0


def _cmd_answer(args: argparse.Namespace) -> int:
    """Собрать answer.csv по настоящему корпусу"""
    from avito_cg.cli.answer import DEFAULT_WEIGHTS, run
    from avito_cg.index.lexical import Field
    from avito_cg.retrieval.fusion import FusionConfig

    weights = dict(DEFAULT_WEIGHTS)
    for name, value in (
        ("гео", args.geo_weight),
        ("фасеты", args.facet_weight),
        ("микрокатегория", args.microcat_weight),
        ("плотный", args.dense_weight),
    ):
        if value is not None:
            weights[name] = value

    run(
        fields=[
            Field("title", args.title, 0.6),
            Field("params", args.params, 0.75),
            Field("description", args.description, 0.75),
        ],
        output=args.out,
        with_filter=args.with_filter,
        fusion=None if args.no_geo else FusionConfig(mode=args.fusion),
        weights=weights,
        rerank=not args.no_rerank,
        dense_depth=args.dense_depth,
    )
    return 0


def _cmd_param_keys(args: argparse.Namespace) -> int:
    """Сверить закреплённый словарь ключей параметров со свежевыведенным

    Печатает оба, чтобы было видно цену закрепления: словарь в пакете держит
    воспроизводимость, а выведенный заново по корпусу может разбирать чуть лучше.
    Менять закреплённый имеет смысл только вместе с пересборкой всей цепочки
    """
    from avito_cg.data.io import load_benchmark_items
    from avito_cg.data.params import ParamsParser
    from avito_cg.index.fields import PINNED_KEYS, discover_parser, load_parser

    texts = (
        load_benchmark_items(columns=["item_infm_params_text"])["item_infm_params_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    nonempty = [text for text in texts if text]

    def coverage(parser: ParamsParser) -> float:
        return sum(parser.coverage(text) == 1.0 for text in nonempty) / len(nonempty)

    pinned = load_parser()
    fresh = discover_parser(texts, sample=args.sample, seed=args.seed)
    print(f"строк с непустыми параметрами: {len(nonempty)}")
    print(
        f"  закреплённый: {len(pinned.keys):3d} ключей, разобрано от первого токена "
        f"{coverage(pinned):.4f}"
    )
    print(
        f"  выведенный:   {len(fresh.keys):3d} ключей, разобрано от первого токена "
        f"{coverage(fresh):.4f}"
    )
    only_fresh = sorted(set(fresh.keys) - set(pinned.keys))
    only_pinned = sorted(set(pinned.keys) - set(fresh.keys))
    print(f"  есть только в выведенном ({len(only_fresh)}): {only_fresh[:5]}")
    print(f"  есть только в закреплённом ({len(only_pinned)}): {only_pinned[:5]}")

    if args.write:
        fresh.save(PINNED_KEYS)
        print(f"\nсловарь перезаписан: {PINNED_KEYS}")
        print("теперь надо удалить индексы и таблицы признаков в data/artifacts")
        print("и пересобрать всю цепочку: поле params, а с ним и ответ, изменились")
    else:
        print(f"\nничего не записано, для перезаписи нужен --write ({PINNED_KEYS})")
    return 0


def _cmd_errors(args: argparse.Namespace) -> int:
    """Разбор промахов финальной конфигурации на локальном бенчмарке"""
    from avito_cg.cli.errors import run

    run(top_k=args.top_k)
    return 0


def _cmd_fuse(_: argparse.Namespace) -> int:
    """Слияние BM25F с гео на локальном бенчмарке"""
    from avito_cg.cli.fuse import run

    run()
    return 0


def _cmd_rerank(args: argparse.Namespace) -> int:
    """Замерить переранжирование на локальном бенчмарке"""
    from avito_cg.cli.rerank import run

    run(donor=not args.no_donor, save=args.save)
    return 0


def _cmd_donor_set(args: argparse.Namespace) -> int:
    """Набрать донорские запросы для обучения переранжировщика"""
    from avito_cg.cli.rerank import build_donor

    build_donor(n_queries=args.n_queries, device=args.device)
    return 0


def _cmd_train_encoder(args: argparse.Namespace) -> int:
    """Дообучить би-энкодер на разрешённых парах"""
    from avito_cg.cli.encoder import run_training
    from avito_cg.train.biencoder import TrainingConfig

    run_training(
        TrainingConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            max_pairs=args.max_pairs,
            frozen_embeddings=args.freeze_embeddings,
        ),
        output=args.out,
    )
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    """Закодировать корпус дообученным энкодером"""
    from avito_cg.cli.encoder import run_encoding

    run_encoding(
        args.scope,
        model_path=args.model,
        output=args.out,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Одна подкоманда на шаг пайплайна, в порядке их выполнения

    Импорты внутри обработчиков, а не наверху модуля, намеренно: `avito-cg check-data`
    не должен тянуть torch и catboost ради проверки, что три parquet лежат на месте
    """
    parser = argparse.ArgumentParser(prog="avito-cg", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-data", help="проверить наличие исходных файлов")
    check.set_defaults(func=_cmd_check_data)

    eda = subparsers.add_parser("eda", help="разбор данных: таблицы и графики")
    eda.set_defaults(func=_cmd_eda)

    split = subparsers.add_parser("split", help="собрать локальный бенчмарк из train")
    split.add_argument(
        "--seen-share",
        type=float,
        default=0.096,
        help="доля корпуса, знакомая обучающей части; 1.0 это наивный сплит",
    )
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--no-sanity", action="store_true", help="пропустить контрольные модели")
    split.set_defaults(func=_cmd_split)

    baseline = subparsers.add_parser("baseline", help="BM25F на локальном бенчмарке")
    baseline.add_argument("--no-grid", action="store_true", help="пропустить подбор весов полей")
    baseline.set_defaults(func=_cmd_baseline)

    fuse = subparsers.add_parser("fuse", help="слияние лексики с гео на локальном бенчмарке")
    fuse.set_defaults(func=_cmd_fuse)

    donor = subparsers.add_parser("donor-set", help="набрать донорские запросы и закодировать их")
    donor.add_argument("--n-queries", type=int, default=40000, help="бюджет по числу запросов")
    donor.add_argument("--device", default="auto", help="auto, cuda или cpu")
    donor.set_defaults(func=_cmd_donor_set)

    rerank = subparsers.add_parser("rerank", help="переранжирование на локальном бенчмарке")
    rerank.add_argument("--no-donor", action="store_true", help="только отложенные запросы")
    rerank.add_argument(
        "--save", default=None, help="имя конфигурации из сводки, которую сохранить для ответа"
    )
    rerank.set_defaults(func=_cmd_rerank)

    encoder = subparsers.add_parser("train-encoder", help="дообучить би-энкодер, нужен GPU")
    encoder.add_argument("--model", default="intfloat/multilingual-e5-base")
    encoder.add_argument("--batch-size", type=int, default=96)
    encoder.add_argument("--epochs", type=int, default=2)
    encoder.add_argument("--lr", type=float, default=2e-5)
    encoder.add_argument("--device", default="auto", help="auto, cuda или cpu")
    encoder.add_argument("--max-pairs", type=int, default=None, help="обрезать выборку для пробы")
    encoder.add_argument(
        "--freeze-embeddings", action="store_true", help="не обучать таблицу эмбеддингов"
    )
    encoder.add_argument("--out", type=_resolve, default=None)
    encoder.set_defaults(func=_cmd_train_encoder)

    encode = subparsers.add_parser("encode-corpus", help="закодировать корпус энкодером")
    encode.add_argument("scope", choices=("local", "benchmark"))
    encode.add_argument("--model", type=_resolve, default=None)
    encode.add_argument("--device", default="auto")
    encode.add_argument("--batch-size", type=int, default=256)
    encode.add_argument("--out", type=_resolve, default=None)
    encode.set_defaults(func=_cmd_encode)

    answer = subparsers.add_parser("make-answer", help="собрать answer.csv по настоящему корпусу")
    answer.add_argument("--title", type=float, default=20.0, help="вес заголовка")
    answer.add_argument("--params", type=float, default=0.5, help="вес параметров")
    answer.add_argument("--description", type=float, default=1.0, help="вес описания")
    answer.add_argument("--with-filter", action="store_true", help="дописать фильтр к запросу")
    answer.add_argument("--no-geo", action="store_true", help="только лексика, без слияния с гео")
    answer.add_argument(
        "--fusion",
        choices=("raw", "normalized", "rrf"),
        default="normalized",
        help="как согласовывать масштабы лексики и гео",
    )
    answer.add_argument("--geo-weight", type=float, default=None, help="вес гео в скоре")
    answer.add_argument("--facet-weight", type=float, default=None, help="вес фасетов")
    answer.add_argument("--microcat-weight", type=float, default=None, help="вес микрокатегории")
    answer.add_argument("--dense-weight", type=float, default=None, help="вес плотного поиска")
    answer.add_argument(
        "--dense-depth", type=int, default=200, help="сколько кандидатов приводит плотный поиск"
    )
    answer.add_argument("--no-rerank", action="store_true", help="не применять переранжировщик")
    answer.add_argument("--out", type=_resolve, default=None)
    answer.set_defaults(func=_cmd_answer)

    errors = subparsers.add_parser("errors", help="разбор промахов на локальном бенчмарке")
    errors.add_argument("--top-k", type=int, default=50)
    errors.set_defaults(func=_cmd_errors)

    keys = subparsers.add_parser("param-keys", help="пересобрать словарь ключей параметров")
    keys.add_argument("--sample", type=int, default=20000, help="сколько объявлений смотреть")
    keys.add_argument("--seed", type=int, default=0)
    keys.add_argument("--write", action="store_true", help="перезаписать закреплённый словарь")
    keys.set_defaults(func=_cmd_param_keys)

    validate = subparsers.add_parser("validate-answer", help="проверить формат answer.csv")
    validate.add_argument("path", type=_resolve)
    validate.set_defaults(func=_cmd_validate_answer)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
