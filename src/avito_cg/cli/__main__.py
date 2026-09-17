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
        print("не найдены файлы:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
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
    from avito_cg.cli.answer import run
    from avito_cg.index.lexical import Field

    run(
        fields=[
            Field("title", args.title, 0.6),
            Field("params", args.params, 0.75),
            Field("description", args.description, 0.75),
        ],
        output=args.out,
        with_filter=args.with_filter,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
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

    answer = subparsers.add_parser("make-answer", help="собрать answer.csv по настоящему корпусу")
    answer.add_argument("--title", type=float, default=20.0, help="вес заголовка")
    answer.add_argument("--params", type=float, default=0.5, help="вес параметров")
    answer.add_argument("--description", type=float, default=1.0, help="вес описания")
    answer.add_argument("--with-filter", action="store_true", help="дописать фильтр к запросу")
    answer.add_argument("--out", type=_resolve, default=None)
    answer.set_defaults(func=_cmd_answer)

    validate = subparsers.add_parser("validate-answer", help="проверить формат answer.csv")
    validate.add_argument("path", type=_resolve)
    validate.set_defaults(func=_cmd_validate_answer)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
