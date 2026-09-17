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
    from avito_cg.eval.submission import read_submission, validate_submission

    frame = read_submission(args.path)
    queries = load_benchmark_queries(columns=["query_id"])
    items = load_benchmark_items(columns=["item_id"])
    problems = validate_submission(
        frame,
        expected_query_ids=queries["query_id"].astype(str).tolist(),
        corpus_item_ids=set(items["item_id"].astype(str)),
    )
    if not problems:
        print(f"{args.path}: формат в порядке, {len(frame)} строк")
        return 0
    print(f"{args.path}: найдены проблемы", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avito-cg", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-data", help="проверить наличие исходных файлов")
    check.set_defaults(func=_cmd_check_data)

    validate = subparsers.add_parser("validate-answer", help="проверить формат answer.csv")
    validate.add_argument("path", type=_resolve)
    validate.set_defaults(func=_cmd_validate_answer)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
