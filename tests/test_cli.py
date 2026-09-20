"""CLI и документация не должны расходиться

Расхождение здесь стоит дорого и не ловится глазами: README обещал цепочку из четырёх
команд, а собрать отправленный ответ можно было только девятью, и `make-answer` в такой
усечённой цепочке молча отдавал версию на три с лишним пункта хуже. Поэтому связь
«что есть в CLI» и «что написано в документации» проверяется тестом.
"""

import re
from pathlib import Path

import pytest

from avito_cg.cli.__main__ import build_parser
from avito_cg.config import PATHS

DOCS = [PATHS.root / "README.md", *sorted((PATHS.root / "docs").glob("*.md"))]


def commands() -> set[str]:
    subparsers = build_parser()._subparsers._group_actions[0]
    return set(subparsers.choices)


def mentioned() -> set[str]:
    found: set[str] = set()
    for doc in DOCS:
        found.update(re.findall(r"avito-cg ([a-z][a-z-]+)", doc.read_text(encoding="utf-8")))
    return found


# команды, у которых есть обязательный позиционный аргумент
REQUIRED_ARGS = {"encode-corpus": ["local"], "validate-answer": ["answer.csv"]}


def test_every_command_has_a_handler():
    """Подкоманда без func падает не при разборе аргументов, а при запуске"""
    parser = build_parser()
    for name in commands():
        args = parser.parse_args([name, *REQUIRED_ARGS.get(name, [])])
        assert callable(args.func), name


def test_documentation_mentions_only_real_commands():
    assert mentioned() - commands() == set()


def test_every_command_is_documented():
    assert commands() - mentioned() == set()


def anchors(doc: Path) -> set[str]:
    """Якоря, которые GitHub делает из заголовков: нижний регистр, пробелы в дефисы"""
    found = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", doc.read_text(encoding="utf-8"), re.M):
        clean = re.sub(r"[^\w\s-]", "", heading.replace("`", "")).strip().lower()
        found.add(re.sub(r"\s+", "-", clean))
    return found


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_relative_links_resolve(doc: Path):
    """И файл на месте, и якорь внутри него существует

    Ссылка на несуществующий якорь молча приводит проверяющего в начало файла
    вместо нужного раздела, и глазами это не ловится
    """
    broken = []
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        destination = doc if not path else doc.parent / path
        if not destination.exists():
            broken.append(f"[{label}]({target}): нет файла")
        elif anchor and anchor not in anchors(destination):
            broken.append(f"[{label}]({target}): нет якоря")
    assert not broken, f"{doc.name}: {broken}"


def test_no_flag_duplicates_a_config_constant():
    """Умолчание флага не должно дублировать константу из config

    Именно на этом решение один раз молча собралось не в той конфигурации: константа
    переехала в config, а у флага осталось своё старое значение по умолчанию, и оно
    перебило конфиг. Отпечаток при этом совпал со старым кэшем, признаки приехали
    от предыдущей конфигурации, а модель применилась новая. Ничего не упало.
    """
    from avito_cg import config

    numeric = {
        name: value
        for name, value in vars(config).items()
        if isinstance(value, int) and not isinstance(value, bool) and name.isupper()
    }
    defaults = build_parser().parse_args(["make-answer"])
    clashing = [
        f"--{key.replace('_', '-')}={value} дублирует config.{name}"
        for key, value in vars(defaults).items()
        if isinstance(value, int) and not isinstance(value, bool)
        for name, constant in numeric.items()
        if constant == value and key.replace("_", "") in name.lower().replace("_", "")
    ]
    assert not clashing, clashing
