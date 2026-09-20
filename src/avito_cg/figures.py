"""Графики для разбора данных

Ничего не считают, только рисуют то, что посчитал analysis.py. Сохраняю в PNG,
чтобы вставлять в докcы: ноутбук с интерактивом на гитхабе всё равно не открывается
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ACCENT = "#0a7cff"
MUTED = "#9aa4b2"
WARN = "#e8710a"


def _setup() -> None:
    """Общий стиль для всех графиков: одинаковые оси, сетка и размер шрифта"""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
        }
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def filter_ceiling(report: pd.DataFrame, path: Path, *, top_k: int = 50) -> Path:
    """Чем платим за сокращение корпуса

    По горизонтали сколько кандидатов остаётся, по вертикали сколько полноты
    при этом теряется. Вертикаль на 50 это размер ответа: всё, что левее,
    целиком помещается в ответ без всякого ранжирования
    """
    _setup()
    ordered = report.sort_values("кандидатов, медиана")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = ordered["кандидатов, медиана"].to_numpy(dtype=float)
    y = ordered["потолок полноты"].to_numpy(dtype=float)
    ax.scatter(x, y, s=90, color=ACCENT, zorder=3)
    for index, (label, xi, yi) in enumerate(zip(ordered["фильтр"], x, y, strict=True)):
        ax.annotate(
            f"{label}\n{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(10, 14 if index % 2 == 0 else -34),
            fontsize=9,
            color="#333",
        )
    ax.axvline(top_k, color=WARN, linestyle="--", linewidth=1.2)
    ax.text(top_k * 1.12, 0.76, f"размер ответа = {top_k}", color=WARN, fontsize=9, rotation=90)
    ax.set_xscale("log")
    ax.set_xlim(top_k / 2, len(report) and x.max() * 8)
    ax.set_xlabel("кандидатов на запрос, медиана")
    ax.set_ylabel("потолок Recall@50")
    ax.set_ylim(0.72, 1.06)
    ax.set_title("Структурные фильтры: полнота против числа кандидатов")
    return _save(fig, path)


def lexical_coverage(report: pd.DataFrame, path: Path) -> Path:
    """Сколько токенов запроса вообще встречается в тексте выбранного объявления"""
    _setup()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = report["поле"].tolist()
    zero = report["нет общих токенов"].to_numpy()
    full = report["покрыт полностью"].to_numpy()
    partial = 1 - zero - full
    ax.barh(labels, zero, color=WARN, label="нет общих токенов")
    ax.barh(labels, partial, left=zero, color=MUTED, label="частичное пересечение")
    ax.barh(labels, full, left=zero + partial, color=ACCENT, label="все токены запроса на месте")
    for index, value in enumerate(zero):
        ax.text(
            value / 2, index, f"{value:.1%}", va="center", ha="center", fontsize=9, color="white"
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("доля пар «запрос - выбранное объявление»")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Лексический разрыв: что теряет поиск по словам")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def distance_recall(report: pd.DataFrame, path: Path, *, exact_match: float | None = None) -> Path:
    """Полнота гео-фильтра в зависимости от радиуса"""
    _setup()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        report["порог, км"],
        report["доля пар внутри"],
        "o-",
        color=ACCENT,
        linewidth=1.8,
        markersize=6,
    )
    if exact_match is not None:
        ax.axhline(exact_match, color=WARN, linestyle="--", linewidth=1.2)
        ax.text(
            1.2,
            exact_match + 0.012,
            f"точное совпадение location_id: {exact_match:.3f}",
            color=WARN,
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("радиус вокруг центра поисковой локации, км")
    ax.set_ylabel("доля выбранных объявлений внутри")
    ax.set_title("География: расстояние работает лучше, чем совпадение идентификатора")
    return _save(fig, path)


def positives_per_query(counts: pd.Series, path: Path, *, max_shown: int = 6) -> Path:
    """Сколько релевантных объявлений приходится на запрос"""
    _setup()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    shares = [float((counts == n).mean()) for n in range(1, max_shown)]
    shares.append(float((counts >= max_shown).mean()))
    labels = [str(n) for n in range(1, max_shown)] + [f"{max_shown}+"]
    bars = ax.bar(labels, shares, color=ACCENT)
    bars[0].set_color(WARN)
    for bar, share in zip(bars, shares, strict=True):
        if share > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                share + 0.012,
                f"{share:.1%}",
                ha="center",
                fontsize=9,
            )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("релевантных объявлений на запрос")
    ax.set_ylabel("доля запросов")
    ax.set_title("У большинства запросов ровно один правильный ответ")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def query_length_shift(train_lengths: np.ndarray, bench_lengths: np.ndarray, path: Path) -> Path:
    """Запросы бенчмарка длиннее обучающих, и это придётся учесть в валидации"""
    _setup()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0.5, 11.5, 1).tolist()
    ax.hist(
        np.clip(train_lengths, 0, 10),
        bins=bins,
        density=True,
        alpha=0.65,
        color=MUTED,
        label="обучение",
    )
    ax.hist(
        np.clip(bench_lengths, 0, 10),
        bins=bins,
        density=True,
        alpha=0.65,
        color=ACCENT,
        label="бенчмарк",
    )
    ax.axvline(train_lengths.mean(), color=MUTED, linestyle="--", linewidth=1.3)
    ax.axvline(bench_lengths.mean(), color=ACCENT, linestyle="--", linewidth=1.3)
    ax.set_xlabel("слов в запросе")
    ax.set_ylabel("плотность")
    ax.legend(frameon=False)
    ax.set_title(
        f"Сдвиг распределения: {train_lengths.mean():.2f} слова в обучении против "
        f"{bench_lengths.mean():.2f} в бенчмарке"
    )
    return _save(fig, path)


def title_duplicates(counts: np.ndarray, path: Path) -> Path:
    """Сколько объявлений делят заголовок с другими"""
    _setup()
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = np.sort(counts)
    ax.plot(np.arange(1, ordered.size + 1) / ordered.size, ordered, color=ACCENT, linewidth=1.8)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("доля объявлений корпуса")
    ax.set_ylabel("объявлений с таким же заголовком")
    ax.set_title("Половина корпуса неразличима по заголовку")
    return _save(fig, path)


def microcat_topk(report: pd.DataFrame, path: Path) -> Path:
    """Предсказуемость микрокатегории по одному тексту запроса"""
    _setup()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(report["k"], report["точность top-k"], "o-", color=ACCENT, linewidth=1.8, markersize=7)
    for k, value in zip(report["k"], report["точность top-k"], strict=True):
        ax.annotate(
            f"{value:.3f}",
            (k, value),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=9,
            ha="center",
        )
    ax.set_xscale("log")
    ax.set_xticks(report["k"].tolist())
    ax.set_xticklabels(report["k"].tolist())
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("сколько микрокатегорий оставляем")
    ax.set_ylabel("доля попаданий")
    ax.set_title("Микрокатегория предсказывается по тексту запроса")
    return _save(fig, path)


def candidates_after_filters(sizes: np.ndarray, path: Path, *, top_k: int = 50) -> Path:
    """Сколько кандидатов остаётся после всех структурных фильтров"""
    _setup()
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = np.sort(np.maximum(sizes, 1))
    share = np.arange(1, ordered.size + 1) / ordered.size
    ax.plot(ordered, share, color=ACCENT, linewidth=1.8)
    ax.axvline(top_k, color=WARN, linestyle="--", linewidth=1.2)
    reached = float((sizes <= top_k).mean())
    ax.plot([top_k], [reached], "o", color=WARN, markersize=8)
    ax.annotate(
        f"{reached:.1%} запросов помещаются\nв ответ целиком",
        (top_k, reached),
        textcoords="offset points",
        xytext=(14, -6),
        fontsize=9,
        color=WARN,
    )
    ax.set_xscale("log")
    ax.set_xlabel("кандидатов после гео, фасетов и микрокатегории")
    ax.set_ylabel("доля запросов бенчмарка")
    ax.set_ylim(0, 1.02)
    ax.set_title("После структурных фильтров корпус сжимается до десятков объявлений")
    return _save(fig, path)


def location_sizes(counts: np.ndarray, path: Path) -> Path:
    """Насколько неравномерно объявления разложены по локациям"""
    _setup()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(counts, bins=np.logspace(0, np.log10(counts.max()), 40).tolist(), color=ACCENT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("объявлений в локации")
    ax.set_ylabel("локаций")
    ax.set_title(
        f"Корпус по локациям: медиана {np.median(counts):.0f}, максимум {counts.max():.0f}"
    )
    return _save(fig, path)
