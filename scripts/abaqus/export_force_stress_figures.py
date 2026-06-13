#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ODB 提取后的反力、应力和接触压力表导出第 3 章补充图。

脚本绘制实际切削进程下的等效反力曲线、阶段均值，以及 Mises 应力和
接触压力的分位统计图。数值来源应由 `extract_force_stress_metrics.py`
生成，图件只负责展示，不重新读取 ODB。

Export supplementary force, stress, and contact-pressure figures from processed
CSV tables. ODB parsing belongs to `extract_force_stress_metrics.py`; this
script only renders the verified tabular results.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[2] / "output" / ".mplconfig")))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CASE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]
COLORS = {
    "S1": "#8c2d04",
    "S2": "#cc6b1f",
    "S3": "#016c59",
    "S4": "#1c9099",
    "S5": "#225ea8",
    "S6": "#6a51a3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export force/stress supplementary figures for V2 thesis.")
    parser.add_argument("--datadir", default=str(ROOT / "output" / "spreadsheet" / "abaqus"))
    parser.add_argument("--figdir", default=str(ROOT / "output" / "figures" / "abaqus_v2_extra"))
    return parser.parse_args()


def configure_matplotlib() -> None:
    installed = {font.name for font in fm.fontManager.ttflist}
    candidates = ["Times New Roman", "DejaVu Serif"]
    family = next((name for name in candidates if name in installed), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#d0d0d0",
            "savefig.facecolor": "white",
            "font.size": 11.0,
            "axes.titlesize": 12.4,
            "axes.labelsize": 11.6,
            "legend.fontsize": 10.2,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def rows_by_case(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result = {case: [] for case in CASE_ORDER}
    for row in rows:
        case = row["case"].upper()
        if case in result:
            result[case].append(row)
    return result


def case_labels(summary_rows: list[dict[str, str]]) -> list[str]:
    lookup = {row["case"].upper(): row for row in summary_rows}
    return [
        f"{case}\nR{float(lookup[case]['radius_mm']):.0f} t{float(lookup[case]['nominal_thickness_mm']):.1f}"
        for case in CASE_ORDER
    ]


def annotate_bars(
    ax: plt.Axes,
    bars,
    values: list[float],
    fmt: str,
    pad: float = 2.5,
    y_positions: list[float] | None = None,
) -> None:
    for index, (bar, value) in enumerate(zip(bars, values)):
        y_position = bar.get_height() if y_positions is None else y_positions[index]
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, y_position),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
        )


def binned_mean(xs: list[float], ys: list[float], bins: int = 28) -> tuple[list[float], list[float]]:
    bucket_x: list[list[float]] = [[] for _ in range(bins)]
    bucket_y: list[list[float]] = [[] for _ in range(bins)]
    for x_value, y_value in zip(xs, ys):
        if x_value < 0.0 or x_value > 1.0:
            continue
        index = min(bins - 1, max(0, int(x_value * bins)))
        bucket_x[index].append(x_value)
        bucket_y[index].append(y_value)
    out_x: list[float] = []
    out_y: list[float] = []
    for index in range(bins):
        if not bucket_y[index]:
            continue
        out_x.append(float(np.mean(bucket_x[index])))
        out_y.append(float(np.mean(bucket_y[index])))
    return out_x, out_y


def save_figure(fig: plt.Figure, figdir: Path, stem: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(figdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_force(force_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> plt.Figure:
    by_case = rows_by_case(force_rows)
    labels = case_labels(summary_rows)
    x = np.arange(len(CASE_ORDER))
    force_scale = 1.0e8

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.4), constrained_layout=True)

    ax = axes[0]
    for case in CASE_ORDER:
        rows = [row for row in by_case[case] if 0.0 <= as_float(row, "eta") <= 1.0]
        xs = [as_float(row, "eta") for row in rows]
        ys = [as_float(row, "force_mag") / force_scale for row in rows]
        bx, by = binned_mean(xs, ys)
        ax.plot(
            bx,
            by,
            color=COLORS[case],
            linewidth=2.0,
            alpha=0.92,
            label=case,
        )
    ax.set_title("(a) Binned mean equivalent cutting force")
    ax.set_xlabel("Actual cutting progress (eta)")
    ax.set_ylabel("Equivalent force (10$^8$ N)")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.78)
    ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.965), borderpad=0.35, handlelength=1.7)

    ax = axes[1]
    lookup = {row["case"].upper(): row for row in summary_rows}
    means = [float(lookup[case]["force_mean"]) / force_scale for case in CASE_ORDER]
    stds = [float(lookup[case]["force_std"]) / force_scale for case in CASE_ORDER]
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4.0,
        width=0.58,
        color=[COLORS[case] for case in CASE_ORDER],
        edgecolor="#333333",
        linewidth=0.7,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#444444"},
    )
    ax.set_title("(b) Mean equivalent force and temporal fluctuation")
    ax.set_ylabel("Equivalent force (10$^8$ N)")
    ax.set_xticks(x, labels)
    upper_caps = [m + s for m, s in zip(means, stds)]
    ax.set_ylim(0, max(upper_caps) * 1.34)
    ax.grid(axis="y", linestyle="--", alpha=0.78)
    annotate_bars(ax, bars, means, "{:.2f}", 4.0, y_positions=upper_caps)

    return fig


def plot_stress(stress_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> plt.Figure:
    by_case = rows_by_case(stress_rows)
    labels = case_labels(summary_rows)
    x = np.arange(len(CASE_ORDER))
    lookup = {row["case"].upper(): row for row in summary_rows}

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.4), constrained_layout=True)

    ax = axes[0]
    for case in CASE_ORDER:
        rows = sorted(by_case[case], key=lambda row: as_float(row, "eta"))
        ax.plot(
            [as_float(row, "eta") for row in rows],
            [as_float(row, "mises_p95") for row in rows],
            color=COLORS[case],
            marker="o",
            markersize=3.2,
            linewidth=1.75,
            alpha=0.95,
            label=case,
        )
    ax.set_title("(a) Evolution of the 95% Mises-stress quantile")
    ax.set_xlabel("Actual cutting progress (eta)")
    ax.set_ylabel("Mises stress (MPa)")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.78)
    ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.965), borderpad=0.35, handlelength=1.7)

    ax = axes[1]
    p95_means = [float(lookup[case]["mises_p95_mean"]) for case in CASE_ORDER]
    p95_stds = [float(lookup[case]["mises_p95_std"]) for case in CASE_ORDER]
    max_peaks = [float(lookup[case]["mises_max_peak"]) for case in CASE_ORDER]
    bars = ax.bar(
        x,
        p95_means,
        yerr=p95_stds,
        capsize=4.0,
        width=0.58,
        color=[COLORS[case] for case in CASE_ORDER],
        edgecolor="#333333",
        linewidth=0.7,
        alpha=0.88,
        label="95% quantile mean",
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#444444"},
    )
    ax.plot(
        x,
        max_peaks,
        color="#202020",
        marker="D",
        markersize=4.7,
        linewidth=1.8,
        label="Peak Mises stress",
    )
    ax.set_title("(b) Mean fluctuation and peak stress comparison")
    ax.set_ylabel("Mises stress (MPa)")
    ax.set_xticks(x, labels)
    upper_caps = [m + s for m, s in zip(p95_means, p95_stds)]
    ax.set_ylim(0, max(max(max_peaks), max(upper_caps)) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.78)
    ax.legend(loc="upper left", borderpad=0.35)
    annotate_bars(ax, bars, p95_means, "{:.0f}", 4.0, y_positions=upper_caps)

    return fig


def main() -> None:
    args = parse_args()
    datadir = Path(args.datadir)
    figdir = Path(args.figdir)
    configure_matplotlib()
    force_rows = read_csv_rows(datadir / "force_history_by_cut_progress.csv")
    stress_rows = read_csv_rows(datadir / "stress_contact_by_cut_progress.csv")
    summary_rows = read_csv_rows(datadir / "force_stress_summary.csv")
    save_figure(plot_force(force_rows, summary_rows), figdir, "fig03_15_force_response_combo")
    save_figure(plot_stress(stress_rows, summary_rows), figdir, "fig03_16_stress_side_evidence_combo")
    print(f"Saved force/stress figures to: {figdir}")


if __name__ == "__main__":
    main()
