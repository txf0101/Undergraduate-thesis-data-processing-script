#!/usr/bin/env python3
"""
导出第 3 章补充判读图，用于说明峰值位移、壁厚保持率、响应时序和测线梯度。

输入为 `export_thesis_english_figures.py` 写出的算例与测线指标表。图件
用于把主要图中较分散的数值指标压缩成对比视图，便于检查曲率半径、
名义壁厚和轴向位置对风险指标的影响。

Export supplementary Abaqus figures from the case and line metric tables. The
plots condense peak displacement, thickness retention, timing differences, and
line-gradient indicators into compact comparison views.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[2] / "output" / ".mplconfig")))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CASE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export additional V2 thesis figures for the Abaqus chapter.")
    parser.add_argument(
        "--datadir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="Directory containing processed Abaqus CSV/JSON outputs.",
    )
    parser.add_argument(
        "--figdir",
        default=str(ROOT / "output" / "figures" / "abaqus_v2_extra"),
        help="Directory for the additional V2 figures.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    installed = {font.name for font in fm.fontManager.ttflist}
    font_candidates = ["Times New Roman", "DejaVu Serif"]
    family = next((name for name in font_candidates if name in installed), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 0.8,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#d0d0d0",
            "savefig.facecolor": "white",
            "font.size": 11.0,
            "axes.titlesize": 12.4,
            "axes.labelsize": 11.6,
            "legend.fontsize": 10.4,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
        }
    )


def load_inputs(datadir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_metrics = pd.read_csv(datadir / "thesis_case_metrics.csv")
    line_metrics = pd.read_csv(datadir / "thesis_line_metrics.csv")
    case_metrics["case"] = case_metrics["case"].astype(str).str.upper()
    line_metrics["case"] = line_metrics["case"].astype(str).str.upper()
    case_metrics = case_metrics.set_index("case").loc[CASE_ORDER].reset_index()
    line_metrics = line_metrics.set_index("case").loc[CASE_ORDER].reset_index()
    return case_metrics, line_metrics


def case_labels(metrics: pd.DataFrame) -> list[str]:
    return [
        f"{row.case}\nR{row.radius_mm:.0f} t{row.nominal_thickness_mm:.1f}"
        for row in metrics.itertuples(index=False)
    ]


def annotate_bars(axis: plt.Axes, bars, fmt: str = "{:.2f}", pad: float = 2.0) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#333333",
        )


def save_figure(fig: plt.Figure, figdir: Path, stem: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(figdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_response_margin(metrics: pd.DataFrame) -> plt.Figure:
    labels = case_labels(metrics)
    x = np.arange(len(metrics))
    colors = {"bar": "#c56a2d", "bar2": "#7c4d8a", "line": "#1f6f9f", "line2": "#237a57"}

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.6), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(
        x,
        metrics["peak_displacement_mm"],
        width=0.56,
        color=colors["bar"],
        edgecolor="#6f3b1c",
        linewidth=0.7,
        label="Peak mid-surface displacement",
    )
    ax.set_title("(a) Displacement and final thickness")
    ax.set_ylabel("Displacement (mm)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(metrics["peak_displacement_mm"]) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.85)
    annotate_bars(ax, bars, "{:.3f}", 3.0)

    ax_r = ax.twinx()
    ax_r.plot(
        x,
        metrics["end_frame_min_thickness_mm"],
        color=colors["line"],
        marker="o",
        markersize=5.5,
        linewidth=2.0,
        label="Cut-completion minimum true thickness",
    )
    ax_r.set_ylabel("True thickness (mm)")
    ax_r.set_ylim(0, max(metrics["end_frame_min_thickness_mm"]) * 1.18)
    handles, names = ax.get_legend_handles_labels()
    handles_r, names_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, names + names_r, loc="upper left", borderpad=0.4)

    ax = axes[1]
    bars = ax.bar(
        x,
        metrics["minimum_true_thickness_mm"],
        width=0.56,
        color=colors["bar2"],
        edgecolor="#51325d",
        linewidth=0.7,
        label="Whole-process minimum true thickness",
    )
    ax.set_title("(b) Minimum thickness and retention ratio")
    ax.set_ylabel("True thickness (mm)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(metrics["minimum_true_thickness_mm"]) * 1.30)
    ax.grid(axis="y", linestyle="--", alpha=0.85)
    annotate_bars(ax, bars, "{:.3f}", 3.0)

    retention_pct = metrics["end_frame_retention_ratio"] * 100.0
    ax_r = ax.twinx()
    ax_r.plot(
        x,
        retention_pct,
        color=colors["line2"],
        marker="D",
        markersize=5.0,
        linewidth=2.0,
        label="Cut-completion retention ratio",
    )
    ax_r.set_ylabel("Retention ratio (%)")
    ax_r.set_ylim(0, max(retention_pct) * 1.22)
    handles, names = ax.get_legend_handles_labels()
    handles_r, names_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, names + names_r, loc="upper left", borderpad=0.4)

    return fig


def plot_timing_recovery(metrics: pd.DataFrame) -> plt.Figure:
    labels = case_labels(metrics)
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.6), constrained_layout=True)

    ax = axes[0]
    series = [
        ("Worst-thickness frame", metrics["worst_thickness_frame_raw"], "#b13d3d", "o"),
        ("Peak-displacement frame", metrics["peak_displacement_frame_raw"], "#2b6f9f", "s"),
        ("Cut-completion frame", metrics["end_frame_raw"], "#4f7c3a", "^"),
    ]
    for name, values, color, marker in series:
        ax.plot(x, values, color=color, marker=marker, markersize=5.2, linewidth=2.0, label=name)
    ax.set_title("(a) Characteristic frames")
    ax.set_ylabel("Raw frame")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 212)
    ax.grid(axis="y", linestyle="--", alpha=0.85)
    ax.legend(loc="lower right", ncol=1, borderpad=0.4)

    ax = axes[1]
    lead_frames = metrics["peak_displacement_frame_raw"] - metrics["worst_thickness_frame_raw"]
    recovery_mm = metrics["end_frame_min_thickness_mm"] - metrics["minimum_true_thickness_mm"]
    bars = ax.bar(
        x,
        lead_frames,
        width=0.56,
        color="#d0a43b",
        edgecolor="#80641f",
        linewidth=0.7,
        label="Worst-thickness lead over peak displacement",
    )
    ax.set_title("(b) Lead frame and thickness recovery")
    ax.set_ylabel("Lead frame")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(lead_frames) * 1.25 if max(lead_frames) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.85)
    annotate_bars(ax, bars, "{:.0f}", 3.0)

    ax_r = ax.twinx()
    ax_r.plot(
        x,
        recovery_mm,
        color="#7b4f9d",
        marker="o",
        markersize=5.4,
        linewidth=2.0,
        label="Thickness recovery at cut completion",
    )
    ax_r.set_ylabel("Recovery (mm)")
    ax_r.set_ylim(0, max(recovery_mm) * 1.28)
    handles, names = ax.get_legend_handles_labels()
    handles_r, names_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, names + names_r, loc="upper right", borderpad=0.4)

    return fig


def plot_line_gradient(metrics: pd.DataFrame, line_metrics: pd.DataFrame) -> plt.Figure:
    labels = case_labels(metrics)
    x = np.arange(len(metrics))
    width = 0.34

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.6), constrained_layout=True)

    ax = axes[0]
    b1 = ax.bar(
        x - width / 2,
        line_metrics["line_01_peak_mm"],
        width=width,
        color="#b4553d",
        edgecolor="#743626",
        linewidth=0.7,
        label="Line 01 peak",
    )
    b2 = ax.bar(
        x + width / 2,
        line_metrics["line_40_peak_mm"],
        width=width,
        color="#5478a6",
        edgecolor="#334b68",
        linewidth=0.7,
        label="Line 40 peak",
    )
    ax.set_title("(a) Free-edge and constrained-side peaks")
    ax.set_ylabel("Displacement (mm)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(line_metrics["line_01_peak_mm"].max(), line_metrics["line_40_peak_mm"].max()) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.85)
    annotate_bars(ax, b1, "{:.3f}", 3.0)
    annotate_bars(ax, b2, "{:.3f}", 3.0)
    ax.legend(loc="upper left", borderpad=0.4)

    ax = axes[1]
    bars = ax.bar(
        x,
        line_metrics["attenuation_40_vs_01_pct"],
        width=0.56,
        color="#5e8c61",
        edgecolor="#36543a",
        linewidth=0.7,
        label="Line 40 attenuation relative to Line 01",
    )
    ax.set_title("(b) Attenuation and case peak")
    ax.set_ylabel("Attenuation (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(line_metrics["attenuation_40_vs_01_pct"]) * 1.24)
    ax.grid(axis="y", linestyle="--", alpha=0.85)
    annotate_bars(ax, bars, "{:.1f}", 3.0)

    ax_r = ax.twinx()
    ax_r.plot(
        x,
        metrics["peak_displacement_mm"],
        color="#2a5d96",
        marker="o",
        markersize=5.4,
        linewidth=2.0,
        label="Case peak displacement",
    )
    ax_r.set_ylabel("Peak displacement (mm)")
    ax_r.set_ylim(0, max(metrics["peak_displacement_mm"]) * 1.28)
    handles, names = ax.get_legend_handles_labels()
    handles_r, names_r = ax_r.get_legend_handles_labels()
    ax.legend(handles + handles_r, names + names_r, loc="upper left", borderpad=0.4)

    return fig


def write_summary(figdir: Path, metrics: pd.DataFrame, line_metrics: pd.DataFrame) -> None:
    summary = {
        "max_peak_displacement_case": str(metrics.loc[metrics["peak_displacement_mm"].idxmax(), "case"]),
        "max_peak_displacement_mm": float(metrics["peak_displacement_mm"].max()),
        "min_process_thickness_case": str(metrics.loc[metrics["minimum_true_thickness_mm"].idxmin(), "case"]),
        "min_process_thickness_mm": float(metrics["minimum_true_thickness_mm"].min()),
        "max_end_retention_case": str(metrics.loc[metrics["end_frame_retention_ratio"].idxmax(), "case"]),
        "max_end_retention_pct": float(metrics["end_frame_retention_ratio"].max() * 100.0),
        "max_lead_case": str(
            metrics.loc[
                (metrics["peak_displacement_frame_raw"] - metrics["worst_thickness_frame_raw"]).idxmax(),
                "case",
            ]
        ),
        "max_lead_frames": int((metrics["peak_displacement_frame_raw"] - metrics["worst_thickness_frame_raw"]).max()),
        "max_gradient_case": str(line_metrics.loc[line_metrics["attenuation_40_vs_01_pct"].idxmax(), "case"]),
        "max_gradient_pct": float(line_metrics["attenuation_40_vs_01_pct"].max()),
    }
    (figdir / "v2_extra_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    datadir = Path(args.datadir)
    figdir = Path(args.figdir)
    configure_matplotlib()
    metrics, line_metrics = load_inputs(datadir)
    save_figure(plot_response_margin(metrics), figdir, "fig03_12_response_margin_combo")
    save_figure(plot_timing_recovery(metrics), figdir, "fig03_13_timing_recovery_combo")
    save_figure(plot_line_gradient(metrics, line_metrics), figdir, "fig03_14_line_gradient_combo")
    write_summary(figdir, metrics, line_metrics)
    print(f"Saved additional V2 figures to: {figdir}")


if __name__ == "__main__":
    main()
