#!/usr/bin/env python3
"""
导出第 3 章最终英文图件，并写出图件依赖的算例指标表。

脚本读取位移、实际切削进程、真壁厚和切削完成帧厚度场等处理结果，
生成响应窗口、实际进程对齐、峰值位移、壁厚演化、耦合散点和终止帧
场图。`thesis_case_metrics.csv/json` 与 `thesis_line_metrics.csv/json`
由本脚本同步写出，供后续补充图和正文整理使用。

Export the final English figure set for the Abaqus chapter and write the
case/line metric tables consumed by downstream figure and document scripts.
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
import yaml


ROOT = Path(__file__).resolve().parents[2]
CASE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]
LINE_ORDER = ["01", "10", "20", "30", "40"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export English thesis figures for the Abaqus section.")
    parser.add_argument(
        "--datadir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="Directory that stores processed Abaqus spreadsheets.",
    )
    parser.add_argument(
        "--figdir",
        default=str(ROOT / "output" / "figures" / "abaqus_thesis_en"),
        help="Directory for exported English figures.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "scripts" / "abaqus" / "case_config.yml"),
        help="Case configuration file path.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    installed = {font.name for font in fm.fontManager.ttflist}
    family = ["Times New Roman"] if "Times New Roman" in installed else ["DejaVu Serif"]
    mpl.rcParams["font.family"] = family
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["axes.edgecolor"] = "#222222"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["grid.color"] = "#d9d9d9"
    mpl.rcParams["grid.linewidth"] = 0.6
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["font.size"] = 12.0
    mpl.rcParams["axes.titlesize"] = 14.0
    mpl.rcParams["axes.labelsize"] = 12.5
    mpl.rcParams["legend.fontsize"] = 11.5
    mpl.rcParams["xtick.labelsize"] = 11.5
    mpl.rcParams["ytick.labelsize"] = 11.5


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def case_colors() -> dict[str, str]:
    return {
        "S1": "#8c2d04",
        "S2": "#cc4c02",
        "S3": "#016c59",
        "S4": "#1c9099",
        "S5": "#225ea8",
        "S6": "#253494",
    }


def thickness_colors() -> dict[float, str]:
    return {1.0: "#c0504d", 1.5: "#4f81bd"}


def load_inputs(datadir: Path) -> dict[str, object]:
    inputs: dict[str, object] = {
        "raw_metrics": pd.read_csv(datadir / "raw_displacement_metrics.csv.gz"),
        "aligned_energy": pd.read_csv(datadir / "aligned_energy_by_cut_progress.csv"),
        "aligned_line_history": pd.read_csv(datadir / "aligned_line_history_by_cut_progress.csv"),
        "ks_ke": pd.read_csv(datadir / "ks_ke_summary.csv"),
        "thickness_summary": pd.read_csv(datadir / "true_thickness_summary_all_cases.csv"),
        "aligned_thickness": pd.read_csv(datadir / "aligned_true_thickness_by_cut_progress.csv"),
        "thickness_end_field": pd.read_csv(datadir / "true_thickness_end_field.csv.gz"),
        "thickness_key_metrics": json.loads((datadir / "true_thickness_key_metrics.json").read_text(encoding="utf-8")),
        "cut_milestones": pd.read_csv(datadir / "cut_progress_milestones.csv"),
    }
    for key in [
        "raw_metrics",
        "aligned_energy",
        "aligned_line_history",
        "ks_ke",
        "thickness_summary",
        "aligned_thickness",
        "thickness_end_field",
        "cut_milestones",
    ]:
        frame = inputs[key]
        if isinstance(frame, pd.DataFrame) and "case" in frame.columns:
            frame["case"] = frame["case"].astype(str).str.upper()
        if isinstance(frame, pd.DataFrame) and "line_id" in frame.columns:
            frame["line_id"] = frame["line_id"].astype(str).str.zfill(2)
    return inputs


def save_figure(figure: plt.Figure, figdir: Path, stem: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(figdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def build_case_metrics(inputs: dict[str, object], config: dict) -> pd.DataFrame:
    raw_metrics = inputs["raw_metrics"]
    ks_ke = inputs["ks_ke"]
    cut_milestones = inputs["cut_milestones"]
    thickness_key_metrics = inputs["thickness_key_metrics"]["cases"]

    peak_idx = raw_metrics.groupby("case")["w_peak_abs_mm"].idxmax()
    peak_disp = raw_metrics.loc[
        peak_idx,
        ["case", "line_id", "frame_raw", "w_peak_abs_mm", "x_at_w_peak_mm"],
    ].rename(
        columns={
            "line_id": "peak_line_id",
            "frame_raw": "peak_displacement_frame_raw",
            "w_peak_abs_mm": "peak_displacement_mm",
            "x_at_w_peak_mm": "peak_x_mm",
        }
    )

    metric_rows: list[dict[str, float | int | str]] = []
    ks_lookup = ks_ke.set_index("case").to_dict("index")
    cut_lookup = cut_milestones.set_index("case").to_dict("index")
    for case_id in CASE_ORDER:
        case_cfg = config["cases"][case_id]
        peak_row = peak_disp[peak_disp["case"] == case_id].iloc[0].to_dict()
        thickness_metrics = thickness_key_metrics[case_id]
        metric_rows.append(
            {
                "case": case_id,
                "radius_mm": float(case_cfg["radius_mm"]),
                "nominal_thickness_mm": float(case_cfg["nominal_thickness_mm"]),
                "speed_mm_s": float(case_cfg["speed_mm_s"]),
                "ks_raw": int(ks_lookup[case_id]["ks_raw"]),
                "ke_raw": int(ks_lookup[case_id]["ke_raw"]),
                "cut_start_frame_raw": int(cut_lookup[case_id]["cut_start_frame_raw"]),
                "cut_q1_frame_raw": int(cut_lookup[case_id]["cut_q1_frame_raw"]),
                "cut_q2_frame_raw": int(cut_lookup[case_id]["cut_q2_frame_raw"]),
                "cut_q3_frame_raw": int(cut_lookup[case_id]["cut_q3_frame_raw"]),
                "cut_done_frame_raw": int(cut_lookup[case_id]["cut_done_frame_raw"]),
                "analysis_end_frame_raw": int(cut_lookup[case_id]["analysis_end_frame_raw"]),
                "peak_energy_frame_raw": int(ks_lookup[case_id]["peak_frame_raw"]),
                "peak_displacement_frame_raw": int(peak_row["peak_displacement_frame_raw"]),
                "peak_displacement_mm": float(peak_row["peak_displacement_mm"]),
                "peak_line_id": str(peak_row["peak_line_id"]),
                "peak_x_mm": float(peak_row["peak_x_mm"]),
                "worst_thickness_frame_raw": int(thickness_metrics["worst_frame_raw"]),
                "minimum_true_thickness_mm": float(thickness_metrics["minimum_true_thickness_mm"]),
                "maximum_thinning_mm": float(thickness_metrics["maximum_thinning_mm"]),
                "end_frame_raw": int(thickness_metrics["end_frame_raw"]),
                "end_frame_min_thickness_mm": float(thickness_metrics["end_frame_min_thickness_mm"]),
                "end_frame_mean_thickness_mm": float(thickness_metrics["end_frame_mean_thickness_mm"]),
                "end_frame_retention_ratio": float(
                    thickness_metrics["end_frame_min_thickness_mm"] / case_cfg["nominal_thickness_mm"]
                ),
                "min_theta_deg": float(thickness_metrics["minimum_location"]["theta_deg"]),
                "min_axial_mm": float(thickness_metrics["minimum_location"]["axial_mm"]),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("case").reset_index(drop=True)
    return metrics


def write_case_metrics(datadir: Path, metrics: pd.DataFrame) -> None:
    metrics.to_csv(datadir / "thesis_case_metrics.csv", index=False, encoding="utf-8-sig")
    (datadir / "thesis_case_metrics.json").write_text(
        metrics.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_line_metrics(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    line_peak = (
        raw_metrics.groupby(["case", "line_id"], as_index=False)
        .agg(peak_w_mm=("w_peak_abs_mm", "max"))
        .sort_values(["case", "line_id"])
    )
    pivot = line_peak.pivot(index="case", columns="line_id", values="peak_w_mm").reindex(index=CASE_ORDER, columns=LINE_ORDER)
    rows: list[dict[str, float | str]] = []
    for case_id in CASE_ORDER:
        row = {
            "case": case_id,
            "line_01_peak_mm": float(pivot.loc[case_id, "01"]),
            "line_10_peak_mm": float(pivot.loc[case_id, "10"]),
            "line_20_peak_mm": float(pivot.loc[case_id, "20"]),
            "line_30_peak_mm": float(pivot.loc[case_id, "30"]),
            "line_40_peak_mm": float(pivot.loc[case_id, "40"]),
            "attenuation_40_vs_01_pct": float((1.0 - pivot.loc[case_id, "40"] / pivot.loc[case_id, "01"]) * 100.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_line_metrics(datadir: Path, line_metrics: pd.DataFrame) -> None:
    line_metrics.to_csv(datadir / "thesis_line_metrics.csv", index=False, encoding="utf-8-sig")
    (datadir / "thesis_line_metrics.json").write_text(
        line_metrics.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_figure_notes(figdir: Path) -> None:
    notes = [
        "# Chapter 3 Figure Notes",
        "",
        "- `fig03_01_response_window_en`: Use this figure to explain why direct comparison by raw frame can bias the conclusion.",
        "- `fig03_02_alignment_compare_en`: Show the necessity of mapping all cases onto the actual cutting progress defined by STATUS.",
        "- `fig03_03_line_history_en`: Highlight the response gradient from the free edge toward the constrained edge under the actual cutting-progress axis.",
        "- `fig03_04_peak_heatmap_en`: Compress the case-line interaction into one compact view for quick comparison.",
        "- `fig03_05_factor_interaction_en`: Present the main and interaction effects of radius and wall thickness using cut-completion thickness metrics.",
        "- `fig03_06_event_timeline_en`: Compare the actual cutting window, the main response window, peak displacement, and worst thinning.",
        "- `fig03_07_true_thickness_evolution_en`: Track the minimum, mean, and upper-envelope true thickness during the actual cutting stage.",
        "- `fig03_08_coupling_scatter_en`: Explain the coupling between global deformation amplitude and thickness risk.",
        "- `fig03_09_end_field_heatmaps_en`: Locate thickness-sensitive regions at the actual cutting-completion frame.",
        "- `fig03_10_odb_global_placeholder_en` to `fig03_12_odb_thickness_placeholder_en`: Replace these placeholders with ODB screenshots after visual export.",
    ]
    (figdir / "figure_notes.md").write_text("\n".join(notes), encoding="utf-8")


def plot_fig03_01_response_window(raw_metrics: pd.DataFrame, ks_ke: pd.DataFrame) -> plt.Figure:
    case_energy = (
        raw_metrics.groupby(["case", "frame_raw"], as_index=False)
        .agg(energy_l1_mm=("energy_l1_mm", "first"), energy_smooth_mm=("energy_smooth_mm", "first"))
        .sort_values(["case", "frame_raw"])
    )
    summary_lookup = ks_ke.set_index("case").to_dict("index")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    axes = axes.ravel()
    for axis, case_id in zip(axes, CASE_ORDER):
        group = case_energy[case_energy["case"] == case_id]
        summary = summary_lookup[case_id]
        axis.plot(group["frame_raw"], group["energy_l1_mm"], color="#bdbdbd", linewidth=1.0, label="Raw energy")
        axis.plot(
            group["frame_raw"],
            group["energy_smooth_mm"],
            color=case_colors()[case_id],
            linewidth=2.0,
            label="Smoothed energy",
        )
        axis.axhline(summary["threshold_value"], color="#c0504d", linestyle="--", linewidth=0.9, label="Threshold")
        axis.axvline(summary["ks_raw"], color="#4f81bd", linestyle=":", linewidth=0.9)
        axis.axvline(summary["ke_raw"], color="#4f81bd", linestyle=":", linewidth=0.9)
        axis.set_title(f"{case_id} response window")
        axis.set_xlabel("Raw frame")
        axis.set_ylabel("Displacement energy (mm)")
        axis.grid(True, linestyle=":")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_fig03_02_alignment_compare(raw_metrics: pd.DataFrame, aligned_energy: pd.DataFrame) -> plt.Figure:
    case_energy = (
        raw_metrics.groupby(["case", "frame_raw"], as_index=False)
        .agg(energy_smooth_mm=("energy_smooth_mm", "first"))
        .sort_values(["case", "frame_raw"])
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for case_id, group in case_energy.groupby("case"):
        axes[0].plot(
            group["frame_raw"],
            group["energy_smooth_mm"] / group["energy_smooth_mm"].max(),
            linewidth=1.8,
            color=case_colors()[case_id],
            label=case_id,
        )
    axes[0].set_title("(a) Before alignment")
    axes[0].set_xlabel("Raw frame")
    axes[0].set_ylabel("Normalized energy")
    axes[0].grid(True, linestyle=":")

    for case_id, group in aligned_energy.groupby("case"):
        axes[1].plot(group["eta"], group["energy_norm"], linewidth=1.8, color=case_colors()[case_id], label=case_id)
    axes[1].set_title("(b) After alignment")
    axes[1].set_xlabel("Actual cutting progress, eta")
    axes[1].set_ylabel("Normalized energy")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(loc="best")
    figure.tight_layout()
    return figure


def plot_fig03_03_line_history(aligned_line_history: pd.DataFrame) -> plt.Figure:
    line_palette = {"01": "#8c510a", "10": "#d8b365", "20": "#5ab4ac", "30": "#01665e", "40": "#2c7fb8"}

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, case_id in zip(axes, CASE_ORDER):
        case_group = aligned_line_history[aligned_line_history["case"] == case_id]
        for line_id, line_group in case_group.groupby("line_id"):
            axis.plot(
                line_group["eta"],
                line_group["w_peak_abs_mm"],
                linewidth=1.8,
                color=line_palette[line_id],
                label=f"Line {line_id}",
            )
        axis.set_title(f"{case_id} linewise response")
        axis.set_xlabel("Actual cutting progress, eta")
        axis.set_ylabel("Peak |w| (mm)")
        axis.grid(True, linestyle=":")
    axes[0].legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_fig03_04_peak_heatmap(raw_metrics: pd.DataFrame) -> plt.Figure:
    heatmap_df = (
        raw_metrics.groupby(["case", "line_id"], as_index=False)
        .agg(w_peak_abs_mm=("w_peak_abs_mm", "max"))
        .pivot(index="case", columns="line_id", values="w_peak_abs_mm")
        .reindex(index=CASE_ORDER, columns=LINE_ORDER)
    )

    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    image = axis.imshow(heatmap_df.to_numpy(), cmap="YlGnBu", aspect="auto")
    axis.set_title("Peak mid-surface displacement map")
    axis.set_xticks(range(len(heatmap_df.columns)))
    axis.set_xticklabels([f"Line {col}" for col in heatmap_df.columns])
    axis.set_yticks(range(len(heatmap_df.index)))
    axis.set_yticklabels(heatmap_df.index.tolist())
    for row_idx, case_id in enumerate(heatmap_df.index):
        for col_idx, line_id in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[case_id, line_id]
            normalized = image.norm(value)
            text_color = "white" if normalized >= 0.58 else "#222222"
            axis.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=10.5, color=text_color)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Peak |w| (mm)")
    figure.tight_layout()
    return figure


def plot_fig03_05_factor_interaction(metrics: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for thickness in sorted(metrics["nominal_thickness_mm"].unique()):
        subset = metrics[metrics["nominal_thickness_mm"] == thickness].sort_values("radius_mm")
        color = thickness_colors()[thickness]
        axes[0].plot(
            subset["radius_mm"],
            subset["peak_displacement_mm"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"t = {thickness:.1f} mm",
        )
        axes[1].plot(
            subset["radius_mm"],
            subset["end_frame_min_thickness_mm"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"t = {thickness:.1f} mm",
        )
        for _, row in subset.iterrows():
            if row["case"] == "S5":
                axes[0].text(
                    row["radius_mm"],
                    row["peak_displacement_mm"] - 0.018,
                    row["case"],
                    ha="center",
                    va="top",
                    fontsize=10.5,
                )
            else:
                axes[0].text(
                    row["radius_mm"],
                    row["peak_displacement_mm"] + 0.016,
                    row["case"],
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                )
            label_y = row["end_frame_min_thickness_mm"] + 0.026
            label_va = "bottom"
            if row["case"] == "S4":
                label_y = row["end_frame_min_thickness_mm"] - 0.032
                label_va = "top"
            axes[1].text(
                row["radius_mm"],
                label_y,
                row["case"],
                ha="center",
                va=label_va,
                fontsize=10.5,
            )
    axes[0].set_title("(a) Interaction on peak displacement")
    axes[0].set_xlabel("Radius (mm)")
    axes[0].set_ylabel("Peak displacement (mm)")
    axes[0].grid(True, linestyle=":")

    axes[1].set_title("(b) Interaction on cut-completion minimum true thickness")
    axes[1].set_xlabel("Radius (mm)")
    axes[1].set_ylabel("Cut-completion minimum true thickness (mm)")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(loc="best")
    figure.tight_layout()
    return figure


def plot_fig03_06_event_timeline(metrics: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12.9, 5.4))
    y_positions = np.arange(len(metrics))
    for y_pos, row in zip(y_positions, metrics.itertuples(index=False)):
        color = case_colors()[row.case]
        axis.hlines(y_pos, row.cut_start_frame_raw, row.cut_done_frame_raw, color="#cfcfcf", linewidth=6, alpha=0.95)
        axis.hlines(y_pos, row.ks_raw, row.ke_raw, color=color, linewidth=3.4, alpha=0.95)
        axis.scatter(row.peak_displacement_frame_raw, y_pos, s=80, color="#222222", marker="o", zorder=3)
        axis.scatter(row.worst_thickness_frame_raw, y_pos, s=80, color="#c0504d", marker="s", zorder=3)
        axis.scatter(row.cut_done_frame_raw, y_pos, s=74, color="#4f81bd", marker="D", zorder=3)
        axis.scatter(
            row.analysis_end_frame_raw,
            y_pos,
            s=76,
            facecolors="white",
            edgecolors="#4f81bd",
            linewidths=1.4,
            marker="^",
            zorder=3,
        )
        axis.text(max(row.analysis_end_frame_raw, row.cut_done_frame_raw) + 3, y_pos, row.case, va="center", fontsize=11)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(metrics["case"])
    axis.set_xlabel("Raw frame")
    axis.set_ylabel("Case")
    axis.set_title("Actual cutting window and characteristic frames")
    axis.grid(True, axis="x", linestyle=":")
    legend_handles = [
        mpl.lines.Line2D([0], [0], color="#cfcfcf", linewidth=6, label="Actual cutting window"),
        mpl.lines.Line2D([0], [0], color="#777777", linewidth=3.4, label="Main response window"),
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=8, label="Peak displacement"),
        mpl.lines.Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0504d", markersize=8, label="Worst thinning"),
        mpl.lines.Line2D([0], [0], marker="D", color="w", markerfacecolor="#4f81bd", markersize=8, label="Cut completion"),
        mpl.lines.Line2D([0], [0], marker="^", color="#4f81bd", markerfacecolor="white", markersize=8, label="Analysis end"),
    ]
    axis.set_xlim(-3, max(metrics["analysis_end_frame_raw"].max(), metrics["cut_done_frame_raw"].max()) + 18)
    axis.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    figure.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    return figure


def plot_fig03_07_true_thickness(aligned_thickness: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, case_id in zip(axes, CASE_ORDER):
        case_group = aligned_thickness[aligned_thickness["case"] == case_id].sort_values("eta")
        axis.plot(case_group["eta"], case_group["thickness_min_mm"], color="#c0504d", linewidth=1.8, label="Minimum")
        axis.plot(case_group["eta"], case_group["thickness_mean_mm"], color="#4f81bd", linewidth=1.8, label="Mean")
        axis.plot(case_group["eta"], case_group["thickness_p95_mm"], color="#7f7f7f", linewidth=1.6, label="95th percentile")
        axis.set_title(f"{case_id} true-thickness evolution")
        axis.set_xlabel("Actual cutting progress, eta")
        axis.set_ylabel("True thickness (mm)")
        axis.grid(True, linestyle=":")
    axes[0].legend(loc="best")
    figure.tight_layout()
    return figure


def plot_fig03_08_coupling_scatter(metrics: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    markers = {30.0: "o", 40.0: "s", 50.0: "^"}
    for row in metrics.itertuples(index=False):
        color = thickness_colors()[row.nominal_thickness_mm]
        marker = markers[row.radius_mm]
        axes[0].scatter(row.peak_displacement_mm, row.maximum_thinning_mm, color=color, marker=marker, s=90)
        axes[1].scatter(row.peak_displacement_mm, row.end_frame_min_thickness_mm, color=color, marker=marker, s=90)
        axes[0].text(row.peak_displacement_mm + 0.01, row.maximum_thinning_mm, row.case, fontsize=10.5, va="center")
        axes[1].text(row.peak_displacement_mm + 0.01, row.end_frame_min_thickness_mm, row.case, fontsize=10.5, va="center")
    axes[0].set_title("(a) Peak displacement vs. maximum thinning")
    axes[0].set_xlabel("Peak displacement (mm)")
    axes[0].set_ylabel("Maximum thinning (mm)")
    axes[0].grid(True, linestyle=":")
    axes[1].set_title("(b) Peak displacement vs. cut-completion minimum thickness")
    axes[1].set_xlabel("Peak displacement (mm)")
    axes[1].set_ylabel("Cut-completion minimum true thickness (mm)")
    axes[1].grid(True, linestyle=":")
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markersize=8, label="R = 30 mm"),
        mpl.lines.Line2D([0], [0], marker="s", color="w", markerfacecolor="#666666", markersize=8, label="R = 40 mm"),
        mpl.lines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#666666", markersize=8, label="R = 50 mm"),
        mpl.lines.Line2D([0], [0], color=thickness_colors()[1.0], linewidth=2, label="t = 1.0 mm"),
        mpl.lines.Line2D([0], [0], color=thickness_colors()[1.5], linewidth=2, label="t = 1.5 mm"),
    ]
    axes[1].legend(handles=legend_handles, loc="best")
    figure.tight_layout()
    return figure


def plot_fig03_09_end_field_heatmaps(thickness_end_field: pd.DataFrame) -> plt.Figure:
    field = thickness_end_field.copy()
    field = field[(field["in_cutting_zone"] == True) & (field["is_physical"] == True)].copy()
    field["theta_bucket_deg"] = field["theta_bucket_deg"].round(2)
    field["axial_bucket_mm"] = field["axial_bucket_mm"].round(2)
    vlim = max(abs(field["thickness_deviation_mm"].min()), abs(field["thickness_deviation_mm"].max()))

    figure, axes = plt.subplots(2, 3, figsize=(16.2, 8.8), constrained_layout=True)
    axes = axes.ravel()
    images = []
    for axis, case_id in zip(axes, CASE_ORDER):
        case_group = field[field["case"] == case_id]
        pivot = case_group.pivot_table(
            index="axial_bucket_mm",
            columns="theta_bucket_deg",
            values="thickness_deviation_mm",
            aggfunc="mean",
        )
        pivot = pivot.sort_index(ascending=False)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        images.append(image)
        axis.set_title(f"{case_id} cut-completion thickness deviation")
        axis.set_xlabel("Angular bin (deg)")
        axis.set_ylabel("Axial position (mm)")
        x_ticks = np.linspace(0, len(pivot.columns) - 1, num=min(5, len(pivot.columns))).astype(int)
        y_ticks = np.linspace(0, len(pivot.index) - 1, num=min(5, len(pivot.index))).astype(int)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels([f"{pivot.columns[idx]:.1f}" for idx in x_ticks])
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([f"{pivot.index[idx]:.1f}" for idx in y_ticks])
    colorbar = figure.colorbar(images[0], ax=axes.tolist(), shrink=0.88)
    colorbar.set_label("Thickness deviation (mm)")
    return figure


def make_placeholder_figure(title: str, lines: list[str]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10.8, 4.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    rect = mpl.patches.FancyBboxPatch(
        (0.06, 0.14),
        0.88,
        0.72,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#888888",
        facecolor="#f7f7f7",
    )
    axis.add_patch(rect)
    axis.text(0.5, 0.70, title, ha="center", va="center", fontsize=17, fontweight="bold")
    axis.text(0.5, 0.43, "\n".join(lines), ha="center", va="center", fontsize=12.0, linespacing=1.5)
    return figure


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    datadir = Path(args.datadir)
    figdir = Path(args.figdir)
    config = load_config(args.config)
    inputs = load_inputs(datadir)
    metrics = build_case_metrics(inputs, config)
    line_metrics = build_line_metrics(inputs["raw_metrics"])

    write_case_metrics(datadir, metrics)
    write_line_metrics(datadir, line_metrics)
    figdir.mkdir(parents=True, exist_ok=True)
    write_figure_notes(figdir)

    figures = [
        ("fig03_01_response_window_en", plot_fig03_01_response_window(inputs["raw_metrics"], inputs["ks_ke"])),
        ("fig03_02_alignment_compare_en", plot_fig03_02_alignment_compare(inputs["raw_metrics"], inputs["aligned_energy"])),
        ("fig03_03_line_history_en", plot_fig03_03_line_history(inputs["aligned_line_history"])),
        ("fig03_04_peak_heatmap_en", plot_fig03_04_peak_heatmap(inputs["raw_metrics"])),
        ("fig03_05_factor_interaction_en", plot_fig03_05_factor_interaction(metrics)),
        ("fig03_06_event_timeline_en", plot_fig03_06_event_timeline(metrics)),
        ("fig03_07_true_thickness_evolution_en", plot_fig03_07_true_thickness(inputs["aligned_thickness"])),
        ("fig03_08_coupling_scatter_en", plot_fig03_08_coupling_scatter(metrics)),
        ("fig03_09_end_field_heatmaps_en", plot_fig03_09_end_field_heatmaps(inputs["thickness_end_field"])),
        (
            "fig03_10_odb_global_placeholder_en",
            make_placeholder_figure(
                "Insert ODB global deformation contour here",
                [
                    "Recommended content:",
                    "Overall deformation contour at the characteristic frame",
                    "Include cutter position and displacement scale",
                ],
            ),
        ),
        (
            "fig03_11_odb_local_placeholder_en",
            make_placeholder_figure(
                "Insert ODB local defect-sensitive view here",
                [
                    "Recommended content:",
                    "Local enlarged view of the minimum-thickness region",
                    "Mark theta and axial coordinates of the critical point",
                ],
            ),
        ),
        (
            "fig03_12_odb_thickness_placeholder_en",
            make_placeholder_figure(
                "Insert ODB thickness-path visualization here",
                [
                    "Recommended content:",
                    "Section or path plot across the thinning zone",
                    "Use the same frame as the local enlarged view when possible",
                ],
            ),
        ),
    ]

    for stem, figure in figures:
        save_figure(figure, figdir, stem)

    print(f"Exported {len(figures)} English thesis figures to {figdir}")
    print(f"Wrote case metrics to {datadir / 'thesis_case_metrics.csv'}")
    print(f"Wrote line metrics to {datadir / 'thesis_line_metrics.csv'}")


if __name__ == "__main__":
    main()
