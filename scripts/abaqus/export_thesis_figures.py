#!/usr/bin/env python3
from __future__ import annotations

"""Export thesis figures from processed Abaqus tables.

本脚本读取第 3 章 Abaqus 数据链已经生成的表格，集中导出测线与种子点
示意图、能量窗口图、对齐对比图、中面位移时程、峰值热力图、曲率半径
与壁厚影响图、真壁厚时程和旧近似量偏差图。若真壁厚 CSV 尚未生成，
脚本会输出占位图，提醒需要先运行 ODB 提取流程。

This script reads processed Abaqus tables and exports Chapter 3 figures:
measurement-line schematics, energy-window diagnostics, alignment comparison,
mid-surface displacement histories, peak heatmaps, radius/thickness effect
plots, true-thickness histories, and legacy-bias plots. If true-thickness CSV
files are absent, placeholder figures are generated.
"""

import argparse
import math
import os
from pathlib import Path

# Keep the Matplotlib cache in the project output tree for reproducible,
# sandbox-friendly figure generation.
# 将 Matplotlib 缓存放在项目 output 目录中，便于受限环境稳定导图。
os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[2] / "output" / ".mplconfig")))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, case_settings, line_positions, load_case_config, nominal_inner_radius, nominal_outer_radius, theta_deg


def parse_args() -> argparse.Namespace:
    """Parse configuration, processed-data, and figure-output paths.

    默认读取 `output/spreadsheet/abaqus` 中的数据表，图件写入
    `output/figures/abaqus`。

    By default, processed tables are read from `output/spreadsheet/abaqus` and
    figures are written to `output/figures/abaqus`.
    """
    parser = argparse.ArgumentParser(description="导出论文图件与图注说明。")
    parser.add_argument(
        "--config",
        default=str(ROOT / "scripts" / "abaqus" / "case_config.yml"),
        help="案例配置文件路径。",
    )
    parser.add_argument(
        "--datadir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="表格输入目录。",
    )
    parser.add_argument(
        "--figdir",
        default=str(ROOT / "output" / "figures" / "abaqus"),
        help="图件输出目录。",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    """Configure fonts, axes, grid, and legend defaults.

    这些设置保证同一批图在不同机器上具有接近的字体、边框和网格样式。

    These settings keep fonts, borders, and grid styles consistent across
    machines.
    """
    mpl.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS", "DejaVu Serif"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["axes.edgecolor"] = "#222222"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["grid.color"] = "#d9d9d9"
    mpl.rcParams["grid.linewidth"] = 0.6
    mpl.rcParams["legend.frameon"] = False


def save_figure(figure: plt.Figure, figdir: Path, stem: str) -> tuple[Path, Path]:
    """Save one figure as high-resolution PNG and vector PDF.

    PNG 便于 Word 插图，PDF 便于后期矢量排版。

    PNG is convenient for Word insertion, while PDF supports vector editing and
    layout work.
    """
    figdir.mkdir(parents=True, exist_ok=True)
    png_path = figdir / f"{stem}.png"
    pdf_path = figdir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def load_inputs(datadir: Path) -> dict[str, pd.DataFrame]:
    """Load all processed Abaqus tables required by the figures.

    真壁厚表采用可选读取策略。缺少 `true_thickness_points_*.csv` 时仍可导出
    其他图件，并生成真壁厚占位图。

    True-thickness tables are optional. When `true_thickness_points_*.csv` is
    absent, other figures are still exported and thickness placeholders are
    generated.
    """
    inputs = {
        "raw_metrics": pd.read_csv(datadir / "raw_displacement_metrics.csv.gz"),
        "aligned_raw": pd.read_csv(datadir / "aligned_displacement_long.csv.gz"),
        "aligned_grid": pd.read_csv(datadir / "aligned_displacement_grid.csv.gz"),
        "ks_ke": pd.read_csv(datadir / "ks_ke_summary.csv"),
        "seeds": pd.read_csv(datadir / "thickness_point_seeds.csv"),
        "legacy_audit": pd.read_csv(datadir / "legacy_formula_audit.csv"),
    }
    true_thickness_files = sorted(datadir.glob("true_thickness_points_*.csv"))
    if true_thickness_files:
        inputs["true_thickness"] = pd.concat([pd.read_csv(path) for path in true_thickness_files], ignore_index=True)
    else:
        inputs["true_thickness"] = pd.DataFrame()
    for key in ["raw_metrics", "aligned_raw", "aligned_grid"]:
        if "line_id" in inputs[key].columns:
            inputs[key]["line_id"] = inputs[key]["line_id"].astype(str).str.zfill(2)
    if "case" in inputs["seeds"].columns:
        inputs["seeds"]["case"] = inputs["seeds"]["case"].astype(str).str.upper()
    return inputs


def case_colors() -> dict[str, str]:
    """Return stable colors for S1-S6.

    固定配色可让多张图中的工况识别保持一致。

    Stable colors keep case identification consistent across figures.
    """
    return {
        "S1": "#8c2d04",
        "S2": "#cc4c02",
        "S3": "#016c59",
        "S4": "#1c9099",
        "S5": "#225ea8",
        "S6": "#253494",
    }


def point_markers() -> dict[str, tuple[str, str]]:
    """Return marker and color mapping for seed-point types.

    控制点、凹陷点和凸起点使用不同符号，方便阅读真壁厚图和种子位置图。

    Control, dent, and bulge points use different marker styles for readability.
    """
    return {
        "control": ("o", "#333333"),
        "dent": ("v", "#c0504d"),
        "bulge": ("^", "#3c8d0d"),
    }


def plot_a1_line_seed_schematic(config: dict[str, object], seeds: pd.DataFrame) -> plt.Figure:
    """Create the line-location and seed-projection schematic.

    左图说明五条测线沿悬臂方向的位置，右图说明缺陷点在单曲率截面上的
    投影位置。

    The left panel shows the five measurement-line positions; the right panel
    projects defect seeds onto the single-curvature cross-section.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axis = axes[0]
    axis.set_title("图A1(a) 五条测线位置示意")
    axis.set_xlim(0, 1)
    axis.set_ylim(8.5, -0.3)
    axis.set_xticks([])
    axis.set_ylabel("距自由端位置 / mm")
    wall_x = np.array([0.25, 0.75, 0.75, 0.25, 0.25])
    wall_y = np.array([0.0, 0.0, 8.2, 8.2, 0.0])
    axis.plot(wall_x, wall_y, color="#222222", linewidth=1.2)
    for line_id, line_mm in sorted(line_positions(config).items()):
        axis.hlines(line_mm, 0.25, 0.75, colors="#1f4e79", linestyles="--", linewidth=1.0)
        axis.text(0.77, line_mm, f"{line_id} 行 ({line_mm:.1f} mm)", va="center", fontsize=10)
    axis.text(0.5, -0.1, "自由端", ha="center", va="top", fontsize=10)
    axis.text(0.5, 8.35, "约束端", ha="center", va="bottom", fontsize=10)

    polar_axis = axes[1]
    polar_axis.set_title("图A1(b) 缺陷点种子在单曲率截面上的投影")
    polar_axis.set_aspect("equal")
    for case_id, case_group in seeds.groupby("case"):
        case_cfg = case_settings(config, case_id)
        outer_radius = nominal_outer_radius(case_cfg)
        inner_radius = nominal_inner_radius(case_cfg)
        theta_min = case_group["theta_seed_deg"].min()
        theta_max = case_group["theta_seed_deg"].max()
        theta_arc = np.deg2rad(np.linspace(theta_min - 3, theta_max + 3, 200))
        polar_axis.plot(outer_radius * np.cos(theta_arc), outer_radius * np.sin(theta_arc), color="#777777", linewidth=0.8)
        polar_axis.plot(inner_radius * np.cos(theta_arc), inner_radius * np.sin(theta_arc), color="#bbbbbb", linewidth=0.8)
        polar_axis.text(
            outer_radius * np.cos(np.deg2rad(theta_min - 2)),
            outer_radius * np.sin(np.deg2rad(theta_min - 2)),
            case_id,
            fontsize=9,
        )
        for _, row in case_group.iterrows():
            marker, color = point_markers()[row["point_type"]]
            polar_axis.scatter(row["x_seed_mm"], row["y_seed_mm"], marker=marker, color=color, s=28)
    polar_axis.set_xlabel("X / mm")
    polar_axis.set_ylabel("Y / mm")
    polar_axis.grid(True, linestyle=":")
    return figure


def plot_a2_energy_windows(ks_ke: pd.DataFrame, raw_metrics: pd.DataFrame) -> plt.Figure:
    """Plot raw and smoothed displacement-energy windows for all cases.

    该图复核 `ks_raw` 和 `ke_raw` 的选取依据。

    This figure documents how `ks_raw` and `ke_raw` are selected.
    """
    case_energy = (
        raw_metrics.groupby(["case", "frame_raw"], as_index=False)
        .agg(energy_l1_mm=("energy_l1_mm", "first"), energy_smooth_mm=("energy_smooth_mm", "first"))
        .sort_values(["case", "frame_raw"])
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    summary_lookup = ks_ke.set_index("case").to_dict("index")
    colors = case_colors()
    for axis, case_id in zip(axes, sorted(case_energy["case"].unique())):
        group = case_energy[case_energy["case"] == case_id]
        summary = summary_lookup[case_id]
        axis.plot(group["frame_raw"], group["energy_l1_mm"], color="#bdbdbd", linewidth=1.0, label="原始能量")
        axis.plot(group["frame_raw"], group["energy_smooth_mm"], color=colors[case_id], linewidth=2.0, label="平滑能量")
        axis.axhline(summary["peak_energy"] * summary["threshold_ratio"], color="#c0504d", linestyle="--", linewidth=0.9)
        axis.axvline(summary["ks_raw"], color="#4f81bd", linestyle=":", linewidth=0.9)
        axis.axvline(summary["ke_raw"], color="#4f81bd", linestyle=":", linewidth=0.9)
        axis.set_title(f"{case_id} 主响应窗口")
        axis.set_xlabel("原始帧号")
        axis.set_ylabel("E_k / mm")
    figure.tight_layout()
    return figure


def plot_a3_alignment_compare(raw_metrics: pd.DataFrame, aligned_grid: pd.DataFrame) -> plt.Figure:
    """Compare frame histories before and after `xi` alignment.

    左图保留原始帧号，右图使用归一化切削进程，展示跨工况对齐的效果。

    The left panel uses raw frames; the right panel uses normalized cutting
    progress to show cross-case alignment.
    """
    case_energy = (
        raw_metrics.groupby(["case", "frame_raw"], as_index=False)
        .agg(energy_smooth_mm=("energy_smooth_mm", "first"))
        .sort_values(["case", "frame_raw"])
    )
    aligned_energy = (
        aligned_grid.groupby(["case", "xi"], as_index=False)
        .agg(energy_l1_mm=("w_mid_mm", lambda series: float(np.abs(series).sum())))
        .sort_values(["case", "xi"])
    )
    aligned_energy["energy_norm"] = aligned_energy.groupby("case")["energy_l1_mm"].transform(lambda s: s / s.max())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = case_colors()
    for case_id, group in case_energy.groupby("case"):
        norm = group["energy_smooth_mm"] / group["energy_smooth_mm"].max()
        axes[0].plot(group["frame_raw"], norm, color=colors[case_id], linewidth=1.8, label=case_id)
    axes[0].set_title("图A3(a) 对齐前原始帧比较")
    axes[0].set_xlabel("原始帧号")
    axes[0].set_ylabel("归一化能量")
    axes[0].grid(True, linestyle=":")
    for case_id, group in aligned_energy.groupby("case"):
        axes[1].plot(group["xi"], group["energy_norm"], color=colors[case_id], linewidth=1.8, label=case_id)
    axes[1].set_title("图A3(b) 对齐后 xi 比较")
    axes[1].set_xlabel("归一化切削进程 ξ")
    axes[1].set_ylabel("归一化能量")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(loc="best")
    figure.tight_layout()
    return figure


def plot_a4_line_histories(aligned_grid: pd.DataFrame) -> plt.Figure:
    """Plot aligned peak mid-surface displacement histories by line.

    每个子图对应一个工况，每条曲线对应一条测线，展示不同位置的让刀响应。

    Each subplot is one case and each curve is one measurement line, showing
    location-dependent deflection response.
    """
    peak_by_xi = (
        aligned_grid.groupby(["case", "line_id", "line_mm", "xi"], as_index=False)
        .agg(w_peak_abs_mm=("w_mid_mm", lambda series: float(np.abs(series).max())))
        .sort_values(["case", "line_mm", "xi"])
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    line_palette = {
        "01": "#8c510a",
        "10": "#d8b365",
        "20": "#5ab4ac",
        "30": "#01665e",
        "40": "#2c7fb8",
    }
    for axis, case_id in zip(axes, sorted(peak_by_xi["case"].unique())):
        case_group = peak_by_xi[peak_by_xi["case"] == case_id]
        for line_id, line_group in case_group.groupby("line_id"):
            axis.plot(
                line_group["xi"],
                line_group["w_peak_abs_mm"],
                linewidth=1.8,
                color=line_palette[line_id],
                label=f"{line_id} 行",
            )
        axis.set_title(f"{case_id} 测线时程")
        axis.set_xlabel("归一化切削进程 ξ")
        axis.set_ylabel("|w| 峰值 / mm")
        axis.grid(True, linestyle=":")
    axes[0].legend(loc="upper left", ncol=1)
    figure.tight_layout()
    return figure


def plot_a5_peak_heatmap(raw_metrics: pd.DataFrame) -> plt.Figure:
    """Create a case-by-line heatmap of peak mid-surface displacement.

    热力图压缩展示工况和测线两个因素下的峰值位移差异。

    The heatmap compactly shows peak-displacement differences by case and line.
    """
    heatmap_df = (
        raw_metrics.groupby(["case", "line_id"], as_index=False)
        .agg(w_peak_abs_mm=("w_peak_abs_mm", "max"))
        .pivot(index="case", columns="line_id", values="w_peak_abs_mm")
        .reindex(index=["S1", "S2", "S3", "S4", "S5", "S6"], columns=["01", "10", "20", "30", "40"])
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    image = axis.imshow(heatmap_df.to_numpy(), cmap="YlGnBu", aspect="auto")
    axis.set_title("图A5 六算例峰值中面位移热力图")
    axis.set_xticks(range(len(heatmap_df.columns)))
    axis.set_xticklabels([f"{col} 行" for col in heatmap_df.columns])
    axis.set_yticks(range(len(heatmap_df.index)))
    axis.set_yticklabels(heatmap_df.index.tolist())
    for row_idx, case_id in enumerate(heatmap_df.index):
        for col_idx, line_id in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[case_id, line_id]
            axis.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=9, color="#222222")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("|w| 峰值 / mm")
    figure.tight_layout()
    return figure


def plot_a6_radius_thickness_factor(raw_metrics: pd.DataFrame, config: dict[str, object]) -> plt.Figure:
    """Plot the joint effect of curvature radius and nominal thickness.

    柱状图把工况编号映射到半径和壁厚组合，用于比较几何参数对峰值位移
    的影响。

    The bar chart maps case ids to radius/thickness combinations and compares
    their effect on peak displacement.
    """
    case_peak = raw_metrics.groupby("case", as_index=False).agg(peak_w_mm=("w_peak_abs_mm", "max"))
    case_peak["radius_mm"] = case_peak["case"].map(lambda case_id: case_settings(config, case_id)["radius_mm"])
    case_peak["thickness_mm"] = case_peak["case"].map(lambda case_id: case_settings(config, case_id)["nominal_thickness_mm"])
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    radius_values = sorted(case_peak["radius_mm"].unique())
    width = 0.28
    offsets = {1.0: -width / 2, 1.5: width / 2}
    colors = {1.0: "#c0504d", 1.5: "#4f81bd"}
    for thickness in sorted(case_peak["thickness_mm"].unique()):
        subset = case_peak[case_peak["thickness_mm"] == thickness].sort_values("radius_mm")
        x = np.arange(len(radius_values)) + offsets[thickness]
        axis.bar(x, subset["peak_w_mm"], width=width, color=colors[thickness], label=f"t={thickness:.1f} mm")
        for x_pos, (_, row) in zip(x, subset.iterrows()):
            axis.text(x_pos, row["peak_w_mm"] + 0.01, row["case"], ha="center", va="bottom", fontsize=9)
    axis.set_xticks(np.arange(len(radius_values)))
    axis.set_xticklabels([f"R={radius:.0f} mm" for radius in radius_values])
    axis.set_ylabel("峰值中面位移 / mm")
    axis.set_title("图A6 曲率半径与壁厚对峰值位移的双因素比较")
    axis.grid(True, axis="y", linestyle=":")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_thickness_placeholder(title: str, message: str) -> plt.Figure:
    """Create a placeholder figure when true-thickness data are unavailable.

    占位图让批量导图流程不中断，也明确提示缺少的前置数据。

    The placeholder keeps batch export running and clearly identifies the
    missing upstream data.
    """
    figure, axis = plt.subplots(figsize=(8.5, 3.8))
    axis.axis("off")
    axis.text(0.5, 0.68, title, ha="center", va="center", fontsize=16, weight="bold")
    axis.text(
        0.5,
        0.38,
        message,
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#d0d0d0"},
    )
    return figure


def plot_a7_true_thickness(true_thickness: pd.DataFrame) -> plt.Figure:
    """Plot true wall-thickness time histories by point type.

    真壁厚来自 ODB 内外表面节点配对结果，按控制点、凹陷点和凸起点分组。

    True thickness comes from ODB inner/outer node pairing and is grouped by
    control, dent, and bulge point types.
    """
    if true_thickness.empty:
        return plot_thickness_placeholder(
            "图A7 真壁厚时程图待 ODB 提取后自动生成",
            "当前工作区尚未产出 true_thickness_points_<case>.csv。\n在 Abaqus 环境执行 extract_true_thickness_odb.py 后重新运行本脚本即可生成该图。",
        )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    markers = point_markers()
    for axis, case_id in zip(axes, sorted(true_thickness["case"].unique())):
        case_group = true_thickness[true_thickness["case"] == case_id].sort_values("frame_raw")
        for point_type, group in case_group.groupby("point_type"):
            marker, color = markers.get(point_type, ("o", "#222222"))
            axis.plot(group["frame_raw"], group["thickness_true_mm"], marker=marker, color=color, linewidth=1.4, label=point_type)
        axis.set_title(f"{case_id} 真壁厚时程")
        axis.set_xlabel("原始帧号")
        axis.set_ylabel("真壁厚 / mm")
        axis.grid(True, linestyle=":")
    axes[0].legend(loc="best")
    figure.tight_layout()
    return figure


def plot_a8_true_vs_legacy(true_thickness: pd.DataFrame) -> plt.Figure:
    """Compare legacy radius approximation with ODB-derived true thickness.

    散点偏离对角线的程度反映旧近似量与真壁厚之间的差异。

    Distance from the diagonal line shows the difference between legacy
    approximation and ODB-derived true thickness.
    """
    if true_thickness.empty:
        return plot_thickness_placeholder(
            "图A8 旧近似量与真壁厚偏差图待 ODB 提取后自动生成",
            "该图需要同时存在 thickness_legacy_mm 与 thickness_true_mm。\n脚本已支持自动导出，待 ODB 结果落地后重新运行即可。",
        )
    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    markers = point_markers()
    for point_type, group in true_thickness.groupby("point_type"):
        marker, color = markers.get(point_type, ("o", "#222222"))
        axis.scatter(group["thickness_legacy_mm"], group["thickness_true_mm"], marker=marker, color=color, s=34, alpha=0.85, label=point_type)
    lower = min(true_thickness["thickness_legacy_mm"].min(), true_thickness["thickness_true_mm"].min())
    upper = max(true_thickness["thickness_legacy_mm"].max(), true_thickness["thickness_true_mm"].max())
    axis.plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=1.0)
    axis.set_xlabel("旧近似量 / mm")
    axis.set_ylabel("真壁厚 / mm")
    axis.set_title("图A8 旧近似量与真壁厚偏差对比")
    axis.grid(True, linestyle=":")
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def plot_a9_seed_locations(config: dict[str, object], seeds: pd.DataFrame) -> plt.Figure:
    """Plot seed-point positions for each case.

    该图用于说明缺陷点和控制点在每个单曲率截面上的空间位置。

    This figure documents spatial locations of defect and control points on
    each single-curvature section.
    """
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=False)
    axes = axes.ravel()
    markers = point_markers()
    for axis, case_id in zip(axes, sorted(seeds["case"].unique())):
        case_cfg = case_settings(config, case_id)
        group = seeds[seeds["case"] == case_id]
        theta_values = np.deg2rad(np.linspace(group["theta_seed_deg"].min() - 3, group["theta_seed_deg"].max() + 3, 240))
        outer_radius = nominal_outer_radius(case_cfg)
        inner_radius = nominal_inner_radius(case_cfg)
        axis.plot(outer_radius * np.cos(theta_values), outer_radius * np.sin(theta_values), color="#1f1f1f", linewidth=1.1)
        axis.plot(inner_radius * np.cos(theta_values), inner_radius * np.sin(theta_values), color="#a6a6a6", linewidth=1.0)
        for _, row in group.iterrows():
            marker, color = markers[row["point_type"]]
            axis.scatter(row["x_seed_mm"], row["y_seed_mm"], marker=marker, color=color, s=34)
            axis.text(row["x_seed_mm"] + 0.5, row["y_seed_mm"] + 0.5, row["frame_label"], fontsize=8)
        axis.set_title(f"{case_id} 缺陷点种子位置")
        axis.set_xlabel("X / mm")
        axis.set_ylabel("Y / mm")
        axis.grid(True, linestyle=":")
        axis.set_aspect("equal")
    figure.tight_layout()
    return figure


def write_captions(figdir: Path, has_true_thickness: bool) -> None:
    """Write a markdown list of exported figures and recommended usage.

    图注清单记录每张图建议放入的正文小节，便于论文排版时追踪图件来源。

    The caption list records recommended manuscript placement and helps trace
    figure provenance during writing.
    """
    captions = [
        ("图A1", "五条测线与缺陷点种子示意图。建议放在仿真数据来源与测点定义小节，用于说明测线口径与缺陷点来源。"),
        ("图A2", "六算例原始能量序列与 ks/ke 自动识别结果。建议放在时间轴对齐方法小节，证明自动窗选的依据。"),
        ("图A3", "对齐前后归一化能量对比图。建议放在时间轴对齐方法小节，说明直接比较原始帧会造成结论偏差。"),
        ("图A4", "不同测线位置的中面位移时程图。建议放在对齐后位移规律小节，突出自由端附近与约束端附近的响应差异。"),
        ("图A5", "峰值中面位移热力图。建议放在位移总体规律小节，用于压缩展示算例与位置双因素差异。"),
        ("图A6", "曲率半径与壁厚对峰值位移的双因素比较。建议放在参数影响分析小节，直接支撑 R 与 t 的刚度效应结论。"),
        (
            "图A7",
            "真壁厚时程图。建议放在缺陷处真壁厚重算小节，比较控制点、凹陷点与凸起点的厚度演化。"
            if has_true_thickness
            else "真壁厚时程图占位图。当前尚未运行 ODB 提取，待生成 true_thickness_points_<case>.csv 后替换为正式图。",
        ),
        (
            "图A8",
            "旧近似量与真壁厚偏差对比图。建议放在缺陷处真壁厚重算小节，用于说明旧工作簿结果为何不能直接作为结论。"
            if has_true_thickness
            else "旧近似量与真壁厚偏差图占位图。当前缺少 ODB 真厚度结果，待运行提取脚本后自动替换。",
        ),
        ("图A9", "代表性缺陷点空间位置图。建议放在壁厚重算结果前，用于交代各算例缺陷点与控制点的空间位置。"),
    ]
    lines = ["# 图表清单", ""]
    for figure_id, text in captions:
        lines.append(f"## {figure_id}")
        lines.append(text)
        lines.append("")
    (figdir / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Load processed data and export all Abaqus thesis figures.

    主流程只消费已处理表格，不读取原始 ODB 或原始位移工作簿。

    The main workflow consumes processed tables only; raw ODB files and raw
    displacement workbooks are not read here.
    """
    args = parse_args()
    configure_matplotlib()
    config = load_case_config(args.config)
    datadir = Path(args.datadir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(datadir)

    figures = [
        ("figA1_line_seed_schematic", plot_a1_line_seed_schematic(config, inputs["seeds"])),
        ("figA2_energy_windows", plot_a2_energy_windows(inputs["ks_ke"], inputs["raw_metrics"])),
        ("figA3_alignment_compare", plot_a3_alignment_compare(inputs["raw_metrics"], inputs["aligned_grid"])),
        ("figA4_line_histories", plot_a4_line_histories(inputs["aligned_grid"])),
        ("figA5_peak_heatmap", plot_a5_peak_heatmap(inputs["raw_metrics"])),
        ("figA6_radius_thickness_factor", plot_a6_radius_thickness_factor(inputs["raw_metrics"], config)),
        ("figA7_true_thickness_timeseries", plot_a7_true_thickness(inputs["true_thickness"])),
        ("figA8_true_vs_legacy_bias", plot_a8_true_vs_legacy(inputs["true_thickness"])),
        ("figA9_seed_locations_by_case", plot_a9_seed_locations(config, inputs["seeds"])),
    ]
    for stem, figure in figures:
        save_figure(figure, figdir, stem)
    write_captions(figdir, has_true_thickness=not inputs["true_thickness"].empty)
    print(f"已输出图件目录: {figdir}")


if __name__ == "__main__":
    main()
