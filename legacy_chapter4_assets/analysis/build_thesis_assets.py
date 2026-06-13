from __future__ import annotations

"""Build Chapter 4 experimental figures and summary tables.

本脚本整理第 4 章实体实验数据资产。它读取三坐标壁厚工作簿、批量
粗糙度结果、Gaussian 粗糙度点云和实验照片，生成壁厚误差图、重复性
图、粗糙度指标图、壁厚误差与 Sa 对应图、粗糙度高度图，以及供正文
引用的统计 CSV。

This script prepares Chapter 4 experimental assets. It reads a CMM wall-thickness
workbook, batch roughness results, Gaussian roughness point clouds, and
experimental photos. It exports wall-thickness figures, repeatability figures,
roughness charts, wall-error versus Sa scatter plots, roughness maps, and
summary CSV tables.
"""

import json
import os
from pathlib import Path

# Store Matplotlib cache inside the open-source bundle output tree so figure
# generation does not depend on a user-specific home directory.
# 将 Matplotlib 缓存放入开源包输出目录，避免依赖用户主目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "analysis" / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
FIG = OUT / "figures"
ANA = OUT / "analysis"
FIG.mkdir(parents=True, exist_ok=True)
ANA.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.titlepad": 10,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    }
)


def parse_thickness(path: Path) -> pd.DataFrame:
    """Convert the CMM workbook into a tidy wall-thickness table.

    原始工作簿按工况分块记录切前厚度、理论切后厚度和切后测量厚度。
    本函数转成长表，并计算壁厚误差、实际去除量和去除比例。

    The source workbook is grouped by case and stores pre-cut thickness,
    theoretical post-cut thickness, and measured post-cut thickness. This
    function converts it to a tidy long table and computes thickness error,
    actual removal, and removal ratio.
    """
    raw = pd.read_excel(path, header=None)
    records: list[dict[str, object]] = []
    current = ""
    before: dict[str, float] = {}
    theory: dict[str, float] = {}
    for _, row in raw.iloc[2:].iterrows():
        if pd.notna(row[0]):
            current = str(row[0])
            before = {"W1": float(row[1]), "W2": float(row[2])}
            theory = {"W1": float(row[6]), "W2": float(row[7])}
        distance = float(str(row[3]).replace("mm", ""))
        for workpiece, col in (("W1", 4), ("W2", 5)):
            after = float(row[col])
            records.append(
                {
                    "case": current,
                    "sample": current.split("-")[0],
                    "workpiece": workpiece,
                    "distance_mm": distance,
                    "before_mm": before[workpiece],
                    "theoretical_after_mm": theory[workpiece],
                    "measured_after_mm": after,
                    "thickness_error_mm": after - theory[workpiece],
                    "actual_removal_mm": before[workpiece] - after,
                }
            )
    df = pd.DataFrame(records)
    df["removal_ratio"] = df["actual_removal_mm"] / 0.5
    return df


def load_roughness(path: Path) -> pd.DataFrame:
    """Load batch roughness results and derive case/workpiece labels.

    批量粗糙度结果中的 folder 字段编码工件号和样件号，本函数解析出正文
    表格使用的 `case` 和 `workpiece` 字段。

    The `folder` field in the batch roughness table encodes workpiece and sample
    ids. This helper derives the `case` and `workpiece` fields used in the
    manuscript tables.
    """
    df = pd.read_csv(path)
    df["workpiece"] = df["folder"].str.extract(r"TXF(\d)-")[0].map({"1": "W1", "2": "W2"})
    df["case"] = "S" + df["folder"].str.replace("TXF", "", regex=False).str.replace(r"^[12]-", "", regex=True)
    return df


def savefig(fig: plt.Figure, name: str) -> str:
    """Save a Matplotlib figure and return its path relative to `OUT`.

    相对路径写入 manifest，便于后续 Word 或 Markdown 组稿脚本引用。

    Relative paths are stored in the manifest for downstream Word or Markdown
    assembly scripts.
    """
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(path.relative_to(OUT))


def export_photo(src: Path, name: str, max_width: int = 2400, quality: int = 88) -> str:
    """Resize and export one experimental photo for document use.

    原始照片可能尺寸过大。压缩后的 JPEG 用于正文插图，减少文档体积。

    Raw photos may be large. Resized JPEG exports are used as document figures
    to reduce file size.
    """
    im = Image.open(src).convert("RGB")
    if im.width > max_width:
        new_h = int(round(im.height * max_width / im.width))
        im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
    dst = FIG / name
    im.save(dst, "JPEG", quality=quality, optimize=True)
    return str(dst.relative_to(OUT))


def make_measurement_scheme() -> str:
    """Draw the CMM measurement-line layout.

    图中标出距自由端 1 mm、4 mm 和 7 mm 的测线位置，对应壁厚误差统计
    的空间口径。

    The figure marks lines 1 mm, 4 mm, and 7 mm from the free end, matching the
    spatial grouping used in thickness-error statistics.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 10.8)
    ax.set_ylim(-0.4, 3.7)
    wall = plt.Rectangle((0.8, 0.55), 8.8, 1.25, facecolor="#d9e8f5", edgecolor="#32506b", linewidth=1.2)
    clamp = plt.Rectangle((8.85, 0.25), 0.75, 1.85, facecolor="#b5b5b5", edgecolor="#555555", linewidth=1.0)
    ax.add_patch(wall)
    ax.add_patch(clamp)
    ax.text(0.8, 2.15, "Free end", ha="center", va="bottom", fontsize=11)
    ax.text(9.25, 2.15, "Fixed end", ha="center", va="bottom", fontsize=11)
    ax.annotate("", xy=(0.8, 0.15), xytext=(9.6, 0.15), arrowprops={"arrowstyle": "<->", "lw": 1.0})
    ax.text(5.2, -0.05, "Cantilever direction", ha="center", va="top")
    positions = [1.8, 4.8, 7.8]
    distances = [1, 4, 7]
    colors = ["#c44e52", "#55a868", "#4c72b0"]
    for x, d, c in zip(positions, distances, colors):
        ax.plot([x, x], [0.48, 1.87], color=c, linewidth=2.6)
        ax.text(x, 2.45, f"{d} mm", ha="center", va="bottom", color=c, fontsize=11)
        ax.annotate("", xy=(x, 2.3), xytext=(0.8, 2.3), arrowprops={"arrowstyle": "<->", "lw": 0.9, "color": c})
    ax.text(5.0, 3.15, "CMM measurement lines from the free end", ha="center", va="center", fontsize=12)
    ax.axis("off")
    return savefig(fig, "fig4_3_cmm_measurement_lines.png")


def make_thickness_figures(df: pd.DataFrame) -> list[str]:
    """Create wall-thickness error figures and summary CSV files.

    输出包括沿悬臂方向的误差曲线、工况均值柱状图、位置效应柱状图和
    重复性图，同时写出长表、工况统计、位置统计和样件统计。

    Outputs include cantilever-direction error profiles, case mean bars,
    position-effect bars, repeatability bars, and CSV summaries by record,
    case, position, and sample.
    """
    assets: list[str] = []
    distance_order = [7.0, 4.0, 1.0]

    profile = (
        df.groupby(["sample", "distance_mm"], as_index=False)["thickness_error_mm"]
        .mean()
        .sort_values(["sample", "distance_mm"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]
    for sample, color in zip(["S1", "S2", "S3", "S4", "S5", "S6"], colors):
        sub = profile[profile["sample"] == sample].set_index("distance_mm").loc[distance_order].reset_index()
        ax.plot(
            sub["distance_mm"],
            sub["thickness_error_mm"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=sample,
            color=color,
        )
    ax.invert_xaxis()
    ax.set_xticks(distance_order, [f"{d:g}" for d in distance_order])
    ax.set_xlabel("Distance from free end (mm)")
    ax.set_ylabel("Thickness error, $e_h$ (mm)")
    ax.set_title("Thickness error profiles along the cantilever direction")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    assets.append(savefig(fig, "fig4_4_thickness_error_profiles.png"))

    case_summary = df.groupby("case")["thickness_error_mm"].agg(["mean", "std", "min", "max"]).reset_index()
    case_summary["case"] = pd.Categorical(case_summary["case"], ["S1", "S2", "S3", "S4", "S5-1", "S5-2", "S6"])
    case_summary = case_summary.sort_values("case")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.arange(len(case_summary))
    ax.bar(x, case_summary["mean"], yerr=case_summary["std"], capsize=4, color="#4c72b0", edgecolor="#26394d")
    ax.set_xticks(x, case_summary["case"].astype(str))
    ax.set_xlabel("Case")
    ax.set_ylabel("Mean thickness error (mm)")
    ax.set_title("Mean residual wall-thickness error by case")
    for xi, yi in zip(x, case_summary["mean"]):
        ax.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.5)
    assets.append(savefig(fig, "fig4_5_case_mean_thickness_error.png"))

    dist_summary = df.groupby("distance_mm")["thickness_error_mm"].agg(["mean", "std", "min", "max"]).loc[distance_order]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    x = np.arange(len(dist_summary))
    ax.bar(x, dist_summary["mean"], yerr=dist_summary["std"], capsize=4, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_xticks(x, [f"{d:g}" for d in dist_summary.index])
    ax.set_xlabel("Distance from free end (mm)")
    ax.set_ylabel("Mean thickness error (mm)")
    ax.set_title("Position effect on residual wall-thickness error")
    for xi, yi in zip(x, dist_summary["mean"]):
        ax.text(xi, yi + 0.014, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.5)
    assets.append(savefig(fig, "fig4_6_distance_mean_error.png"))

    piv = df.pivot_table(index=["case", "distance_mm"], columns="workpiece", values="thickness_error_mm")
    piv["abs_diff"] = (piv["W1"] - piv["W2"]).abs()
    # Repeatability is expressed as the mean absolute difference between paired
    # workpieces under the same case and distance.
    # 重复性用同工况、同距离下两件工件壁厚误差的平均绝对差表示。
    repeat = piv.groupby("case")["abs_diff"].mean().reindex(["S1", "S2", "S3", "S4", "S5-1", "S5-2", "S6"]).reset_index()
    s5_pair = (
        df.pivot_table(index=["workpiece", "distance_mm"], columns="case", values="thickness_error_mm")
        .assign(abs_diff=lambda x: (x["S5-1"] - x["S5-2"]).abs())
        ["abs_diff"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    x = np.arange(len(repeat))
    ax.bar(x, repeat["abs_diff"], color="#55a868", edgecolor="#2f6040")
    ax.axhline(s5_pair, color="#c44e52", linestyle="--", linewidth=1.4, label=f"S5-1 vs S5-2 mean = {s5_pair:.3f} mm")
    ax.set_xticks(x, repeat["case"])
    ax.set_xlabel("Case")
    ax.set_ylabel("Mean absolute difference (mm)")
    ax.set_title("Repeatability of thickness-error measurements")
    ax.set_ylim(0, max(0.080, repeat["abs_diff"].max() * 1.25))
    ax.legend(frameon=False, loc="upper left")
    for xi, yi in zip(x, repeat["abs_diff"]):
        ax.text(xi, yi + 0.003, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.0)
    assets.append(savefig(fig, "fig4_7_repeatability_error.png"))

    sample_summary = df.groupby("sample")["thickness_error_mm"].agg(["mean", "std", "min", "max"]).reset_index()
    df.to_csv(ANA / "thickness_measurements_long.csv", index=False, encoding="utf-8-sig")
    case_summary.to_csv(ANA / "thickness_case_summary.csv", index=False, encoding="utf-8-sig")
    dist_summary.reset_index().to_csv(ANA / "thickness_distance_summary.csv", index=False, encoding="utf-8-sig")
    sample_summary.to_csv(ANA / "thickness_sample_summary.csv", index=False, encoding="utf-8-sig")
    return assets


def make_roughness_figures(rough: pd.DataFrame, thick: pd.DataFrame) -> list[str]:
    """Create roughness charts and merge roughness with wall-thickness error.

    粗糙度结果使用批量脚本导出的 Gaussian 粗糙度面指标。合并表按工况和
    工件号连接三坐标平均壁厚误差与 Sa、Sq、Sz。

    Roughness results use Gaussian-surface metrics exported by the batch script.
    The merged table joins mean CMM wall-thickness error with Sa, Sq, and Sz by
    case and workpiece.
    """
    assets: list[str] = []
    order = ["TXF1-1", "TXF1-2", "TXF1-3", "TXF1-4", "TXF1-5-1", "TXF1-5-2", "TXF1-6", "TXF2-1", "TXF2-2", "TXF2-3", "TXF2-4", "TXF2-5-1", "TXF2-5-2", "TXF2-6"]
    rough = rough.set_index("folder").loc[order].reset_index()
    labels = rough["folder"].str.replace("TXF", "", regex=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    x = np.arange(len(rough))
    width = 0.38
    ax.bar(x - width / 2, rough["Sa_um"], width, label="$S_a$", color="#4c72b0")
    ax.bar(x + width / 2, rough["Sq_um"], width, label="$S_q$", color="#dd8452")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel("Measured surface")
    ax.set_ylabel("Areal roughness (um)")
    ax.set_title("Areal roughness parameters $S_a$ and $S_q$")
    ax.legend(frameon=False, ncol=2)
    assets.append(savefig(fig, "fig4_13_roughness_sa_sq_by_surface.png"))

    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    ax.plot(x, rough["Sz_um"], marker="o", linewidth=1.7, color="#c44e52")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel("Measured surface")
    ax.set_ylabel("$S_z$ (um)")
    ax.set_title("Maximum height of Gaussian roughness surface")
    assets.append(savefig(fig, "fig4_14_roughness_sz_by_surface.png"))

    thick_mean = thick.groupby(["case", "workpiece"], as_index=False)["thickness_error_mm"].mean()
    merged = thick_mean.merge(rough[["case", "workpiece", "Sa_um", "Sq_um", "Sz_um"]], on=["case", "workpiece"], how="inner")
    # This scatter supports the Chapter 4 discussion that wall-thickness error
    # and areal roughness are evaluated as separate quality indices.
    # 该散点图用于支撑第 4 章关于壁厚误差和面粗糙度分属不同质量指标的讨论。
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for wp, color, marker in [("W1", "#4c72b0", "o"), ("W2", "#dd8452", "s")]:
        sub = merged[merged["workpiece"] == wp]
        ax.scatter(sub["thickness_error_mm"], sub["Sa_um"], s=48, color=color, marker=marker, label=wp, edgecolor="white", linewidth=0.7)
        for _, r in sub.iterrows():
            ax.text(r["thickness_error_mm"] + 0.004, r["Sa_um"] + 0.004, r["case"].replace("S", ""), fontsize=7.5)
    ax.set_xlabel("Mean thickness error (mm)")
    ax.set_ylabel("$S_a$ (um)")
    ax.set_title("Relation between wall-thickness error and $S_a$")
    ax.legend(frameon=False)
    assets.append(savefig(fig, "fig4_15_thickness_error_vs_sa.png"))

    rough_summary = rough.groupby("sample")[["Sa_um", "Sq_um", "Sz_um"]].agg(["mean", "std", "min", "max"]).round(6)
    rough.to_csv(ANA / "roughness_measurements_for_thesis.csv", index=False, encoding="utf-8-sig")
    rough_summary.to_csv(ANA / "roughness_sample_summary.csv", encoding="utf-8-sig")
    merged.to_csv(ANA / "merged_thickness_roughness.csv", index=False, encoding="utf-8-sig")
    return assets


def make_profile_map(workpiece: int, name: str) -> str:
    """Render Gaussian roughness height maps for one workpiece.

    每个样件文件夹读取 `gaussian_filtered_point_cloud.csv`，按 x、y 网格
    还原粗糙度高度面，再统一颜色范围绘制。

    For each sample folder, `gaussian_filtered_point_cloud.csv` is reshaped into
    an x-y roughness-height grid and plotted with a shared color range.
    """
    folders = [
        f"TXF{workpiece}-1",
        f"TXF{workpiece}-2",
        f"TXF{workpiece}-3",
        f"TXF{workpiece}-4",
        f"TXF{workpiece}-5-1",
        f"TXF{workpiece}-5-2",
        f"TXF{workpiece}-6",
    ]
    labels = ["S1", "S2", "S3", "S4", "S5-1", "S5-2", "S6"]
    grids = []
    for folder in folders:
        p = ROOT / "output" / "surface_roughness" / folder / "gaussian_filtered_point_cloud.csv"
        data = pd.read_csv(p)
        grid = data.pivot(index="y_um", columns="x_um", values="gaussian_roughness_z_um").sort_index()
        grids.append(grid.values)

    all_vals = np.concatenate([grid.ravel() for grid in grids])
    vmin, vmax = np.percentile(all_vals, [2, 98])
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.45), constrained_layout=True)
    images = []
    for ax, grid, label, letter in zip(axes.flat, grids, labels, list("abcdefg"), strict=False):
        image = ax.imshow(
            grid,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=[0, 1000, 0, 1000],
            interpolation="nearest",
        )
        images.append(image)
        ax.set_title(f"({letter}) {label}", loc="left", fontsize=9.2, pad=3)
        ax.set_xticks([0, 500, 1000])
        ax.set_yticks([0, 500, 1000])
        ax.tick_params(labelsize=7, length=2)
    axes.flat[-1].axis("off")
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    for ax in axes[1, :3]:
        ax.set_xlabel("x (um)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (um)", fontsize=8)
    cbar = fig.colorbar(images[0], ax=axes[:, :], shrink=0.86, pad=0.012)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Gaussian roughness height (um)", fontsize=8)
    fig.suptitle(f"Gaussian roughness height maps of Workpiece {workpiece}", fontsize=12)
    return savefig(fig, name)


def main() -> None:
    """Run the Chapter 4 asset-building workflow.

    主流程为：导出照片、解析三坐标壁厚数据、读取批量粗糙度结果、生成
    壁厚和粗糙度图件、绘制两个工件的粗糙度高度图，并写出 manifest。

    The workflow exports photos, parses CMM thickness data, reads batch
    roughness results, generates thickness and roughness figures, renders
    roughness maps for two workpieces, and writes a manifest.
    """
    photo_assets = {
        "fig4_1_experimental_equipment.jpg": export_photo(DATA_ROOT / "photos" / "experimental_equipment.png", "fig4_1_experimental_equipment.jpg", max_width=2600),
        "fig4_2_specimen_tool.jpg": export_photo(DATA_ROOT / "photos" / "specimen_tool.png", "fig4_2_specimen_tool.jpg", max_width=2100),
        "fig4_8_tool_wear.jpg": export_photo(DATA_ROOT / "photos" / "tool_wear_before_after.png", "fig4_8_tool_wear.jpg", max_width=2100),
        "fig4_9_optical_morphology.jpg": export_photo(DATA_ROOT / "photos" / "optical_surface_morphology.png", "fig4_9_optical_morphology.jpg", max_width=1900),
        "fig4_10_sem_morphology.jpg": export_photo(DATA_ROOT / "photos" / "sem_surface_morphology.png", "fig4_10_sem_morphology.jpg", max_width=2200),
    }
    thick = parse_thickness(DATA_ROOT / "experimental" / "cmm_wall_thickness.xlsx")
    rough = load_roughness(ROOT / "output" / "surface_roughness" / "batch_roughness_results.csv")
    generated = [make_measurement_scheme()]
    generated.extend(make_thickness_figures(thick))
    generated.extend(make_roughness_figures(rough, thick))
    photo_assets["fig4_11_profile_workpiece_1.png"] = make_profile_map(1, "fig4_11_profile_workpiece_1.png")
    photo_assets["fig4_12_profile_workpiece_2.png"] = make_profile_map(2, "fig4_12_profile_workpiece_2.png")

    manifest = {
        "photo_assets": photo_assets,
        "generated_charts": generated,
        "overall_thickness_error": {
            "mean_mm": float(thick["thickness_error_mm"].mean()),
            "std_mm": float(thick["thickness_error_mm"].std()),
            "min_mm": float(thick["thickness_error_mm"].min()),
            "max_mm": float(thick["thickness_error_mm"].max()),
        },
        "overall_roughness": {
            "Sa_mean_um": float(rough["Sa_um"].mean()),
            "Sq_mean_um": float(rough["Sq_um"].mean()),
            "Sz_mean_um": float(rough["Sz_um"].mean()),
        },
    }
    (ANA / "thesis_asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
