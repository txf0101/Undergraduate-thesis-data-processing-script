#!/usr/bin/env python3
from __future__ import annotations

"""
汇总各算例的真壁厚结果，并生成后续画图需要的统一数据集。

输入：
1. `true_thickness_frame_summary_<case>.csv`：每帧统计量；
2. `true_thickness_fullfield_<case>.csv.gz`：切削区全场真壁厚；
3. `ks_ke_summary.csv`：位移主响应窗识别结果；
4. `cut_progress_all_frames.csv` / `cut_progress_milestones.csv`：基于 STATUS 的实际切削进程结果。

输出：
1. 全算例帧级汇总表；
2. 按实际切削进程对齐后的壁厚汇总表；
3. 按实际切削完成帧截取的真壁厚场；
4. 供 README、出图脚本和论文整理直接引用的关键指标 JSON。

This script merges framewise full-field thickness summaries, displacement
response windows, and STATUS-based cutting-progress milestones. It exports
case-level thickness histories, eta-aligned summaries, cut-completion fields,
and key metrics used by the figure pipeline.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 ODB 真壁厚统计结果。")
    parser.add_argument(
        "--datadir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="真壁厚与位移表格所在目录。",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "scripts" / "abaqus" / "case_config.yml"),
        help="案例配置文件路径。",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pick_frame_rows(group: pd.DataFrame, target_frame: int) -> pd.DataFrame:
    exact = group[group["frame_raw"] == target_frame].copy()
    if not exact.empty:
        return exact
    distances = (group["frame_raw"] - int(target_frame)).abs()
    nearest_frame = int(group.loc[distances.idxmin(), "frame_raw"])
    return group[group["frame_raw"] == nearest_frame].copy()


def build_key_metrics(summary_all: pd.DataFrame, config: dict) -> dict[str, object]:
    key_metrics: dict[str, object] = {"cases": {}, "comparisons": {}}

    for case_id, group in summary_all.groupby("case"):
        group = group.sort_values("frame_raw").reset_index(drop=True)
        min_row = group.loc[group["thickness_min_mm"].idxmin()]
        if "cut_done_frame_raw" in group.columns and group["cut_done_frame_raw"].notna().any():
            end_frame = int(group["cut_done_frame_raw"].dropna().iloc[0])
        elif "ke_raw" in group.columns and group["ke_raw"].notna().any():
            end_frame = int(group["ke_raw"].dropna().iloc[0])
        else:
            end_frame = int(group["frame_raw"].max())
        end_row = pick_frame_rows(group, end_frame).iloc[0]
        if "analysis_end_frame_raw" in group.columns and group["analysis_end_frame_raw"].notna().any():
            analysis_end_frame = int(group["analysis_end_frame_raw"].dropna().iloc[0])
        else:
            analysis_end_frame = int(group["frame_raw"].max())
        analysis_end_row = pick_frame_rows(group, analysis_end_frame).iloc[0]
        key_metrics["cases"][case_id] = {
            "nominal_thickness_mm": float(config["cases"][case_id]["nominal_thickness_mm"]),
            "radius_mm": float(config["cases"][case_id]["radius_mm"]),
            "worst_frame_raw": int(min_row["frame_raw"]),
            "minimum_true_thickness_mm": float(min_row["thickness_min_mm"]),
            "maximum_thinning_mm": float(min_row["max_thinning_mm"]),
            "minimum_location": {
                "theta_deg": float(min_row["min_theta_deg"]),
                "axial_mm": float(min_row["min_axial_mm"]),
            },
            "end_frame_raw": int(end_row["frame_raw"]),
            "target_cut_done_frame_raw": int(end_frame),
            "end_frame_min_thickness_mm": float(end_row["thickness_min_mm"]),
            "end_frame_mean_thickness_mm": float(end_row["thickness_mean_mm"]),
            "analysis_end_frame_raw": int(analysis_end_row["frame_raw"]),
            "analysis_end_min_thickness_mm": float(analysis_end_row["thickness_min_mm"]),
            "analysis_end_mean_thickness_mm": float(analysis_end_row["thickness_mean_mm"]),
        }

    peak_table = pd.DataFrame.from_dict(key_metrics["cases"], orient="index").reset_index(names="case")
    peak_table["radius_mm"] = peak_table["case"].map(lambda c: float(config["cases"][c]["radius_mm"]))
    peak_table["nominal_thickness_mm"] = peak_table["case"].map(lambda c: float(config["cases"][c]["nominal_thickness_mm"]))

    key_metrics["comparisons"] = {
        "max_thinning_by_case": peak_table[["case", "maximum_thinning_mm"]].to_dict(orient="records"),
        "mean_max_thinning_by_nominal_thickness": (
            peak_table.groupby("nominal_thickness_mm", as_index=False)["maximum_thinning_mm"].mean().to_dict(orient="records")
        ),
        "mean_max_thinning_by_radius": (
            peak_table.groupby("radius_mm", as_index=False)["maximum_thinning_mm"].mean().to_dict(orient="records")
        ),
    }
    return key_metrics


def main() -> None:
    args = parse_args()
    datadir = Path(args.datadir)
    config = load_config(args.config)

    summary_files = sorted(datadir.glob("true_thickness_frame_summary_*.csv"))
    fullfield_files = sorted(datadir.glob("true_thickness_fullfield_*.csv.gz"))
    if not summary_files:
        raise FileNotFoundError(f"未在 {datadir} 找到真壁厚帧级统计文件。")
    if not fullfield_files:
        raise FileNotFoundError(f"未在 {datadir} 找到真壁厚全场文件。")

    summary_all = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
    fullfield_all = pd.concat([pd.read_csv(path) for path in fullfield_files], ignore_index=True)
    summary_all["case"] = summary_all["case"].astype(str).str.upper()
    fullfield_all["case"] = fullfield_all["case"].astype(str).str.upper()

    ks_ke_path = datadir / "ks_ke_summary.csv"
    if ks_ke_path.exists():
        ks_ke = pd.read_csv(ks_ke_path)
        ks_ke["case"] = ks_ke["case"].astype(str).str.upper()
        summary_all = summary_all.merge(ks_ke[["case", "ks_raw", "ke_raw", "peak_frame_raw"]], on="case", how="left")
        fullfield_all = fullfield_all.merge(ks_ke[["case", "ks_raw", "ke_raw", "peak_frame_raw"]], on="case", how="left")
        summary_all["in_alignment_window"] = summary_all["frame_raw"].between(summary_all["ks_raw"], summary_all["ke_raw"])
        fullfield_all["in_alignment_window"] = fullfield_all["frame_raw"].between(fullfield_all["ks_raw"], fullfield_all["ke_raw"])
        summary_all["xi"] = (summary_all["frame_raw"] - summary_all["ks_raw"]) / (summary_all["ke_raw"] - summary_all["ks_raw"])
        fullfield_all["xi"] = (fullfield_all["frame_raw"] - fullfield_all["ks_raw"]) / (fullfield_all["ke_raw"] - fullfield_all["ks_raw"])
        summary_all.loc[~summary_all["in_alignment_window"], "xi"] = pd.NA
        fullfield_all.loc[~fullfield_all["in_alignment_window"], "xi"] = pd.NA
    else:
        summary_all["in_alignment_window"] = False
        fullfield_all["in_alignment_window"] = False
        summary_all["xi"] = pd.NA
        fullfield_all["xi"] = pd.NA

    cut_progress_path = datadir / "cut_progress_all_frames.csv"
    milestones_path = datadir / "cut_progress_milestones.csv"
    if cut_progress_path.exists() and milestones_path.exists():
        cut_progress = pd.read_csv(cut_progress_path)
        milestones = pd.read_csv(milestones_path)
        cut_progress["case"] = cut_progress["case"].astype(str).str.upper()
        milestones["case"] = milestones["case"].astype(str).str.upper()

        progress_columns = [
            "case",
            "frame_raw",
            "time_s",
            "failed_element_count",
            "failed_ratio",
            "is_cut_started",
            "is_cut_done",
            "in_cut_window",
            "eta",
        ]
        milestone_columns = [
            "case",
            "progress_mode",
            "path_keep_fraction",
            "status_cut_start_frame_raw",
            "status_cut_q1_frame_raw",
            "status_cut_q2_frame_raw",
            "status_cut_q3_frame_raw",
            "status_cut_done_frame_raw",
            "cut_start_frame_raw",
            "cut_q1_frame_raw",
            "cut_q2_frame_raw",
            "cut_q3_frame_raw",
            "cut_done_frame_raw",
            "analysis_end_frame_raw",
            "final_failed_element_count",
            "window_start_failed_element_count",
            "window_done_failed_element_count",
            "status_cut_start_time_s",
            "status_cut_q1_time_s",
            "status_cut_q2_time_s",
            "status_cut_q3_time_s",
            "status_cut_done_time_s",
            "cut_start_time_s",
            "cut_q1_time_s",
            "cut_q2_time_s",
            "cut_q3_time_s",
            "cut_done_time_s",
            "analysis_end_time_s",
        ]
        summary_all = summary_all.merge(milestones[milestone_columns], on="case", how="left")
        fullfield_all = fullfield_all.merge(milestones[milestone_columns], on="case", how="left")
        summary_all = summary_all.merge(cut_progress[progress_columns], on=["case", "frame_raw"], how="left")
        fullfield_all = fullfield_all.merge(cut_progress[progress_columns], on=["case", "frame_raw"], how="left")
    else:
        summary_all["in_cut_window"] = False
        fullfield_all["in_cut_window"] = False
        summary_all["eta"] = pd.NA
        fullfield_all["eta"] = pd.NA

    aligned_summary = summary_all[summary_all["in_cut_window"]].copy()
    end_field_rows = []
    for case_id, group in fullfield_all.groupby("case"):
        if "cut_done_frame_raw" in group.columns and group["cut_done_frame_raw"].notna().any():
            end_frame = int(group["cut_done_frame_raw"].dropna().iloc[0])
        elif "ke_raw" in group.columns and group["ke_raw"].notna().any():
            end_frame = int(group["ke_raw"].dropna().iloc[0])
        else:
            end_frame = int(group["frame_raw"].max())
        subset = pick_frame_rows(group, end_frame)
        subset["end_frame_raw"] = int(subset["frame_raw"].iloc[0])
        subset["target_end_frame_raw"] = end_frame
        end_field_rows.append(subset)
    end_field = pd.concat(end_field_rows, ignore_index=True)

    key_metrics = build_key_metrics(summary_all, config)

    summary_all.to_csv(datadir / "true_thickness_summary_all_cases.csv", index=False, encoding="utf-8-sig")
    aligned_summary.to_csv(datadir / "aligned_true_thickness_summary.csv", index=False, encoding="utf-8-sig")
    end_field.to_csv(datadir / "true_thickness_end_field.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    (datadir / "true_thickness_key_metrics.json").write_text(
        json.dumps(key_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已输出: {datadir / 'true_thickness_summary_all_cases.csv'}")
    print(f"已输出: {datadir / 'aligned_true_thickness_summary.csv'}")
    print(f"已输出: {datadir / 'true_thickness_end_field.csv.gz'}")
    print(f"已输出: {datadir / 'true_thickness_key_metrics.json'}")


if __name__ == "__main__":
    main()
