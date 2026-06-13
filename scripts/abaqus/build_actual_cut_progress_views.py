#!/usr/bin/env python3
"""
将位移、能量和真壁厚结果统一映射到实际切削进程 eta。

输入来自位移对齐、真壁厚汇总和 STATUS 里程碑。脚本生成出图用的
`aligned_energy_by_cut_progress.csv`、`aligned_line_history_by_cut_progress.csv`
和 `aligned_true_thickness_by_cut_progress.csv`，用于比较不同算例在同一
切削阶段上的响应差异。

Map displacement, energy, and true-thickness results onto the actual cutting
progress coordinate eta. The exported tables are figure-ready inputs for
cross-case comparison at the same cutting stage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 STATUS 实际切削进程构建出图中间数据。")
    parser.add_argument(
        "--datadir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="Abaqus 表格数据目录。",
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


def load_inputs(datadir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_metrics = pd.read_csv(datadir / "raw_displacement_metrics.csv.gz")
    thickness_summary = pd.read_csv(datadir / "true_thickness_summary_all_cases.csv")
    cut_progress = pd.read_csv(datadir / "cut_progress_all_frames.csv")
    milestones = pd.read_csv(datadir / "cut_progress_milestones.csv")
    for frame in (raw_metrics, thickness_summary, cut_progress, milestones):
        frame["case"] = frame["case"].astype(str).str.upper()
    raw_metrics["line_id"] = raw_metrics["line_id"].astype(str).str.zfill(2)
    return raw_metrics, thickness_summary, cut_progress, milestones


def augment_with_cut_progress(data: pd.DataFrame, cut_progress: pd.DataFrame, milestones: pd.DataFrame) -> pd.DataFrame:
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
    ]
    progress_columns = ["case", "frame_raw", "failed_element_count", "failed_ratio", "time_s", "eta", "in_cut_window"]

    drop_columns = [
        column
        for column in [
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
            "failed_element_count",
            "failed_ratio",
            "time_s",
            "eta",
            "in_cut_window",
        ]
        if column in data.columns
    ]
    frame = data.drop(columns=drop_columns).merge(milestones[milestone_columns], on="case", how="left")
    frame = frame.merge(cut_progress[progress_columns], on=["case", "frame_raw"], how="left")
    return frame


def interpolate_columns(base: pd.DataFrame, target_frames: np.ndarray, value_columns: list[str]) -> pd.DataFrame:
    base = base.sort_values("frame_raw").drop_duplicates(subset=["frame_raw"], keep="last")
    frame_values = base["frame_raw"].to_numpy(dtype=float)
    payload = {"frame_raw": target_frames.astype(int)}
    for column in value_columns:
        y_values = base[column].to_numpy(dtype=float)
        if len(frame_values) == 1:
            payload[column] = np.full_like(target_frames, fill_value=y_values[0], dtype=float)
        else:
            payload[column] = np.interp(target_frames, frame_values, y_values, left=y_values[0], right=y_values[-1])
    return pd.DataFrame(payload)


def build_curve_with_boundaries(
    base: pd.DataFrame,
    cut_progress_case: pd.DataFrame,
    start_frame: int,
    done_frame: int,
    value_columns: list[str],
) -> pd.DataFrame:
    in_window_frames = sorted(int(frame) for frame in base["frame_raw"].unique() if start_frame <= frame <= done_frame)
    target_frames = np.array(sorted(set([start_frame, done_frame] + in_window_frames)), dtype=int)
    curve = interpolate_columns(base, target_frames, value_columns)
    curve = curve.merge(cut_progress_case[["frame_raw", "eta", "time_s", "failed_ratio"]], on="frame_raw", how="left")
    curve["in_cut_window"] = curve["frame_raw"].between(start_frame, done_frame)
    curve = curve[curve["in_cut_window"] & curve["eta"].notna()].copy()
    return curve.sort_values(["eta", "frame_raw"]).reset_index(drop=True)


def resample_curve(curve: pd.DataFrame, value_columns: list[str], grid_size: int) -> pd.DataFrame:
    eta_grid = np.linspace(0.0, 1.0, grid_size)
    grouped = curve.groupby("eta", as_index=False)[value_columns].mean().sort_values("eta")
    payload = {"eta": eta_grid}
    x = grouped["eta"].to_numpy(dtype=float)
    for column in value_columns:
        y = grouped[column].to_numpy(dtype=float)
        if len(x) == 1:
            payload[column] = np.full_like(eta_grid, fill_value=y[0], dtype=float)
        else:
            payload[column] = np.interp(eta_grid, x, y, left=y[0], right=y[-1])
    return pd.DataFrame(payload)


def build_energy_alignment(raw_metrics: pd.DataFrame, cut_progress: pd.DataFrame, milestones: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    energy = (
        raw_metrics.groupby(["case", "frame_raw"], as_index=False)
        .agg(energy_smooth_mm=("energy_smooth_mm", "first"))
        .sort_values(["case", "frame_raw"])
    )
    rows: list[pd.DataFrame] = []
    for case_id, case_group in energy.groupby("case"):
        milestone_row = milestones[milestones["case"] == case_id].iloc[0]
        cp_case = cut_progress[cut_progress["case"] == case_id].copy()
        start_frame = int(milestone_row["cut_start_frame_raw"])
        done_frame = int(milestone_row["cut_done_frame_raw"])
        case_group = case_group.copy()
        peak = float(case_group["energy_smooth_mm"].max())
        case_group["energy_norm"] = case_group["energy_smooth_mm"] / peak if peak > 0 else 0.0
        cp_case = augment_with_cut_progress(cp_case, cut_progress, milestones)
        curve = build_curve_with_boundaries(case_group, cp_case, start_frame, done_frame, ["energy_norm"])
        aligned = resample_curve(curve, ["energy_norm"], grid_size)
        aligned.insert(0, "case", case_id)
        rows.append(aligned)
    return pd.concat(rows, ignore_index=True)


def build_line_histories(raw_metrics: pd.DataFrame, cut_progress: pd.DataFrame, milestones: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    line_metrics = (
        raw_metrics.groupby(["case", "line_id", "line_mm", "frame_raw"], as_index=False)
        .agg(w_peak_abs_mm=("w_peak_abs_mm", "max"))
        .sort_values(["case", "line_mm", "frame_raw"])
    )
    rows: list[pd.DataFrame] = []
    for (case_id, line_id, line_mm), group in line_metrics.groupby(["case", "line_id", "line_mm"]):
        milestone_row = milestones[milestones["case"] == case_id].iloc[0]
        cp_case = cut_progress[cut_progress["case"] == case_id].copy()
        cp_case = augment_with_cut_progress(cp_case, cut_progress, milestones)
        curve = build_curve_with_boundaries(
            group.copy(),
            cp_case,
            int(milestone_row["cut_start_frame_raw"]),
            int(milestone_row["cut_done_frame_raw"]),
            ["w_peak_abs_mm"],
        )
        aligned = resample_curve(curve, ["w_peak_abs_mm"], grid_size)
        aligned.insert(0, "line_mm", float(line_mm))
        aligned.insert(0, "line_id", str(line_id).zfill(2))
        aligned.insert(0, "case", case_id)
        rows.append(aligned)
    return pd.concat(rows, ignore_index=True)


def build_thickness_histories(
    thickness_summary: pd.DataFrame,
    cut_progress: pd.DataFrame,
    milestones: pd.DataFrame,
    grid_size: int,
) -> pd.DataFrame:
    summary = augment_with_cut_progress(thickness_summary, cut_progress, milestones)
    rows: list[pd.DataFrame] = []
    for case_id, group in summary.groupby("case"):
        milestone_row = milestones[milestones["case"] == case_id].iloc[0]
        cp_case = cut_progress[cut_progress["case"] == case_id].copy()
        cp_case = augment_with_cut_progress(cp_case, cut_progress, milestones)
        curve = build_curve_with_boundaries(
            group.copy(),
            cp_case,
            int(milestone_row["cut_start_frame_raw"]),
            int(milestone_row["cut_done_frame_raw"]),
            ["thickness_min_mm", "thickness_mean_mm", "thickness_p95_mm"],
        )
        aligned = resample_curve(curve, ["thickness_min_mm", "thickness_mean_mm", "thickness_p95_mm"], grid_size)
        aligned.insert(0, "case", case_id)
        rows.append(aligned)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    args = parse_args()
    datadir = Path(args.datadir)
    config = load_config(args.config)
    grid_size = int(config["global"]["displacement"]["grid_size"])

    raw_metrics, thickness_summary, cut_progress, milestones = load_inputs(datadir)
    aligned_energy = build_energy_alignment(raw_metrics, cut_progress, milestones, grid_size)
    aligned_line_histories = build_line_histories(raw_metrics, cut_progress, milestones, grid_size)
    aligned_thickness = build_thickness_histories(thickness_summary, cut_progress, milestones, grid_size)

    aligned_energy.to_csv(datadir / "aligned_energy_by_cut_progress.csv", index=False, encoding="utf-8-sig")
    aligned_line_histories.to_csv(datadir / "aligned_line_history_by_cut_progress.csv", index=False, encoding="utf-8-sig")
    aligned_thickness.to_csv(datadir / "aligned_true_thickness_by_cut_progress.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {datadir / 'aligned_energy_by_cut_progress.csv'}")
    print(f"已输出: {datadir / 'aligned_line_history_by_cut_progress.csv'}")
    print(f"已输出: {datadir / 'aligned_true_thickness_by_cut_progress.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
