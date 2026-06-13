#!/usr/bin/env python3
"""
按配置文件中的路径保留比例重写实际切削进程窗口。

该脚本用于 S3 这类总切削路径偏长的算例：先保留 STATUS 原始里程碑，
再把正式比较窗口裁剪到中段有效路径，并重新计算 eta、cut_start、q1、
q2、q3 和 cut_done。输出会覆盖同名里程碑与逐帧进程表。

Rewrite actual cutting-progress windows using case-specific path-keep rules.
For cases with overlong paths, the raw STATUS milestones are preserved while
the comparison window is trimmed to the centred valid segment and eta is rebuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjust cut-progress windows using case-specific path-keep rules.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "scripts" / "abaqus" / "case_config.yml"),
        help="Case configuration file.",
    )
    parser.add_argument(
        "--all-frames",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus" / "cut_progress_all_frames.csv"),
        help="Existing framewise cut-progress CSV.",
    )
    parser.add_argument(
        "--milestones",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus" / "cut_progress_milestones.csv"),
        help="Existing milestone CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="Output directory for rewritten cut-progress files.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def first_geq(values: pd.Series, target: float) -> int:
    matched = values[values >= target]
    if matched.empty:
        return int(values.index[-1])
    return int(matched.index[0])


def existing_status_frame(milestone_row: pd.Series, key: str) -> int:
    status_key = f"status_{key}"
    if status_key in milestone_row.index and pd.notna(milestone_row[status_key]):
        return int(milestone_row[status_key])
    return int(milestone_row[key])


def rewrite_case(case_rows: pd.DataFrame, milestone_row: pd.Series, case_cfg: dict) -> tuple[pd.DataFrame, dict]:
    case_rows = case_rows.sort_values("frame_raw").reset_index(drop=True)
    keep_fraction = case_cfg.get("path_keep_fraction")
    if keep_fraction in (None, ""):
        keep_fraction = 1.0
    keep_fraction = float(keep_fraction)
    if keep_fraction <= 0.0 or keep_fraction > 1.0:
        raise ValueError(f"{milestone_row['case']} 的 path_keep_fraction 非法: {keep_fraction}")

    status_start = existing_status_frame(milestone_row, "cut_start_frame_raw")
    status_q1 = existing_status_frame(milestone_row, "cut_q1_frame_raw")
    status_q2 = existing_status_frame(milestone_row, "cut_q2_frame_raw")
    status_q3 = existing_status_frame(milestone_row, "cut_q3_frame_raw")
    status_done = existing_status_frame(milestone_row, "cut_done_frame_raw")
    analysis_end = int(milestone_row["analysis_end_frame_raw"])

    by_frame = case_rows.set_index("frame_raw", drop=False).sort_index()
    status_start_time = float(by_frame.loc[status_start, "time_s"])
    status_done_time = float(by_frame.loc[status_done, "time_s"])

    if keep_fraction >= 0.999999 or status_done <= status_start:
        cut_start = status_start
        cut_done = status_done
        progress_mode = "status_failed_fraction"
    else:
        trim = 0.5 * (1.0 - keep_fraction) * (status_done_time - status_start_time)
        trimmed_start_time = status_start_time + trim
        trimmed_done_time = status_done_time - trim
        cut_start = first_geq(by_frame["time_s"], trimmed_start_time)
        cut_done = first_geq(by_frame["time_s"], trimmed_done_time)
        progress_mode = "center_path_trimmed"

    start_failed = int(by_frame.loc[cut_start, "failed_element_count"])
    done_failed = int(by_frame.loc[cut_done, "failed_element_count"])
    failed_span = max(done_failed - start_failed, 1)

    q1_target = start_failed + 0.25 * failed_span
    q2_target = start_failed + 0.50 * failed_span
    q3_target = start_failed + 0.75 * failed_span
    window_rows = by_frame.loc[cut_start:cut_done]
    cut_q1 = first_geq(window_rows["failed_element_count"], q1_target)
    cut_q2 = first_geq(window_rows["failed_element_count"], q2_target)
    cut_q3 = first_geq(window_rows["failed_element_count"], q3_target)

    rewritten = case_rows.copy()
    rewritten["is_cut_started"] = rewritten["frame_raw"] >= cut_start
    rewritten["is_cut_done"] = rewritten["frame_raw"] >= cut_done
    rewritten["in_cut_window"] = rewritten["frame_raw"].between(cut_start, cut_done)
    rewritten["eta"] = pd.NA
    in_window = rewritten["in_cut_window"]
    rewritten.loc[in_window, "eta"] = (
        (rewritten.loc[in_window, "failed_element_count"] - start_failed) / float(failed_span)
    ).clip(lower=0.0, upper=1.0)

    rewritten = rewritten[
        [
            "case",
            "frame_raw",
            "frame_id",
            "increment_number",
            "time_s",
            "failed_element_count",
            "failed_ratio",
            "is_cut_started",
            "is_cut_done",
            "in_cut_window",
            "eta",
        ]
    ]

    milestone = {
        "case": str(milestone_row["case"]).strip().upper(),
        "progress_mode": progress_mode,
        "path_keep_fraction": keep_fraction,
        "status_cut_start_frame_raw": status_start,
        "status_cut_q1_frame_raw": status_q1,
        "status_cut_q2_frame_raw": status_q2,
        "status_cut_q3_frame_raw": status_q3,
        "status_cut_done_frame_raw": status_done,
        "cut_start_frame_raw": cut_start,
        "cut_q1_frame_raw": cut_q1,
        "cut_q2_frame_raw": cut_q2,
        "cut_q3_frame_raw": cut_q3,
        "cut_done_frame_raw": cut_done,
        "analysis_end_frame_raw": analysis_end,
        "final_failed_element_count": int(milestone_row["final_failed_element_count"]),
        "window_start_failed_element_count": start_failed,
        "window_done_failed_element_count": done_failed,
        "status_cut_start_time_s": status_start_time,
        "status_cut_q1_time_s": float(by_frame.loc[status_q1, "time_s"]),
        "status_cut_q2_time_s": float(by_frame.loc[status_q2, "time_s"]),
        "status_cut_q3_time_s": float(by_frame.loc[status_q3, "time_s"]),
        "status_cut_done_time_s": status_done_time,
        "cut_start_time_s": float(by_frame.loc[cut_start, "time_s"]),
        "cut_q1_time_s": float(by_frame.loc[cut_q1, "time_s"]),
        "cut_q2_time_s": float(by_frame.loc[cut_q2, "time_s"]),
        "cut_q3_time_s": float(by_frame.loc[cut_q3, "time_s"]),
        "cut_done_time_s": float(by_frame.loc[cut_done, "time_s"]),
        "analysis_end_time_s": float(by_frame.loc[analysis_end, "time_s"]),
    }
    return rewritten, milestone


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_frames = pd.read_csv(args.all_frames)
    milestones = pd.read_csv(args.milestones)
    all_frames["case"] = all_frames["case"].astype(str).str.upper()
    milestones["case"] = milestones["case"].astype(str).str.upper()

    frame_rows: list[pd.DataFrame] = []
    milestone_rows: list[dict] = []
    for case_id, case_group in all_frames.groupby("case", sort=True):
        milestone_row = milestones[milestones["case"] == case_id].iloc[0]
        rewritten, milestone = rewrite_case(case_group, milestone_row, config["cases"][case_id])
        frame_rows.append(rewritten)
        milestone_rows.append(milestone)

    all_frames_out = pd.concat(frame_rows, ignore_index=True).sort_values(["case", "frame_raw"]).reset_index(drop=True)
    milestones_out = pd.DataFrame(milestone_rows).sort_values("case").reset_index(drop=True)

    all_csv = outdir / "cut_progress_all_frames.csv"
    milestones_csv = outdir / "cut_progress_milestones.csv"
    all_frames_out.to_csv(all_csv, index=False, encoding="utf-8-sig")
    milestones_out.to_csv(milestones_csv, index=False, encoding="utf-8-sig")
    (outdir / "cut_progress_milestones.json").write_text(
        json.dumps(milestone_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {all_csv}")
    print(f"Wrote {milestones_csv}")
    print(f"Wrote {outdir / 'cut_progress_milestones.json'}")


if __name__ == "__main__":
    main()
