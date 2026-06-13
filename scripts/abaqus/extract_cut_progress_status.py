#!/usr/bin/env python3
"""
从升级后的 ODB 中读取工件 STATUS 场，识别实际切削开始、四分位进程和
切削完成帧。

位移主响应窗口只描述变形能量集中的时段；STATUS 里程碑给出材料实际
被切除的进程。本脚本输出逐帧失效单元比例和里程碑表，供后续厚度提取、
实际进程对齐和出图脚本共同使用。

Read the workpiece STATUS field from upgraded ODB files and identify cutting
start, quarter-progress frames, and cut completion. The exported framewise table
and milestone table are shared by thickness extraction, actual-progress
alignment, and figure scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from odbAccess import openOdb  # type: ignore


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract actual cutting progress from STATUS fields in upgraded ODBs.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "scripts" / "abaqus" / "case_config.yml"),
        help="Case configuration file.",
    )
    parser.add_argument(
        "--odbdir",
        default=str(ROOT / "output" / "odb_upgraded"),
        help="Directory containing upgraded ODB files.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "output" / "spreadsheet" / "abaqus"),
        help="Directory for exported CSV/JSON files.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        return load_config_fallback(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config_fallback(path: str | Path) -> dict:
    case_pattern = re.compile(r"^(?P<indent>\s+)(?P<case>S[1-6]):\s*$", re.IGNORECASE)
    key_value_pattern = re.compile(r"^\s+(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.*)$")
    config: dict[str, dict] = {"cases": {}}
    current_case: str | None = None
    current_indent = 0

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            case_match = case_pattern.match(line)
            if case_match:
                current_case = case_match.group("case").upper()
                current_indent = len(case_match.group("indent"))
                config["cases"][current_case] = {}
                continue
            if current_case is None:
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent <= current_indent:
                current_case = None
                continue
            match = key_value_pattern.match(line)
            if not match:
                continue
            key = match.group("key")
            value = match.group("value").strip()
            if value == "":
                parsed: str | None = None
            else:
                parsed = value.strip("'\"")
            config["cases"][current_case][key] = parsed
    return config


def resolve_odb_path(odbdir: Path, case_id: str, odb_filename: str) -> Path:
    preferred = odbdir / f"{Path(odb_filename).stem}_upg.odb"
    if preferred.exists():
        return preferred
    matches = sorted(odbdir.glob(f"{case_id}*.odb"))
    if not matches:
        raise FileNotFoundError(f"No upgraded ODB found for {case_id} in {odbdir}")
    return matches[0]


def first_geq(lo: int, hi: int, predicate) -> int:
    answer = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return answer


def build_effective_window(step, failed_at, start_index: int, done_index: int, case_cfg: dict) -> dict[str, int | float | str]:
    keep_fraction = case_cfg.get("path_keep_fraction")
    if keep_fraction in (None, ""):
        keep_fraction = 1.0
    keep_fraction = float(keep_fraction)
    if keep_fraction <= 0.0 or keep_fraction > 1.0:
        raise ValueError(f"path_keep_fraction 非法: {keep_fraction}")

    if keep_fraction >= 0.999999 or done_index <= start_index:
        start_failed = int(failed_at(start_index))
        done_failed = int(failed_at(done_index))
        quarter_1_index = first_geq(start_index, done_index, lambda idx: failed_at(idx) >= start_failed + 0.25 * (done_failed - start_failed))
        quarter_2_index = first_geq(start_index, done_index, lambda idx: failed_at(idx) >= start_failed + 0.50 * (done_failed - start_failed))
        quarter_3_index = first_geq(start_index, done_index, lambda idx: failed_at(idx) >= start_failed + 0.75 * (done_failed - start_failed))
        return {
            "progress_mode": "status_failed_fraction",
            "path_keep_fraction": keep_fraction,
            "cut_start_frame_raw": int(start_index),
            "cut_q1_frame_raw": int(quarter_1_index),
            "cut_q2_frame_raw": int(quarter_2_index),
            "cut_q3_frame_raw": int(quarter_3_index),
            "cut_done_frame_raw": int(done_index),
            "cut_start_time_s": float(step.frames[start_index].frameValue),
            "cut_q1_time_s": float(step.frames[quarter_1_index].frameValue),
            "cut_q2_time_s": float(step.frames[quarter_2_index].frameValue),
            "cut_q3_time_s": float(step.frames[quarter_3_index].frameValue),
            "cut_done_time_s": float(step.frames[done_index].frameValue),
            "window_start_failed_element_count": start_failed,
            "window_done_failed_element_count": done_failed,
        }

    start_time = float(step.frames[start_index].frameValue)
    done_time = float(step.frames[done_index].frameValue)
    duration = done_time - start_time
    trim = 0.5 * (1.0 - keep_fraction) * duration
    trimmed_start_time = start_time + trim
    trimmed_done_time = done_time - trim

    trimmed_start_index = first_geq(start_index, done_index, lambda idx: float(step.frames[idx].frameValue) >= trimmed_start_time)
    trimmed_done_index = first_geq(trimmed_start_index, done_index, lambda idx: float(step.frames[idx].frameValue) >= trimmed_done_time)
    start_failed = int(failed_at(trimmed_start_index))
    done_failed = int(failed_at(trimmed_done_index))
    failed_span = max(done_failed - start_failed, 1)
    quarter_1_index = first_geq(trimmed_start_index, trimmed_done_index, lambda idx: failed_at(idx) >= start_failed + 0.25 * failed_span)
    quarter_2_index = first_geq(trimmed_start_index, trimmed_done_index, lambda idx: failed_at(idx) >= start_failed + 0.50 * failed_span)
    quarter_3_index = first_geq(trimmed_start_index, trimmed_done_index, lambda idx: failed_at(idx) >= start_failed + 0.75 * failed_span)
    return {
        "progress_mode": "center_path_trimmed",
        "path_keep_fraction": keep_fraction,
        "cut_start_frame_raw": int(trimmed_start_index),
        "cut_q1_frame_raw": int(quarter_1_index),
        "cut_q2_frame_raw": int(quarter_2_index),
        "cut_q3_frame_raw": int(quarter_3_index),
        "cut_done_frame_raw": int(trimmed_done_index),
        "cut_start_time_s": float(step.frames[trimmed_start_index].frameValue),
        "cut_q1_time_s": float(step.frames[quarter_1_index].frameValue),
        "cut_q2_time_s": float(step.frames[quarter_2_index].frameValue),
        "cut_q3_time_s": float(step.frames[quarter_3_index].frameValue),
        "cut_done_time_s": float(step.frames[trimmed_done_index].frameValue),
        "window_start_failed_element_count": start_failed,
        "window_done_failed_element_count": done_failed,
    }


def build_case_rows(odb_path: Path, step_name: str, instance_name: str, case_id: str, case_cfg: dict) -> tuple[list[dict], dict]:
    odb = openOdb(str(odb_path), readOnly=True)
    step = odb.steps[step_name]
    instance = odb.rootAssembly.instances[instance_name]

    failed_counts: dict[int, int] = {}
    rows: list[dict] = []

    def failed_at(frame_index: int) -> int:
        if frame_index not in failed_counts:
            status = step.frames[frame_index].fieldOutputs["STATUS"].getSubset(region=instance)
            failed_counts[frame_index] = sum(1 for value in status.values if value.data < 0.999)
        return failed_counts[frame_index]

    end_index = len(step.frames) - 1
    final_failed = failed_at(end_index)
    start_index = first_geq(0, end_index, lambda idx: failed_at(idx) > 0)
    quarter_1_index = first_geq(start_index, end_index, lambda idx: failed_at(idx) >= 0.25 * final_failed)
    quarter_2_index = first_geq(start_index, end_index, lambda idx: failed_at(idx) >= 0.50 * final_failed)
    quarter_3_index = first_geq(start_index, end_index, lambda idx: failed_at(idx) >= 0.75 * final_failed)
    done_index = first_geq(start_index, end_index, lambda idx: failed_at(idx) >= final_failed)
    effective = build_effective_window(step, failed_at, start_index, done_index, case_cfg)
    effective_start = int(effective["cut_start_frame_raw"])
    effective_done = int(effective["cut_done_frame_raw"])
    effective_start_failed = int(effective["window_start_failed_element_count"])
    effective_done_failed = int(effective["window_done_failed_element_count"])
    effective_span = max(effective_done_failed - effective_start_failed, 1)

    for frame_index, frame in enumerate(step.frames):
        failed_count = failed_at(frame_index)
        failed_ratio = 0.0 if final_failed == 0 else failed_count / float(final_failed)
        in_cut_window = effective_start <= frame_index <= effective_done
        eta = None
        if in_cut_window:
            eta = max(0.0, min(1.0, (failed_count - effective_start_failed) / float(effective_span)))
        rows.append(
            {
                "case": case_id,
                "frame_raw": frame_index,
                "frame_id": int(getattr(frame, "frameId", frame_index)),
                "increment_number": int(getattr(frame, "incrementNumber", frame_index)),
                "time_s": float(frame.frameValue),
                "failed_element_count": int(failed_count),
                "failed_ratio": float(failed_ratio),
                "is_cut_started": frame_index >= effective_start,
                "is_cut_done": frame_index >= effective_done,
                "in_cut_window": bool(in_cut_window),
                "eta": eta,
            }
        )

    milestones = {
        "case": case_id,
        "progress_mode": str(effective["progress_mode"]),
        "path_keep_fraction": float(effective["path_keep_fraction"]),
        "status_cut_start_frame_raw": int(start_index),
        "status_cut_q1_frame_raw": int(quarter_1_index),
        "status_cut_q2_frame_raw": int(quarter_2_index),
        "status_cut_q3_frame_raw": int(quarter_3_index),
        "status_cut_done_frame_raw": int(done_index),
        "cut_start_frame_raw": effective_start,
        "cut_q1_frame_raw": int(effective["cut_q1_frame_raw"]),
        "cut_q2_frame_raw": int(effective["cut_q2_frame_raw"]),
        "cut_q3_frame_raw": int(effective["cut_q3_frame_raw"]),
        "cut_done_frame_raw": effective_done,
        "analysis_end_frame_raw": int(end_index),
        "final_failed_element_count": int(final_failed),
        "window_start_failed_element_count": effective_start_failed,
        "window_done_failed_element_count": effective_done_failed,
        "status_cut_start_time_s": float(step.frames[start_index].frameValue),
        "status_cut_q1_time_s": float(step.frames[quarter_1_index].frameValue),
        "status_cut_q2_time_s": float(step.frames[quarter_2_index].frameValue),
        "status_cut_q3_time_s": float(step.frames[quarter_3_index].frameValue),
        "status_cut_done_time_s": float(step.frames[done_index].frameValue),
        "cut_start_time_s": float(effective["cut_start_time_s"]),
        "cut_q1_time_s": float(effective["cut_q1_time_s"]),
        "cut_q2_time_s": float(effective["cut_q2_time_s"]),
        "cut_q3_time_s": float(effective["cut_q3_time_s"]),
        "cut_done_time_s": float(effective["cut_done_time_s"]),
        "analysis_end_time_s": float(step.frames[end_index].frameValue),
    }

    odb.close()
    return rows, milestones


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    odbdir = Path(args.odbdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    milestone_rows: list[dict] = []

    for case_id, case_cfg in sorted(config["cases"].items()):
        odb_path = resolve_odb_path(odbdir, case_id, case_cfg["odb_filename"])
        instance_name = case_cfg.get("instance_name") or f"{case_id}-1"
        step_name = case_cfg.get("step_name") or "Step-1"
        rows, milestones = build_case_rows(odb_path, step_name, instance_name, case_id, case_cfg)
        all_rows.extend(rows)
        milestone_rows.append(milestones)

    all_rows.sort(key=lambda item: (item["case"], item["frame_raw"]))
    milestone_rows.sort(key=lambda item: item["case"])

    all_csv = outdir / "cut_progress_all_frames.csv"
    with all_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    milestones_csv = outdir / "cut_progress_milestones.csv"
    with milestones_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(milestone_rows[0].keys()))
        writer.writeheader()
        writer.writerows(milestone_rows)

    (outdir / "cut_progress_milestones.json").write_text(
        json.dumps(milestone_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {all_csv}")
    print(f"Wrote {milestones_csv}")
    print(f"Wrote {outdir / 'cut_progress_milestones.json'}")


if __name__ == "__main__":
    main()
