#!/usr/bin/env python3
"""
从升级后的 ODB 中提取等效反力、Mises 应力分位值和接触压力统计。

脚本读取实际切削进程里程碑，在切削阶段内抽样帧，并把历史输出中的
RF1/RF2/RF3 与场输出中的 S、CPRESS 归并为论文第 3 章使用的表格。
ODB 文件名、几何参数和实例名均来自示意配置文件，开源使用时只需替换
`case_config.yml/json` 和数据目录。

Extract resultant reaction force, Mises stress percentiles, and contact-pressure
statistics from upgraded ODB files. Milestone frames define the active cutting
window, and the anonymized case configuration supplies ODB names, geometry, and
optional instance names.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RF, stress, and contact-load metrics from upgraded ODB files.")
    parser.add_argument("--config", default=str(ROOT / "scripts" / "abaqus" / "case_config.yml"))
    parser.add_argument("--odb-dir", default=str(ROOT / "output" / "odb_upgraded"))
    parser.add_argument("--datadir", default=str(ROOT / "output" / "spreadsheet" / "abaqus"))
    parser.add_argument("--sample-count", type=int, default=21)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    """Load YAML configuration with a JSON fallback for Abaqus Python runtimes.

    Abaqus 自带 Python 环境可能没有 PyYAML，因此 YAML 读取失败时回退到同名
    JSON 配置。

    Abaqus Python may not include PyYAML, so a same-name JSON file is used when
    YAML loading is unavailable.
    """
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        with path.with_suffix(".json").open("r", encoding="utf-8") as handle:
            return json.load(handle)


def resolve_upgraded_odb(odb_dir: Path, case_cfg: dict) -> Path:
    """Resolve the upgraded ODB path from the anonymized case configuration."""
    base_name = Path(case_cfg["odb_filename"]).stem
    return odb_dir / f"{base_name}_upg.odb"


def read_milestones(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        case = row["case"].upper()
        result[case] = {}
        for key, value in row.items():
            if key == "case" or value == "":
                continue
            try:
                result[case][key] = float(value)
            except ValueError:
                continue
    return result


def read_case_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        case = row["case"].upper()
        result[case] = {}
        for key, value in row.items():
            if key == "case" or value == "":
                continue
            try:
                result[case][key] = float(value)
            except ValueError:
                pass
    return result


def sample_frames(start_frame: int, done_frame: int, sample_count: int, extras: list[int]) -> list[int]:
    if sample_count < 2:
        sample_count = 2
    frames = {
        int(round(start_frame + (done_frame - start_frame) * i / float(sample_count - 1)))
        for i in range(sample_count)
    }
    frames.update(int(frame) for frame in extras)
    frames = {frame for frame in frames if start_frame <= frame <= done_frame}
    return sorted(frames)


def as_float(value) -> float:
    if isinstance(value, (tuple, list)):
        return float(value[0]) if value else 0.0
    return float(value)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def select_history_region(step):
    for region in step.historyRegions.values():
        names = set(region.historyOutputs.keys())
        if {"RF1", "RF2", "RF3"}.issubset(names):
            return region
    raise RuntimeError("No history region containing RF1/RF2/RF3 was found.")


def extract_force_history(step, start_time: float, done_time: float) -> tuple[list[dict[str, float]], dict[str, float]]:
    region = select_history_region(step)
    data1 = list(region.historyOutputs["RF1"].data)
    data2 = list(region.historyOutputs["RF2"].data)
    data3 = list(region.historyOutputs["RF3"].data)
    count = min(len(data1), len(data2), len(data3))
    rows: list[dict[str, float]] = []
    cut_values: list[float] = []

    for index in range(count):
        time_value = float(data1[index][0])
        rf1 = float(data1[index][1])
        rf2 = float(data2[index][1])
        rf3 = float(data3[index][1])
        force = math.sqrt(rf1 * rf1 + rf2 * rf2 + rf3 * rf3)
        eta = (time_value - start_time) / (done_time - start_time) if done_time > start_time else 0.0
        row = {
            "frame_raw": float(index),
            "time_s": time_value,
            "eta": eta,
            "rf1": rf1,
            "rf2": rf2,
            "rf3": rf3,
            "force_mag": force,
        }
        rows.append(row)
        if start_time <= time_value <= done_time:
            cut_values.append(force)

    peak_force = max(cut_values) if cut_values else 0.0
    peak_eta = 0.0
    if peak_force > 0.0:
        for row in rows:
            if start_time <= row["time_s"] <= done_time and abs(row["force_mag"] - peak_force) <= max(1e-9, peak_force * 1e-12):
                peak_eta = row["eta"]
                break

    summary = {
        "force_mean": mean(cut_values),
        "force_std": stdev(cut_values),
        "force_peak": peak_force,
        "force_peak_eta": peak_eta,
    }
    return rows, summary


def positive_values(field) -> list[float]:
    values: list[float] = []
    for value in field.values:
        data = as_float(value.data)
        if data > 0.0 and math.isfinite(data):
            values.append(data)
    return values


def find_contact_field(frame, prefix: str, case: str):
    target = "ASSEMBLY_%s-1" % case
    candidates = []
    for name, field in frame.fieldOutputs.items():
        if not name.strip().startswith(prefix):
            continue
        if target not in name:
            continue
        candidates.append((len(field.values), name, field))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def active_element_labels(frame, instance):
    if "STATUS" not in frame.fieldOutputs:
        return None
    status_field = frame.fieldOutputs["STATUS"].getSubset(region=instance)
    labels = set()
    for value in status_field.values:
        try:
            if as_float(value.data) > 0.5:
                labels.add(int(value.elementLabel))
        except Exception:
            continue
    return labels


def workpiece_stress_values(frame, instance) -> list[float]:
    active_labels = active_element_labels(frame, instance)
    if "MISESMAX" in frame.fieldOutputs:
        stress_field = frame.fieldOutputs["MISESMAX"].getSubset(region=instance)
        values = []
        for value in stress_field.values:
            if active_labels is not None and int(value.elementLabel) not in active_labels:
                continue
            data = as_float(value.data)
            if data > 0.0 and math.isfinite(data):
                values.append(data)
        return values

    if "S" not in frame.fieldOutputs:
        return []
    stress_field = frame.fieldOutputs["S"].getSubset(region=instance)
    values = []
    for value in stress_field.values:
        if active_labels is not None and int(value.elementLabel) not in active_labels:
            continue
        try:
            data = float(value.mises)
        except Exception:
            continue
        if data > 0.0 and math.isfinite(data):
            values.append(data)
    return values


def extract_field_rows(
    odb,
    case: str,
    instance_name: str,
    start_frame: int,
    done_frame: int,
    frames: list[int],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    step = list(odb.steps.values())[0]
    instance = odb.rootAssembly.instances[instance_name]
    rows: list[dict[str, float]] = []

    for frame_index in frames:
        frame = step.frames[frame_index]
        time_value = float(frame.frameValue)
        eta = (frame_index - start_frame) / float(done_frame - start_frame) if done_frame > start_frame else 0.0

        stresses = workpiece_stress_values(frame, instance)
        cpress_field = find_contact_field(frame, "CPRESS", case)
        cpress = positive_values(cpress_field) if cpress_field is not None else []

        row = {
            "frame_raw": float(frame_index),
            "time_s": time_value,
            "eta": eta,
            "mises_p95": percentile(stresses, 95.0),
            "mises_p99": percentile(stresses, 99.0),
            "mises_max": max(stresses) if stresses else 0.0,
            "mises_mean": mean(stresses),
            "active_stress_value_count": float(len(stresses)),
            "cpress_mean_positive": mean(cpress),
            "cpress_p95_positive": percentile(cpress, 95.0),
            "cpress_max": max(cpress) if cpress else 0.0,
            "cpress_positive_count": float(len(cpress)),
        }
        rows.append(row)

    summary = {
        "mises_p95_mean": mean([row["mises_p95"] for row in rows]),
        "mises_p95_std": stdev([row["mises_p95"] for row in rows]),
        "mises_p99_mean": mean([row["mises_p99"] for row in rows]),
        "mises_max_peak": max([row["mises_max"] for row in rows]) if rows else 0.0,
        "mises_max_peak_frame": max(rows, key=lambda row: row["mises_max"])["frame_raw"] if rows else 0.0,
        "cpress_mean_positive_mean": mean([row["cpress_mean_positive"] for row in rows]),
        "cpress_mean_positive_std": stdev([row["cpress_mean_positive"] for row in rows]),
        "cpress_max_peak": max([row["cpress_max"] for row in rows]) if rows else 0.0,
        "cpress_max_peak_frame": max(rows, key=lambda row: row["cpress_max"])["frame_raw"] if rows else 0.0,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    odb_dir = Path(args.odb_dir)
    datadir = Path(args.datadir)
    milestones = read_milestones(datadir / "cut_progress_milestones.csv")
    case_metrics = read_case_metrics(datadir / "thesis_case_metrics.csv")

    from odbAccess import openOdb  # type: ignore

    force_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case in CASE_ORDER:
        print("Processing %s" % case)
        case_cfg = config["cases"][case]
        odb_path = resolve_upgraded_odb(odb_dir, case_cfg)
        instance_name = case_cfg.get("instance_name") or f"{case}-1"
        milestone = milestones[case]
        start_frame = int(milestone["cut_start_frame_raw"])
        done_frame = int(milestone["cut_done_frame_raw"])
        start_time = float(milestone["cut_start_time_s"])
        done_time = float(milestone["cut_done_time_s"])
        extras = [
            int(milestone["cut_q1_frame_raw"]),
            int(milestone["cut_q2_frame_raw"]),
            int(milestone["cut_q3_frame_raw"]),
            int(case_metrics[case].get("peak_displacement_frame_raw", start_frame)),
            int(case_metrics[case].get("worst_thickness_frame_raw", start_frame)),
        ]
        frames = sample_frames(start_frame, done_frame, args.sample_count, extras)

        odb = openOdb(str(odb_path), readOnly=True)
        step = list(odb.steps.values())[0]
        force_history, force_summary = extract_force_history(step, start_time, done_time)
        fields, field_summary = extract_field_rows(odb, case, instance_name, start_frame, done_frame, frames)
        odb.close()

        radius = float(case_cfg["radius_mm"])
        thickness = float(case_cfg["nominal_thickness_mm"])
        for row in force_history:
            row_out = {"case": case, "radius_mm": radius, "nominal_thickness_mm": thickness}
            row_out.update(row)
            force_rows.append(row_out)
        for row in fields:
            row_out = {"case": case, "radius_mm": radius, "nominal_thickness_mm": thickness}
            row_out.update(row)
            field_rows.append(row_out)

        summary = {
            "case": case,
            "radius_mm": radius,
            "nominal_thickness_mm": thickness,
            "cut_start_frame_raw": start_frame,
            "cut_done_frame_raw": done_frame,
            "sampled_field_frame_count": len(frames),
        }
        summary.update(force_summary)
        summary.update(field_summary)
        summary_rows.append(summary)

    write_csv(
        datadir / "force_history_by_cut_progress.csv",
        force_rows,
        [
            "case",
            "radius_mm",
            "nominal_thickness_mm",
            "frame_raw",
            "time_s",
            "eta",
            "rf1",
            "rf2",
            "rf3",
            "force_mag",
        ],
    )
    write_csv(
        datadir / "stress_contact_by_cut_progress.csv",
        field_rows,
        [
            "case",
            "radius_mm",
            "nominal_thickness_mm",
            "frame_raw",
            "time_s",
            "eta",
            "mises_p95",
            "mises_p99",
            "mises_max",
            "mises_mean",
            "active_stress_value_count",
            "cpress_mean_positive",
            "cpress_p95_positive",
            "cpress_max",
            "cpress_positive_count",
        ],
    )
    write_csv(
        datadir / "force_stress_summary.csv",
        summary_rows,
        [
            "case",
            "radius_mm",
            "nominal_thickness_mm",
            "cut_start_frame_raw",
            "cut_done_frame_raw",
            "sampled_field_frame_count",
            "force_mean",
            "force_std",
            "force_peak",
            "force_peak_eta",
            "mises_p95_mean",
            "mises_p95_std",
            "mises_p99_mean",
            "mises_max_peak",
            "mises_max_peak_frame",
            "cpress_mean_positive_mean",
            "cpress_mean_positive_std",
            "cpress_max_peak",
            "cpress_max_peak_frame",
        ],
    )
    (datadir / "force_stress_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("Wrote force/stress metrics to %s" % datadir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
