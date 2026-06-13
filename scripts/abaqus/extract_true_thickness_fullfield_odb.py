#!/usr/bin/env python3
from __future__ import annotations

"""
在 Abaqus Python 环境下，从升级后的 ODB 中直接重建切削区真壁厚结果。

核心思路：
1. 自动识别工件实例，而不是误把刀具实例当作分析对象；
2. 在初始构型下按“圆弧角度 + 轴向位置”把内外表面节点配对；
3. 对每个目标帧重新计算配对节点的当前半径差，得到真壁厚；
4. 输出全场壁厚明细和每帧统计量，供后续画图与论文分析使用。

Run inside Abaqus Python. The script pairs inner and outer surface nodes by
initial angular and axial buckets, then recomputes deformed radial distances at
target frames. It produces full-field thickness rows and frame summaries for
plotting and manuscript tables.
"""

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从升级后的 ODB 提取切削区真壁厚结果。")
    parser.add_argument("--odb", required=True, help="升级后的 ODB 文件路径。")
    parser.add_argument("--case", required=True, help="算例编号，例如 S1。")
    parser.add_argument("--config", required=True, help="案例配置文件路径。")
    parser.add_argument("--out", required=True, help="结果输出目录。")
    parser.add_argument(
        "--frames",
        default="",
        help="额外提取的原始帧号，逗号分隔；会与配置中的标准帧取并集。",
    )
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    config_path = Path(path)
    try:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        json_path = config_path.with_suffix(".json")
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def normalize_case_id(case_token: str) -> str:
    return case_token.strip().upper()


def axis_bundle(coord: tuple[float, float, float], axial_axis: str) -> tuple[float, float, float]:
    if axial_axis == "Z":
        return coord[0], coord[1], coord[2]
    if axial_axis == "Y":
        return coord[0], coord[2], coord[1]
    return coord[1], coord[2], coord[0]


def theta_deg(r1: float, r2: float) -> float:
    angle = math.degrees(math.atan2(r2, r1))
    return angle if angle >= 0 else angle + 360.0


def round_bucket(value: float, step: float) -> float:
    return round(value / step) * step


def select_instance(odb, case_id: str, case_cfg: dict):
    assembly = odb.rootAssembly
    explicit_name = case_cfg.get("instance_name")
    if explicit_name:
        return assembly.instances[explicit_name]

    candidates = []
    for instance_name, instance in assembly.instances.items():
        score = len(instance.nodes)
        if instance_name.upper().startswith(case_id.upper()):
            score += 10**9
        candidates.append((score, instance_name, instance))

    if not candidates:
        raise RuntimeError("ODB 中未找到任何实例。")

    candidates.sort(reverse=True)
    return candidates[0][2]


def select_step(odb, case_cfg: dict):
    step_name = case_cfg.get("step_name") or "Step-1"
    if step_name in odb.steps:
        return odb.steps[step_name]
    return odb.steps[sorted(odb.steps.keys())[0]]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classify_surface(radius: float, case_cfg: dict) -> str:
    nominal_outer = float(case_cfg["radius_mm"]) + float(case_cfg["nominal_thickness_mm"]) / 2.0
    nominal_inner = float(case_cfg["reference_inner_radius_mm"])
    midpoint = (nominal_outer + nominal_inner) / 2.0
    return "outer" if radius >= midpoint else "inner"


def build_surface_pair_map(instance, case_cfg: dict, global_cfg: dict) -> list[dict]:
    axial_axis = str(global_cfg["thickness"].get("axis_name", "Z")).upper()
    theta_step = float(global_cfg["thickness"]["theta_round_deg"])
    axial_step = float(global_cfg["thickness"]["z_round_mm"])
    max_line_position = max(float(value) for value in global_cfg["line_positions_mm"].values())
    cutting_depth = max_line_position + 0.2

    grouped: dict[tuple[float, float], dict[str, tuple[int, tuple[float, float, float], float]]] = defaultdict(dict)
    axial_values: list[float] = []

    for node in instance.nodes:
        coord = tuple(float(value) for value in node.coordinates)
        radial_1, radial_2, axial_value = axis_bundle(coord, axial_axis)
        radius = math.hypot(radial_1, radial_2)
        surface_name = classify_surface(radius, case_cfg)
        key = (round_bucket(theta_deg(radial_1, radial_2), theta_step), round_bucket(axial_value, axial_step))
        axial_values.append(axial_value)

        current = grouped[key].get(surface_name)
        if current is None:
            grouped[key][surface_name] = (int(node.label), coord, radius)
        elif surface_name == "outer" and radius > current[2]:
            grouped[key][surface_name] = (int(node.label), coord, radius)
        elif surface_name == "inner" and radius < current[2]:
            grouped[key][surface_name] = (int(node.label), coord, radius)

    axial_max = max(axial_values)
    cutting_zone_lower = axial_max - cutting_depth

    pair_rows: list[dict] = []
    nominal_thickness = float(case_cfg["nominal_thickness_mm"])
    for (theta_bucket, axial_bucket), grouped_nodes in sorted(grouped.items()):
        if "outer" not in grouped_nodes or "inner" not in grouped_nodes:
            continue

        outer_label, outer_coord, outer_radius = grouped_nodes["outer"]
        inner_label, inner_coord, inner_radius = grouped_nodes["inner"]
        thickness_initial = outer_radius - inner_radius

        # 初始配对厚度若明显偏离名义壁厚，通常意味着该桶跨越了边界或夹持区域，先剔除掉。
        if abs(thickness_initial - nominal_thickness) > max(0.6, nominal_thickness * 0.8):
            continue

        pair_rows.append(
            {
                "theta_bucket_deg": theta_bucket,
                "axial_bucket_mm": axial_bucket,
                "outer_node_label": outer_label,
                "inner_node_label": inner_label,
                "outer_x0_mm": outer_coord[0],
                "outer_y0_mm": outer_coord[1],
                "outer_z0_mm": outer_coord[2],
                "inner_x0_mm": inner_coord[0],
                "inner_y0_mm": inner_coord[1],
                "inner_z0_mm": inner_coord[2],
                "outer_radius0_mm": outer_radius,
                "inner_radius0_mm": inner_radius,
                "thickness0_mm": thickness_initial,
                "in_cutting_zone": axial_bucket >= cutting_zone_lower,
            }
        )
    return pair_rows


def displacement_lookup(frame, instance, nodal_flag):
    field_u = frame.fieldOutputs["U"].getSubset(region=instance, position=nodal_flag)
    lookup = {}
    for value in field_u.values:
        lookup[int(value.nodeLabel)] = tuple(float(item) for item in value.data)
    return lookup


def deformed_coord(coord: tuple[float, float, float], disp: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(coord[idx] + disp[idx] for idx in range(3))


def radial_distance(coord: tuple[float, float, float], axial_axis: str) -> float:
    radial_1, radial_2, _ = axis_bundle(coord, axial_axis)
    return math.hypot(radial_1, radial_2)


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        raise ValueError("空序列无法计算分位数。")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * ratio
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_frame(case_id: str, frame_raw: int, rows: list[dict], nominal_thickness: float) -> dict:
    cutting_rows = [row for row in rows if row["in_cutting_zone"]]
    if not cutting_rows:
        cutting_rows = rows

    valid_rows = [row for row in cutting_rows if row["is_physical"]]
    if not valid_rows:
        valid_rows = cutting_rows

    thickness_values = sorted(float(row["thickness_true_mm"]) for row in valid_rows)
    min_row = min(valid_rows, key=lambda row: float(row["thickness_true_mm"]))
    max_row = max(valid_rows, key=lambda row: float(row["thickness_true_mm"]))
    mean_value = sum(thickness_values) / len(thickness_values)
    median_value = percentile(thickness_values, 0.5)
    p05_value = percentile(thickness_values, 0.05)
    p95_value = percentile(thickness_values, 0.95)

    return {
        "case": case_id,
        "frame_raw": frame_raw,
        "pair_count": len(cutting_rows),
        "valid_pair_count": len(valid_rows),
        "thickness_min_mm": min_row["thickness_true_mm"],
        "thickness_p05_mm": p05_value,
        "thickness_mean_mm": mean_value,
        "thickness_median_mm": median_value,
        "thickness_p95_mm": p95_value,
        "thickness_max_mm": max_row["thickness_true_mm"],
        "min_theta_deg": min_row["theta_bucket_deg"],
        "min_axial_mm": min_row["axial_bucket_mm"],
        "max_theta_deg": max_row["theta_bucket_deg"],
        "max_axial_mm": max_row["axial_bucket_mm"],
        "nominal_thickness_mm": nominal_thickness,
        "max_thinning_mm": nominal_thickness - float(min_row["thickness_true_mm"]),
        "mean_bias_to_nominal_mm": mean_value - nominal_thickness,
    }


def resolve_target_frames(global_cfg: dict, frames_arg: str) -> list[int]:
    standard_frames = {int(frame) for frame in global_cfg["standard_frames"]}
    extra_frames: set[int] = set()
    if frames_arg.strip():
        for token in frames_arg.split(","):
            token = token.strip()
            if not token:
                continue
            extra_frames.add(int(token))
    target_frames = sorted(frame for frame in (standard_frames | extra_frames) if frame >= 0)
    if not target_frames:
        raise ValueError("未解析到任何目标帧。")
    return target_frames


def main(argv: list[str]) -> int:
    try:
        from abaqusConstants import NODAL
        from odbAccess import openOdb
    except Exception as exc:  # pragma: no cover - 仅在 Abaqus 环境执行
        sys.stderr.write(f"当前环境缺少 Abaqus odbAccess: {exc}\n")
        return 2

    args = parse_args(argv)
    config = load_config(args.config)
    case_id = normalize_case_id(args.case)
    case_cfg = config["cases"][case_id]
    global_cfg = config["global"]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    odb = openOdb(path=args.odb, readOnly=True)
    instance = select_instance(odb, case_id, case_cfg)
    step = select_step(odb, case_cfg)
    axial_axis = str(global_cfg["thickness"].get("axis_name", "Z")).upper()

    pair_rows = build_surface_pair_map(instance, case_cfg, global_cfg)
    pair_path = outdir / f"surface_pair_map_{case_id}.csv.gz"
    write_csv(pair_path, pair_rows)

    target_frames = resolve_target_frames(global_cfg, args.frames)
    frame_indices = list(range(len(step.frames)))
    nominal_thickness = float(case_cfg["nominal_thickness_mm"])

    fullfield_rows: list[dict] = []
    summary_rows: list[dict] = []

    for frame_raw in target_frames:
        frame_index = min(max(frame_raw, 0), frame_indices[-1])
        frame = step.frames[frame_index]
        disp_map = displacement_lookup(frame, instance, NODAL)
        frame_rows: list[dict] = []

        for pair_row in pair_rows:
            outer_label = int(pair_row["outer_node_label"])
            inner_label = int(pair_row["inner_node_label"])

            outer_coord = (
                float(pair_row["outer_x0_mm"]),
                float(pair_row["outer_y0_mm"]),
                float(pair_row["outer_z0_mm"]),
            )
            inner_coord = (
                float(pair_row["inner_x0_mm"]),
                float(pair_row["inner_y0_mm"]),
                float(pair_row["inner_z0_mm"]),
            )
            outer_disp = disp_map.get(outer_label, (0.0, 0.0, 0.0))
            inner_disp = disp_map.get(inner_label, (0.0, 0.0, 0.0))

            outer_coord_def = deformed_coord(outer_coord, outer_disp)
            inner_coord_def = deformed_coord(inner_coord, inner_disp)
            outer_radius = radial_distance(outer_coord_def, axial_axis)
            inner_radius = radial_distance(inner_coord_def, axial_axis)
            thickness_true = math.dist(outer_coord_def, inner_coord_def)
            lower_bound = max(0.2, nominal_thickness * 0.2)
            upper_bound = max(nominal_thickness * 1.8, nominal_thickness + 0.6)

            row = {
                "case": case_id,
                "frame_raw": frame_raw,
                "theta_bucket_deg": pair_row["theta_bucket_deg"],
                "axial_bucket_mm": pair_row["axial_bucket_mm"],
                "outer_node_label": outer_label,
                "inner_node_label": inner_label,
                "outer_radius_mm": outer_radius,
                "inner_radius_mm": inner_radius,
                "thickness_true_mm": thickness_true,
                "thickness_deviation_mm": thickness_true - nominal_thickness,
                "in_cutting_zone": pair_row["in_cutting_zone"],
                "is_physical": lower_bound <= thickness_true <= upper_bound,
            }
            frame_rows.append(row)
            fullfield_rows.append(row)

        summary_rows.append(summarize_frame(case_id, frame_raw, frame_rows, nominal_thickness))
        print(f"{case_id} 帧 {frame_raw} 提取完成，切削区配对点数 {summary_rows[-1]['pair_count']}")

    fullfield_path = outdir / f"true_thickness_fullfield_{case_id}.csv.gz"
    summary_path = outdir / f"true_thickness_frame_summary_{case_id}.csv"
    write_csv(fullfield_path, fullfield_rows)
    write_csv(summary_path, summary_rows)

    log_path = outdir / f"true_thickness_extract_log_{case_id}.json"
    log_path.write_text(
        json.dumps(
            {
                "case": case_id,
                "odb": args.odb,
                "instance": instance.name,
                "step": step.name,
                "pair_count_total": len(pair_rows),
                "pair_count_cutting_zone": sum(1 for row in pair_rows if row["in_cutting_zone"]),
                "frames": target_frames,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    odb.close()
    print(f"已输出: {fullfield_path}")
    print(f"已输出: {summary_path}")
    print(f"已输出: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
