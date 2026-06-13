#!/usr/bin/env python3
from __future__ import annotations

"""
在 Abaqus/CAE noGUI 环境下运行，用于从 .odb 重建单曲率薄壁件的内外表面配对与真壁厚结果。

Run this script inside Abaqus/CAE noGUI. It reconstructs inner/outer surface
node pairs from an ODB file and exports true wall-thickness histories for
single-curvature thin-wall cases.

示例:
abaqus cae noGUI=scripts/abaqus/extract_true_thickness_odb.py -- \
  --odb data/abaqus/odb/S1.odb \
  --case S1 \
  --points-csv output/spreadsheet/abaqus/thickness_point_seeds.csv \
  --config scripts/abaqus/case_config.yml \
  --out output/spreadsheet/abaqus
"""

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments passed after Abaqus' `--` separator.

    Abaqus 会占用前半段命令行参数，脚本自己的参数需要放在 `--` 后面。

    Abaqus consumes the first part of the command line, so this script reads
    only the arguments after the `--` separator.
    """
    parser = argparse.ArgumentParser(description="从 Abaqus ODB 提取真壁厚结果。")
    parser.add_argument("--odb", required=True, help="ODB 文件路径。")
    parser.add_argument("--case", required=True, help="案例编号，例如 S1。")
    parser.add_argument("--points-csv", required=True, help="缺陷点种子表 CSV。")
    parser.add_argument("--config", required=True, help="案例配置文件。")
    parser.add_argument("--out", required=True, help="结果输出目录。")
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    """Load YAML configuration with JSON fallback.

    Abaqus 自带 Python 环境有时缺少 PyYAML。此处优先读 YAML，失败后读取
    同名 JSON 配置，保证 noGUI 环境仍可运行。

    Abaqus' bundled Python may not include PyYAML. The function tries YAML first
    and falls back to a same-name JSON file.
    """
    config_path = Path(path)
    try:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        json_path = config_path.with_suffix(".json")
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        raise


def theta_deg(x_mm: float, y_mm: float) -> float:
    """Convert an `(x, y)` coordinate to a 0-360 degree polar angle.

    单曲率截面匹配以内外表面节点的极角为主要索引。

    Polar angle is the main index for matching inner and outer surface nodes on
    the same curved section.
    """
    angle = math.degrees(math.atan2(y_mm, x_mm))
    return angle if angle >= 0 else angle + 360.0


def round_bucket(value: float, step: float) -> float:
    """Round a value to the nearest bucket.

    极角和轴向坐标都需要离散到容差桶，降低网格浮点误差对配对的影响。

    Angle and axial coordinates are bucketed to reduce floating-point noise when
    pairing mesh nodes.
    """
    return round(value / step) * step


def classify_surface(case_cfg: dict, node_coord: tuple[float, float, float]) -> str | None:
    """Classify a node as inner or outer surface using its radius.

    以名义内外半径中点为分界。半径较大的节点归为外表面，半径较小的
    节点归为内表面。

    The midpoint between nominal inner and outer radius is used as the boundary.
    Larger-radius nodes are treated as outer-surface nodes.
    """
    x_mm, y_mm, _ = node_coord
    radius = math.hypot(x_mm, y_mm)
    outer_nominal = float(case_cfg["radius_mm"]) + float(case_cfg["nominal_thickness_mm"]) / 2.0
    inner_nominal = float(case_cfg["reference_inner_radius_mm"])
    midpoint = (outer_nominal + inner_nominal) / 2.0
    if radius >= midpoint:
        return "outer"
    return "inner"


def select_instance(odb, case_cfg: dict):
    """Select the Abaqus instance to read.

    若配置提供 `instance_name`，按配置读取；否则使用 ODB 中排序后的第一个
    instance，适配单零件模型。

    If `instance_name` is configured, it is used directly. Otherwise the first
    sorted instance is selected for single-part ODB files.
    """
    assembly = odb.rootAssembly
    instance_name = case_cfg.get("instance_name")
    if instance_name:
        return assembly.instances[instance_name]
    instance_names = sorted(assembly.instances.keys())
    if not instance_names:
        raise RuntimeError("ODB 中未找到任何 instance。")
    return assembly.instances[instance_names[0]]


def select_step(odb, case_cfg: dict):
    """Select the analysis step containing cutting frames.

    默认读取 `Step-1`。若 ODB 中名称不同，则回退到排序后的第一个 step。

    The default is `Step-1`. If that step name is absent, the first sorted step
    is used.
    """
    step_name = case_cfg.get("step_name") or "Step-1"
    if step_name in odb.steps:
        return odb.steps[step_name]
    first_step_name = sorted(odb.steps.keys())[0]
    return odb.steps[first_step_name]


def load_seed_rows(points_csv: str, case_id: str) -> list[dict]:
    """Load seed points for one case from the seed CSV.

    种子点表来自 `build_thickness_seed_tables.py`，其中包含缺陷点类型、
    帧号和旧近似厚度。

    The seed CSV is produced by `build_thickness_seed_tables.py` and contains
    defect type, frame, and legacy approximate thickness information.
    """
    with open(points_csv, "r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row["case"].upper() == case_id.upper()]


def build_surface_pair_map(instance, case_cfg: dict, global_cfg: dict) -> list[dict]:
    """Build inner/outer surface node pairs for each angle and axial bucket.

    每个桶中保留半径最大的外表面节点和半径最小的内表面节点。该映射是
    真壁厚计算的几何基础。

    For each angle/axial bucket, the largest-radius outer node and smallest-
    radius inner node are retained. This map is the geometric basis for true
    thickness calculation.
    """
    theta_step = float(global_cfg["thickness"]["theta_round_deg"])
    z_step = float(global_cfg["thickness"]["z_round_mm"])
    grouped: dict[tuple[float, float], dict[str, tuple[int, tuple[float, float, float], float]]] = defaultdict(dict)
    for node in instance.nodes:
        # Node coordinates are undeformed positions. Frame displacements are
        # added later so the same pair map can be reused across target frames.
        # 这里读取未变形坐标。各帧位移稍后叠加，因此同一配对映射可复用于
        # 多个目标帧。
        coord = tuple(float(value) for value in node.coordinates)
        surface_name = classify_surface(case_cfg, coord)
        if surface_name is None:
            continue
        x_mm, y_mm, z_mm = coord
        key = (round_bucket(theta_deg(x_mm, y_mm), theta_step), round_bucket(z_mm, z_step))
        radius = math.hypot(x_mm, y_mm)
        current = grouped[key].get(surface_name)
        if current is None:
            grouped[key][surface_name] = (int(node.label), coord, radius)
        elif surface_name == "outer" and radius > current[2]:
            grouped[key][surface_name] = (int(node.label), coord, radius)
        elif surface_name == "inner" and radius < current[2]:
            grouped[key][surface_name] = (int(node.label), coord, radius)

    pair_rows = []
    for (theta_bucket, z_bucket), grouped_nodes in sorted(grouped.items()):
        if "outer" not in grouped_nodes or "inner" not in grouped_nodes:
            continue
        outer_label, outer_coord, outer_radius = grouped_nodes["outer"]
        inner_label, inner_coord, inner_radius = grouped_nodes["inner"]
        pair_rows.append(
            {
                "theta_bucket_deg": theta_bucket,
                "z_bucket_mm": z_bucket,
                "outer_node_label": outer_label,
                "inner_node_label": inner_label,
                "outer_x0_mm": outer_coord[0],
                "outer_y0_mm": outer_coord[1],
                "outer_z0_mm": outer_coord[2],
                "inner_x0_mm": inner_coord[0],
                "inner_y0_mm": inner_coord[1],
                "inner_z0_mm": inner_coord[2],
                "outer_r0_mm": outer_radius,
                "inner_r0_mm": inner_radius,
            }
        )
    return pair_rows


def displacement_lookup(frame, instance):
    """Return a node-label to displacement-vector lookup for one frame.

    Abaqus 的 U 场以节点标签为键，查表后可把节点初始坐标更新到当前帧。

    Abaqus' U field is keyed by node labels. The lookup updates initial node
    coordinates to the current frame.
    """
    field_u = frame.fieldOutputs["U"].getSubset(region=instance, position=NODAL)
    lookup = {}
    for value in field_u.values:
        lookup[int(value.nodeLabel)] = tuple(float(item) for item in value.data)
    return lookup


def select_profile_row(profile_rows: list[dict], point_type: str) -> dict:
    """Select a representative true-thickness row for one seed point.

    凹陷点取最小真壁厚，凸起点取最大真壁厚，控制点取中位代表值。这样
    缺陷类型和局部厚度风险的方向保持一致。

    Dent points use minimum true thickness, bulge points use maximum true
    thickness, and control points use the middle representative value.
    """
    if not profile_rows:
        raise RuntimeError("候选剖面为空，无法生成点结果。")
    if point_type == "dent":
        return min(profile_rows, key=lambda row: row["thickness_true_mm"])
    if point_type == "bulge":
        return max(profile_rows, key=lambda row: row["thickness_true_mm"])
    return sorted(profile_rows, key=lambda row: row["thickness_true_mm"])[len(profile_rows) // 2]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write normal CSV or gzip-compressed CSV according to the file suffix.

    大剖面表行数较多，使用 `.gz` 后缀时自动压缩；点级结果保持普通 CSV。

    Large profile tables are compressed when the output suffix is `.gz`; compact
    point-level outputs remain plain CSV files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    """Run the ODB extraction workflow for one case.

    主流程为：打开 ODB、选择 instance 和 step、建立内外表面配对、读取
    目标帧位移、叠加变形坐标、计算真壁厚与旧近似量偏差、写出剖面表和
    点级结果。

    The workflow opens the ODB, selects instance and step, builds inner/outer
    node pairs, reads target-frame displacements, computes deformed radii, and
    exports true thickness plus legacy-bias tables.
    """
    try:
        from abaqusConstants import NODAL
        from odbAccess import openOdb
    except Exception as exc:  # pragma: no cover - 仅在 Abaqus 环境执行
        sys.stderr.write(f"当前环境缺少 Abaqus odbAccess: {exc}\n")
        return 2

    global NODAL  # noqa: PLW0603
    NODAL = NODAL

    args = parse_args(argv)
    config = load_config(args.config)
    case_id = args.case.upper()
    case_cfg = config["cases"][case_id]
    global_cfg = config["global"]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    log_lines.append(f"CASE={case_id}")
    log_lines.append(f"ODB={args.odb}")

    odb = openOdb(path=args.odb, readOnly=True)
    instance = select_instance(odb, case_cfg)
    step = select_step(odb, case_cfg)
    log_lines.append(f"INSTANCE={instance.name}")
    log_lines.append(f"STEP={step.name}")

    pair_rows = build_surface_pair_map(instance, case_cfg, global_cfg)
    pair_path = outdir / f"surface_pair_map_{case_id}.csv.gz"
    write_csv(pair_path, pair_rows)

    pair_lookup = {(row["theta_bucket_deg"], row["z_bucket_mm"]): row for row in pair_rows}
    pair_by_theta: dict[float, list[dict]] = defaultdict(list)
    for row in pair_rows:
        pair_by_theta[row["theta_bucket_deg"]].append(row)

    standard_frames = set(int(frame) for frame in global_cfg["standard_frames"])
    seed_rows = load_seed_rows(args.points_csv, case_id)
    seed_frames = {int(row["frame_raw"]) for row in seed_rows}
    target_frames = sorted(standard_frames | seed_frames)
    available_frame_indices = list(range(len(step.frames)))

    profile_rows: list[dict] = []
    point_rows: list[dict] = []

    for seed_row in seed_rows:
        # Match every seed point to the closest available angle bucket so the
        # selected profile follows the local curved section.
        # 将每个种子点匹配到最接近的极角桶，使剖面结果对应局部曲率截面。
        seed_theta = float(seed_row["theta_seed_deg"])
        candidate_theta = min(pair_by_theta.keys(), key=lambda theta_value: abs(theta_value - seed_theta))
        seed_row["matched_theta_bucket_deg"] = candidate_theta

    for target_frame in target_frames:
        # Abaqus frame lists are zero-based in Python, while exported frame
        # labels in the thesis workflow are one-based.
        # Abaqus Python 中帧列表为 0 基索引；论文数据链中的帧号按 1 基编号。
        frame_index = min(target_frame - 1, available_frame_indices[-1])
        frame = step.frames[frame_index]
        disp_map = displacement_lookup(frame, instance)

        thickness_profiles_by_seed: dict[str, list[dict]] = defaultdict(list)
        for seed_row in seed_rows:
            theta_bucket = float(seed_row["matched_theta_bucket_deg"])
            for pair_row in pair_by_theta[theta_bucket]:
                # True thickness is computed after adding current-frame
                # displacements to the undeformed inner and outer nodes.
                # 真壁厚在叠加当前帧位移后计算，避免沿用旧工作簿的单点半径近似。
                outer_u = disp_map.get(int(pair_row["outer_node_label"]), (0.0, 0.0, 0.0))
                inner_u = disp_map.get(int(pair_row["inner_node_label"]), (0.0, 0.0, 0.0))
                outer_x = pair_row["outer_x0_mm"] + outer_u[0]
                outer_y = pair_row["outer_y0_mm"] + outer_u[1]
                inner_x = pair_row["inner_x0_mm"] + inner_u[0]
                inner_y = pair_row["inner_y0_mm"] + inner_u[1]
                outer_r = math.hypot(outer_x, outer_y)
                inner_r = math.hypot(inner_x, inner_y)
                true_thickness = outer_r - inner_r
                legacy_thickness = outer_r - float(case_cfg["reference_inner_radius_mm"])
                profile_row = {
                    "case": case_id,
                    "point_id": seed_row["point_id"],
                    "point_type": seed_row["point_type"],
                    "frame_raw": target_frame,
                    "theta_deg": theta_bucket,
                    "z_mm": pair_row["z_bucket_mm"],
                    "outer_r_mm": outer_r,
                    "inner_r_mm": inner_r,
                    "thickness_true_mm": true_thickness,
                    "thickness_legacy_mm": legacy_thickness,
                    "bias_mm": legacy_thickness - true_thickness,
                }
                profile_rows.append(profile_row)
                thickness_profiles_by_seed[seed_row["point_id"]].append(profile_row)

        for seed_row in seed_rows:
            selection = select_profile_row(thickness_profiles_by_seed[seed_row["point_id"]], seed_row["point_type"])
            point_rows.append(selection)

    profile_path = outdir / f"true_thickness_profiles_{case_id}.csv.gz"
    point_path = outdir / f"true_thickness_points_{case_id}.csv"
    write_csv(profile_path, profile_rows)
    write_csv(point_path, point_rows)

    log_path = outdir.parents[1] / "abaqus_logs" / f"odb_extract_log_{case_id}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines.append(f"PAIR_COUNT={len(pair_rows)}")
    log_lines.append(f"PROFILE_ROWS={len(profile_rows)}")
    log_lines.append(f"POINT_ROWS={len(point_rows)}")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    odb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
