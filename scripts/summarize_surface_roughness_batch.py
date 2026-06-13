#!/usr/bin/env python3
from __future__ import annotations

"""Summarize and validate batch surface-roughness outputs.

本脚本读取 `output/surface_roughness` 下的批处理 manifest 和每个加工面的
`roughness_summary.csv`，汇总第 4 章所需的 Gaussian 粗糙度指标，并检查
每个输出目录是否保留了完整结果文件。

This script reads the batch manifest and each surface folder's
`roughness_summary.csv` under `output/surface_roughness`. It summarizes the
Gaussian roughness metrics used in Chapter 4 and validates that each output
folder contains the retained files.
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output" / "surface_roughness"

# Files expected from `process_curved_surface_roughness.py` for every surface.
# 每个加工面完成处理后应保留的结果文件，用于批量完整性检查。
REQUIRED_OUTPUTS = [
    "roughness_summary.csv",
    "roughness_summary.xlsx",
    "method_comparison.csv",
    "processing_report.txt",
    "processing_trace.md",
    "flattened_point_cloud.csv",
    "gaussian_filtered_point_cloud.csv",
    "raw_height_heatmap.png",
    "removed_form_heatmap.png",
    "flattened_residual_heatmap.png",
    "gaussian_waviness_heatmap.png",
    "gaussian_roughness_heatmap.png",
    "raw_height_3d.png",
    "flattened_residual_3d.png",
    "flattened_roughness_3d.png",
]


def read_key_value_csv(path: Path) -> dict[str, str]:
    """Read a two-column `key,value` summary CSV.

    单个表面处理脚本将粗糙度指标和处理参数写为键值表，本函数把它转成字典。

    The single-surface script writes metrics and processing parameters as a
    key-value table. This helper converts it to a dictionary.
    """
    with path.open(encoding="utf-8") as f:
        return {row["key"]: row["value"] for row in csv.DictReader(f)}


def as_float(value: str) -> float:
    """Convert summary values to float for numerical checks.

    CSV 读入后均为字符串，半径、网格和滤波参数检查需要转成数值。

    CSV values are strings; radius, grid, and filter checks need numeric values.
    """
    return float(value)


def main() -> None:
    """Validate every task and write batch summary tables.

    主流程更新 manifest 状态，导出 `batch_roughness_results.csv` 和
    `batch_validation_report.csv/txt`。正文第 4 章的粗糙度汇总表来自该
    批量结果。

    The workflow updates manifest status and exports `batch_roughness_results`
    plus validation reports. The Chapter 4 roughness summary table is derived
    from the batch results.
    """
    manifest_csv = OUTPUT_ROOT / "surface_roughness_batch_manifest.csv"
    manifest_json = OUTPUT_ROOT / "surface_roughness_batch_manifest.json"

    with manifest_csv.open(encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    validation_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    problems: list[str] = []

    for row in manifest_rows:
        # Each manifest row points to one processed surface folder. The script
        # checks both file completeness and key numerical settings.
        # manifest 的每一行对应一个加工面输出目录。本脚本同时检查文件完整性
        # 和关键数值设置。
        folder = row["folder"]
        out_dir = ROOT / row["output_dir"]
        summary_path = out_dir / "roughness_summary.csv"
        missing = [name for name in REQUIRED_OUTPUTS if not (out_dir / name).exists()]
        if not summary_path.exists():
            problems.append(f"{folder}: missing roughness_summary.csv")
            summary: dict[str, str] = {}
        else:
            summary = read_key_value_csv(summary_path)

        radius_ok = False
        sample_ok = False
        gaussian_ok = False
        grid_ok = False
        if summary:
            # The Chapter 4 retained result uses Gaussian roughness with a
            # 250 um cutoff on a 250 by 250 grid. These checks catch accidental
            # reruns with different parameters.
            # 第 4 章保留的结果口径为 250 um 截止波长的 Gaussian 粗糙度面，
            # 网格为 250 x 250。下列检查用于发现参数误改。
            radius_ok = abs(as_float(summary["surface_radius_mm"]) - as_float(row["surface_radius_mm"])) < 1e-9
            sample_ok = summary.get("sample", row["sample"]) == row["sample"]
            gaussian_ok = (
                summary.get("main_result_surface") == "gaussian_roughness"
                and summary.get("gaussian_filter_enabled") == "True"
                and abs(as_float(summary["gaussian_cutoff_um"]) - 250.0) < 1e-9
            )
            grid_ok = (
                int(float(summary.get("grid_x_count", 0))) == 250
                and int(float(summary.get("grid_y_count", 0))) == 250
                and int(float(summary.get("point_count", 0))) == 62500
            )

        status = "done" if summary and not missing and radius_ok and sample_ok and gaussian_ok and grid_ok else "check"
        if status != "done":
            problems.append(
                f"{folder}: status={status}, missing={missing}, radius_ok={radius_ok}, "
                f"sample_ok={sample_ok}, gaussian_ok={gaussian_ok}, grid_ok={grid_ok}"
            )
        row["status"] = status

        validation_rows.append(
            {
                "folder": folder,
                "status": status,
                "missing_outputs": ";".join(missing),
                "radius_ok": radius_ok,
                "sample_ok": sample_ok,
                "gaussian_ok": gaussian_ok,
                "grid_ok": grid_ok,
            }
        )

        if summary:
            result_rows.append(
                {
                    "folder": folder,
                    "input_csv": row["input_csv"],
                    "sample": row["sample"],
                    "surface_radius_mm": f"{as_float(summary['surface_radius_mm']):.2f}",
                    "point_count": summary["point_count"],
                    "grid_x_count": summary["grid_x_count"],
                    "grid_y_count": summary["grid_y_count"],
                    "gaussian_cutoff_um": summary["gaussian_cutoff_um"],
                    "Sa_um": summary["Sa_um"],
                    "Sq_um": summary["Sq_um"],
                    "Sp_um": summary["Sp_um"],
                    "Sv_um": summary["Sv_um"],
                    "Sz_um": summary["Sz_um"],
                    "Ssk": summary["Ssk"],
                    "Sku": summary["Sku"],
                    "Sa_nm": summary["Sa_nm"],
                    "Sq_nm": summary["Sq_nm"],
                    "Sz_nm": summary["Sz_nm"],
                    "form_removed_unfiltered_Sa_um": summary["form_removed_unfiltered_Sa_um"],
                    "form_removed_unfiltered_Sq_um": summary["form_removed_unfiltered_Sq_um"],
                    "form_removed_unfiltered_Sz_um": summary["form_removed_unfiltered_Sz_um"],
                    "curvature_angle_deg": summary["curvature_angle_deg"],
                    "curvature_sign": summary["curvature_sign"],
                    "auto_quadratic_radius_estimate_um": summary["auto_quadratic_radius_estimate_um"],
                    "output_dir": row["output_dir"],
                }
            )

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    by_folder = {row["folder"]: row["status"] for row in manifest_rows}
    for task in manifest.get("tasks", []):
        task["status"] = by_folder.get(task.get("folder"), task.get("status"))
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results_path = OUTPUT_ROOT / "batch_roughness_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    validation_path = OUTPUT_ROOT / "batch_validation_report.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    report_path = OUTPUT_ROOT / "batch_validation_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Surface roughness batch validation report\n")
        f.write("=========================================\n\n")
        f.write(f"total_tasks: {len(validation_rows)}\n")
        f.write(f"done_tasks: {sum(row['status'] == 'done' for row in validation_rows)}\n")
        f.write(f"check_tasks: {sum(row['status'] != 'done' for row in validation_rows)}\n\n")
        f.write("Required checks: all retained outputs exist, radius/sample match manifest, ")
        f.write("main result is Gaussian roughness with 250 um cutoff, grid is 250x250 and 62500 points.\n\n")
        if problems:
            f.write("Problems\n")
            f.write("--------\n")
            for problem in problems:
                f.write(f"- {problem}\n")
        else:
            f.write("Problems\n")
            f.write("--------\n")
            f.write("none\n")

    print(f"Wrote {results_path}")
    print(f"Wrote {validation_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
