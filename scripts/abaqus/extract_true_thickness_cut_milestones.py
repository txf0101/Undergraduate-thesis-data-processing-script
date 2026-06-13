#!/usr/bin/env python3
from __future__ import annotations

"""
按 STATUS 实际切削里程碑补提真壁厚数据。

脚本读取 `cut_progress_milestones.csv` 中的 cut_start、q1、q2、q3 与
cut_done 帧，将其与配置中的标准帧合并，再逐算例调用 Abaqus Python。
生成的关键帧真壁厚结果用于切削完成帧场图和实际进程坐标下的厚度曲线。

The script merges STATUS-derived milestone frames with the configured standard
frames and reruns full-field thickness extraction. These extra frames support
cut-completion field maps and thickness curves on the actual cutting-progress
axis.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPGRADED_DIR = ROOT / "output" / "odb_upgraded"
DEFAULT_OUTDIR = ROOT / "output" / "spreadsheet" / "abaqus"
DEFAULT_CONFIG = ROOT / "scripts" / "abaqus" / "case_config.yml"
DEFAULT_MILESTONES = ROOT / "output" / "spreadsheet" / "abaqus" / "cut_progress_milestones.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按实际切削里程碑补提真壁厚数据。")
    parser.add_argument("--odb-dir", default=str(DEFAULT_UPGRADED_DIR), help="升级后 ODB 所在目录。")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="结果输出目录。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="案例配置文件路径。")
    parser.add_argument("--milestones", default=str(DEFAULT_MILESTONES), help="实际切削里程碑 CSV 路径。")
    parser.add_argument("--abaqus-command", default="abaqus", help="Abaqus command or abaqus.bat path.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        with path.with_suffix(".json").open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_milestones(path: Path) -> dict[str, dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    milestones: dict[str, dict[str, int]] = {}
    for row in rows:
        case_id = str(row["case"]).strip().upper()
        milestones[case_id] = {
            "cut_start_frame_raw": int(row["cut_start_frame_raw"]),
            "cut_q1_frame_raw": int(row["cut_q1_frame_raw"]),
            "cut_q2_frame_raw": int(row["cut_q2_frame_raw"]),
            "cut_q3_frame_raw": int(row["cut_q3_frame_raw"]),
            "cut_done_frame_raw": int(row["cut_done_frame_raw"]),
        }
    return milestones


def resolve_frames(case_id: str, config: dict, milestones: dict[str, dict[str, int]]) -> list[int]:
    standard_frames = {int(frame) for frame in config["global"]["standard_frames"]}
    milestone_frames = set(milestones[case_id].values())
    return sorted(frame for frame in (standard_frames | milestone_frames) if frame >= 0)


def run_case(
    case_id: str,
    odb_path: Path,
    config_path: Path,
    outdir: Path,
    frames: list[int],
    abaqus_command: str,
) -> None:
    command = [
        abaqus_command,
        "python",
        str((ROOT / "scripts" / "abaqus" / "extract_true_thickness_fullfield_odb.py").relative_to(ROOT)),
        "--odb",
        str(odb_path.relative_to(ROOT)),
        "--case",
        case_id,
        "--config",
        str(config_path.relative_to(ROOT)),
        "--out",
        str(outdir.relative_to(ROOT)),
        "--frames",
        ",".join(str(frame) for frame in frames),
    ]
    print(f"开始补提 {case_id}: {odb_path.name} -> frames={frames}")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"完成补提 {case_id}")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    odb_dir = Path(args.odb_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    milestones = load_milestones(Path(args.milestones))

    for case_id, case_cfg in config["cases"].items():
        original_name = Path(case_cfg["odb_filename"]).stem
        odb_path = odb_dir / f"{original_name}_upg.odb"
        if not odb_path.exists():
            raise FileNotFoundError(f"未找到升级后的 ODB: {odb_path}")
        if case_id not in milestones:
            raise KeyError(f"里程碑文件缺少 {case_id}。")
        run_case(
            case_id,
            odb_path,
            config_path,
            outdir,
            resolve_frames(case_id, config, milestones),
            args.abaqus_command,
        )

    print("全部算例的关键帧真壁厚补提完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
