#!/usr/bin/env python3
from __future__ import annotations

"""
批量调用 Abaqus Python，提取六个算例的真壁厚结果。

脚本将 ODB 解析工作交给 `extract_true_thickness_fullfield_odb.py`，
并按配置文件中的工况逐个调用。默认读取升级后的示意 ODB 目录，将全场
真壁厚明细与帧级统计写入统一表格目录。

The script does not parse ODB data directly. It calls
`extract_true_thickness_fullfield_odb.py` case by case, using the anonymized
configuration to find upgraded ODB files and export full-field thickness tables.
"""

import argparse
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPGRADED_DIR = ROOT / "output" / "odb_upgraded"
DEFAULT_OUTDIR = ROOT / "output" / "spreadsheet" / "abaqus"
DEFAULT_CONFIG = ROOT / "scripts" / "abaqus" / "case_config.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量提取六个算例的真壁厚结果。")
    parser.add_argument("--odb-dir", default=str(DEFAULT_UPGRADED_DIR), help="升级后 ODB 所在目录。")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="结果输出目录。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="案例配置文件路径。")
    parser.add_argument("--abaqus-command", default="abaqus", help="Abaqus command or abaqus.bat path.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_case(case_id: str, odb_path: Path, config_path: Path, outdir: Path, abaqus_command: str) -> None:
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
    ]
    print(f"开始提取 {case_id}: {odb_path.name}")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"完成提取 {case_id}")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    odb_dir = Path(args.odb_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    for case_id, case_cfg in config["cases"].items():
        original_name = Path(case_cfg["odb_filename"]).stem
        odb_path = odb_dir / f"{original_name}_upg.odb"
        if not odb_path.exists():
            raise FileNotFoundError(f"未找到升级后的 ODB: {odb_path}")
        run_case(case_id, odb_path, config_path, outdir, args.abaqus_command)

    print("全部算例真壁厚提取完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
