#!/usr/bin/env python3
from __future__ import annotations

"""
批量升级旧版本 Abaqus ODB 文件。

当 ODB 由较旧 Abaqus 版本生成时，后续真壁厚、STATUS 和应力提取前
需要先执行 `abaqus upgrade`。脚本只读取示意输入目录中的 ODB，并将
升级副本写入输出目录，不改写原始文件。

When ODB files are produced by an older Abaqus release, run this script before
true-thickness, STATUS, force, or stress extraction. It calls `abaqus upgrade`
on example input paths and writes upgraded copies to the output directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "abaqus" / "odb"
DEFAULT_OUTPUT = ROOT / "output" / "odb_upgraded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量升级旧版本 ODB 文件。")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT), help="原始 ODB 所在目录。")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT), help="升级后 ODB 输出目录。")
    parser.add_argument("--abaqus-command", default="abaqus", help="Abaqus command or abaqus.bat path.")
    parser.add_argument("--force", action="store_true", help="若已存在升级结果，是否强制重新升级。")
    return parser.parse_args()


def run_upgrade(odb_path: Path, outdir: Path, force: bool, abaqus_command: str) -> Path:
    job_stem = f"{odb_path.stem}_upg"
    output_stem = outdir / job_stem
    output_odb = output_stem.with_suffix(".odb")

    # 如果升级结果已经存在且用户没有要求覆盖，则直接复用，避免浪费时间。
    if output_odb.exists() and not force:
        print(f"跳过已存在文件: {output_odb}")
        return output_odb

    command = [
        abaqus_command,
        "upgrade",
        "-job",
        str(output_stem.relative_to(ROOT)),
        "-odb",
        str(odb_path.relative_to(ROOT)),
    ]
    print(f"开始升级: {odb_path.name}")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"升级完成: {output_odb.name}")
    return output_odb


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    odb_files = sorted(input_dir.glob("*.odb"))
    if not odb_files:
        raise FileNotFoundError(f"未在 {input_dir} 找到任何 ODB 文件。")

    upgraded = []
    for odb_path in odb_files:
        upgraded.append(run_upgrade(odb_path, outdir, force=args.force, abaqus_command=args.abaqus_command))

    print("全部升级完成：")
    for path in upgraded:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
