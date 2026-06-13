from __future__ import annotations

"""Shared helpers for the Abaqus data-processing pipeline.

本文件集中放置第 3 章 Abaqus 数据处理脚本共用的路径、配置读取、
工况编号规范化、测线编号规范化、帧号解析和几何量计算函数。
把这些逻辑集中管理，可以避免位移对齐、真壁厚重算和图件导出脚本
对同一数据口径作出不同解释。

This module contains shared path, configuration, case-id, line-id, frame parsing,
and geometry utilities used by the Chapter 3 Abaqus-processing scripts. Keeping
these rules in one file makes the downstream displacement, thickness, and figure
exports use the same interpretation of the simulation data.
"""

import math
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

# Repository root of the open-source script bundle.
# 开源脚本包根目录，所有默认输入输出路径都以它为基准。
ROOT = Path(__file__).resolve().parents[2]


def load_case_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML case configuration.

    配置文件保存 S1 至 S6 的几何参数、测线位置、标准帧和对齐参数。

    The configuration stores S1-S6 geometry, measurement-line locations,
    standard frames, and alignment parameters.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_case_id(case_token: str) -> str:
    """Return an uppercase case id, for example `s1` -> `S1`.

    工况编号在目录名、工作表名和 CSV 中可能大小写不一致，统一后再参与合并。

    Case ids may come from folder names, sheet names, or CSV fields with mixed
    case. Normalization prevents failed joins later in the pipeline.
    """
    return case_token.strip().upper()


def normalize_line_id(line_token: str | int) -> str:
    """Format a line id as two digits, for example `1` -> `01`.

    Abaqus 位移导出目录可能写作 `1` 或 `01`，正文表格采用两位编号。

    Abaqus export folders may use `1` or `01`; the paper tables use two-digit
    line ids.
    """
    text = str(line_token).strip()
    return f"{int(text):02d}"


def parse_frame_from_name(name: str) -> int:
    """Extract the raw frame number from an exported Excel filename.

    文件名末尾的 `-帧号.xlsx` 是切削过程原始帧索引。

    The trailing `-frame.xlsx` token is treated as the raw cutting-process
    frame index.
    """
    match = re.search(r"-(\d+)\.xlsx$", name)
    if not match:
        raise ValueError(f"无法从文件名解析帧号: {name}")
    return int(match.group(1))


def iter_displacement_files(input_dir: str | Path) -> Iterable[tuple[str, str, int, Path]]:
    """Yield `(case, line_id, frame_raw, path)` for displacement workbooks.

    输入目录按工况和测线组织，本函数负责把目录结构转成后续脚本可合并的
    长表索引。

    The input directory is organized by case and measurement line. This helper
    converts that folder layout into a long-table index used by later scripts.
    """
    base = Path(input_dir)
    for case_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("s")):
        case_id = case_dir.name.upper()
        for xlsx_path in sorted(case_dir.rglob("*.xlsx")):
            line_id = normalize_line_id(xlsx_path.parent.name.split("-")[-1])
            frame_raw = parse_frame_from_name(xlsx_path.name)
            yield case_id, line_id, frame_raw, xlsx_path


def standard_frames(config: dict[str, Any]) -> list[int]:
    """Return the frame list used for common reporting and ODB extraction.

    标准帧用于跨工况比较，也用于真壁厚提取时限定固定采样点。

    Standard frames support cross-case comparison and fixed-frame true-thickness
    extraction.
    """
    return [int(frame) for frame in config["global"]["standard_frames"]]


def line_positions(config: dict[str, Any]) -> dict[str, float]:
    """Read the physical position of each measurement line in millimetres.

    返回值把两位测线编号映射到距自由端位置。

    The returned mapping connects each two-digit line id to its distance from
    the free end.
    """
    return {
        normalize_line_id(line_id): float(position)
        for line_id, position in config["global"]["line_positions_mm"].items()
    }


def case_settings(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Return one case configuration after normalizing the case id.

    各脚本用同一入口读取 S1 至 S6 的半径、名义壁厚和参考内半径。

    All scripts use this entry point to access radius, nominal thickness, and
    reference inner radius for S1-S6.
    """
    return config["cases"][normalize_case_id(case_id)]


def nominal_outer_radius(case_config: dict[str, Any]) -> float:
    """Compute the nominal outer radius of a single-curvature wall.

    外半径等于中面曲率半径加半个名义壁厚。

    The outer radius equals the mid-surface radius plus half the nominal wall
    thickness.
    """
    return float(case_config["radius_mm"]) + float(case_config["nominal_thickness_mm"]) / 2.0


def nominal_inner_radius(case_config: dict[str, Any]) -> float:
    """Return the reference inner radius used for legacy thickness formulas.

    若配置显式给出参考内半径，则以配置为准；否则用中面半径减半壁厚估算。

    If a reference inner radius is provided, it is used directly. Otherwise the
    radius is estimated from the mid-surface radius and nominal thickness.
    """
    if "reference_inner_radius_mm" in case_config:
        return float(case_config["reference_inner_radius_mm"])
    return float(case_config["radius_mm"]) - float(case_config["nominal_thickness_mm"]) / 2.0


def theta_deg(x_mm: float, y_mm: float) -> float:
    """Convert an `(x, y)` point to a 0-360 degree polar angle.

    缺陷点和内外表面节点配对时使用该角度匹配同一曲率截面。

    This angle is used to match defect seeds and inner/outer surface nodes on
    the same curved section.
    """
    angle = math.degrees(math.atan2(y_mm, x_mm))
    return angle if angle >= 0 else angle + 360.0


def frame_sort_key(frame_token: Any) -> tuple[int, str]:
    """Sort mixed frame labels by leading integer and original text.

    某些表格中的帧号可能带有附加标记，排序时先看开头数字。

    Some frame labels carry suffixes. Sorting uses the leading number and then
    the original token as a stable tie-breaker.
    """
    if isinstance(frame_token, (int, float)):
        return int(frame_token), str(frame_token)
    text = str(frame_token)
    match = re.match(r"(\d+)", text)
    if not match:
        return 0, text
    return int(match.group(1)), text
