#!/usr/bin/env python3
from __future__ import annotations

"""Process curved-surface profilometer point clouds for areal roughness.

本脚本用于论文第 4 章三维轮廓仪数据处理。输入是非接触光学三维轮廓仪
导出的点云 CSV，字段为 `x,y,z[,intensity]`。脚本先根据试样几何去除
单曲率圆柱形貌和扫描倾斜平面，再在去形后的残余面上计算三维粗糙度
指标，并可按 ISO 16610 风格的 Gaussian 权重分离波纹面和粗糙度面。

This script processes optical 3D profilometer point-cloud CSV files used in
Chapter 4. Input columns are `x,y,z[,intensity]`. The script removes the known
single-curvature cylindrical form and scanner-tilt plane, calculates areal
roughness metrics on the residual surface, and optionally separates waviness
and roughness using an ISO 16610-style Gaussian weighting.
"""

import argparse
import csv
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ISO_GAUSSIAN_ALPHA = math.sqrt(math.log(2.0) / math.pi)

# Nominal sample geometry used to infer the machined surface radius.
# 样件名义几何参数，用于由样件编号推算加工表面半径。
SAMPLE_GEOMETRY_MM = {
    "S1": {"mid_radius": 30.0, "thickness": 1.0},
    "S2": {"mid_radius": 30.0, "thickness": 1.5},
    "S3": {"mid_radius": 40.0, "thickness": 1.0},
    "S4": {"mid_radius": 40.0, "thickness": 1.5},
    "S5": {"mid_radius": 50.0, "thickness": 1.0},
    "S6": {"mid_radius": 50.0, "thickness": 1.5},
}


def parse_args() -> argparse.Namespace:
    """Parse point-cloud input and processing parameters.

    参数分为四类：输入输出路径、试样几何、曲率去形设置和 Gaussian 滤波
    设置。开源使用者通常只需要替换输入 CSV、样件编号和输出目录。

    Parameters cover input/output paths, sample geometry, curvature form-removal
    settings, and Gaussian-filter settings. Most users only need to change the
    input CSV, sample id, and output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove the single-curvature cylindrical form from profilometer "
            "point-cloud CSV data and export roughness metrics."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV with x,y,z[,intensity] columns.")
    parser.add_argument(
        "--sample",
        choices=sorted(SAMPLE_GEOMETRY_MM),
        help="Sample geometry, such as S1. If omitted, the script tries to infer it from the path.",
    )
    parser.add_argument(
        "--surface-radius-mm",
        type=float,
        help="Actual machined surface radius. Overrides sample/planned-ae/actual-ae calculation.",
    )
    parser.add_argument(
        "--planned-ae-mm",
        type=float,
        default=0.2,
        help="Originally planned radial stock/removal allowance used in the blank radius calculation.",
    )
    parser.add_argument(
        "--actual-ae-mm",
        type=float,
        default=0.5,
        help="Actual radial engagement used in cutting.",
    )
    parser.add_argument(
        "--curvature-angle-deg",
        type=float,
        help="Curvature direction angle in the scan x-y plane. Default: auto from quadratic form fit.",
    )
    parser.add_argument(
        "--curvature-sign",
        choices=("auto", "positive", "negative"),
        default="auto",
        help="Sign of cylindrical sag in z. Default: auto from quadratic form fit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "surface_roughness",
        help="Output directory.",
    )
    parser.add_argument(
        "--crop-border-points",
        type=int,
        default=0,
        help="Optional border crop in grid points before fitting and roughness calculation.",
    )
    parser.add_argument(
        "--gaussian-cutoff-um",
        type=float,
        default=250.0,
        help=(
            "ISO-style Gaussian cutoff wavelength in micrometres. "
            "Default 250 um is usable inside a 1000 um scan; use 800 for the common Ra-based profile cutoff."
        ),
    )
    parser.add_argument(
        "--disable-gaussian-filter",
        action="store_true",
        help="Skip Gaussian roughness/waviness separation.",
    )
    parser.add_argument(
        "--gaussian-truncate-cutoffs",
        type=float,
        default=1.0,
        help="Kernel half-width in cutoff wavelengths. One cutoff is already near-zero at the edges.",
    )
    parser.add_argument(
        "--surface-plot-stride",
        type=int,
        default=2,
        help="Grid stride for rendered 3D surface PNGs.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip diagnostic PNG heatmaps.",
    )
    return parser.parse_args()


def infer_sample_from_path(path: Path) -> str | None:
    """Infer sample id `S1`-`S6` from a file path.

    原始目录和文件名常含 `TXF1-4`、`1-4` 或 `S4` 等模式。本函数只提取
    几何样件编号，不依赖个人目录名。

    Raw paths often contain patterns such as `TXF1-4`, `1-4`, or `S4`. This
    helper extracts only the geometry sample id and does not rely on personal
    folder names.
    """
    text = str(path)
    patterns = [
        r"TXF\d+-(\d+)",
        r"(?:^|/)\d+-(\d+)(?:-\d+)?(?:\.|/|$)",
        r"(?:^|/)S(\d)(?:\.|/|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            sample = f"S{match.group(1)}"
            if sample in SAMPLE_GEOMETRY_MM:
                return sample
    return None


def surface_radius_mm(
    sample: str | None,
    explicit_surface_radius_mm: float | None,
    planned_ae_mm: float,
    actual_ae_mm: float,
) -> tuple[float, dict[str, float | str]]:
    """Determine the actual machined surface radius.

    可直接传入表面半径；若未传入，则由样件中面半径、名义壁厚、计划径向
    余量和实际径向切深计算加工后表面半径。

    The surface radius can be provided directly. Otherwise it is derived from
    mid-surface radius, nominal thickness, planned radial stock, and actual
    radial engagement.
    """
    if explicit_surface_radius_mm is not None:
        return explicit_surface_radius_mm, {
            "radius_source": "explicit --surface-radius-mm",
            "surface_radius_mm": explicit_surface_radius_mm,
        }
    if sample is None:
        raise ValueError("sample could not be inferred; pass --sample or --surface-radius-mm")

    geometry = SAMPLE_GEOMETRY_MM[sample]
    mid_radius = geometry["mid_radius"]
    thickness = geometry["thickness"]
    surface_radius = mid_radius + thickness / 2.0 + planned_ae_mm - actual_ae_mm
    return surface_radius, {
        "radius_source": "sample geometry + planned/actual ae adjustment",
        "sample": sample,
        "nominal_mid_radius_mm": mid_radius,
        "thickness_mm": thickness,
        "planned_ae_mm": planned_ae_mm,
        "actual_ae_mm": actual_ae_mm,
        "surface_radius_mm": surface_radius,
    }


def read_point_cloud(path: Path) -> np.ndarray:
    """Read a finite `x,y,z[,intensity]` point cloud from CSV.

    三维轮廓仪导出的无效值会被剔除，只保留前三列有限坐标和可选强度列。

    Invalid profiler rows are removed. The first three finite coordinate
    columns and optional intensity column are retained.
    """
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("input CSV must contain at least x,y,z columns")
    data = data[:, :4] if data.shape[1] >= 4 else data[:, :3]
    data = data[np.isfinite(data[:, :3]).all(axis=1)]
    if len(data) == 0:
        raise ValueError("input CSV contains no finite x,y,z rows")
    return data


def maybe_crop_grid(data: np.ndarray, border: int) -> np.ndarray:
    """Optionally crop a fixed number of grid points from every border.

    边缘区域可能受扫描拼接或光学噪声影响。裁剪仅适用于完整矩形网格。

    Border regions may carry stitching or optical noise. Cropping is supported
    only for complete rectangular grids.
    """
    if border <= 0:
        return data
    x = data[:, 0]
    y = data[:, 1]
    xs = np.unique(x)
    ys = np.unique(y)
    if len(xs) * len(ys) != len(data):
        raise ValueError("--crop-border-points requires a complete rectangular grid")
    if border * 2 >= len(xs) or border * 2 >= len(ys):
        raise ValueError("--crop-border-points is too large for the grid")
    keep_x = (x >= xs[border]) & (x <= xs[-border - 1])
    keep_y = (y >= ys[border]) & (y <= ys[-border - 1])
    return data[keep_x & keep_y]


def fit_plane(xc: np.ndarray, yc: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Fit a least-squares plane `z = b0 + b1*x + b2*y`.

    该平面代表扫描倾斜和安装偏差，后续与圆柱形貌一起从原始高度中剔除。

    The plane represents scanner tilt and mounting offset. It is removed
    together with the cylindrical form.
    """
    design = np.column_stack([np.ones_like(xc), xc, yc])
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    return beta


def fit_quadratic_orientation(
    xc: np.ndarray, yc: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Estimate curvature direction from a quadratic height fit.

    二次曲面 Hessian 的主方向用于自动判断圆柱曲率方向，二阶导数符号用于
    判断曲面凹凸方向。

    The Hessian of the quadratic fit gives the main curvature direction; the
    second-derivative sign indicates concave/convex orientation.
    """
    design = np.column_stack([np.ones_like(xc), xc, yc, xc * xc, xc * yc, yc * yc])
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    hessian = np.array([[2.0 * beta[3], beta[4]], [beta[4], 2.0 * beta[5]]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    idx = int(np.argmax(np.abs(eigenvalues)))
    direction = eigenvectors[:, idx]
    direction = direction / np.linalg.norm(direction)
    curvature_second_derivative = float(eigenvalues[idx])
    estimated_radius_um = math.inf
    if abs(curvature_second_derivative) > 0:
        estimated_radius_um = 1.0 / abs(curvature_second_derivative)
    return direction, curvature_second_derivative, estimated_radius_um, beta


def direction_from_angle(angle_deg: float) -> np.ndarray:
    """Convert a user-provided direction angle to a unit vector.

    当自动拟合方向不可靠时，可手动指定曲率方向角。

    A manual curvature-direction angle can be used when automatic fitting is
    unreliable.
    """
    angle_rad = math.radians(angle_deg)
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def angle_from_direction(direction: np.ndarray) -> float:
    """Convert a 2D direction vector to degrees.

    输出角度写入摘要文件，便于复核自动识别的曲率方向。

    The output angle is written to the summary file for review.
    """
    return math.degrees(math.atan2(float(direction[1]), float(direction[0])))


def cylindrical_sag(u_um: np.ndarray, radius_um: float, sign: float) -> np.ndarray:
    """Compute cylindrical sag along the curvature direction.

    `u_um` 是沿曲率方向的弦向坐标。圆柱 sag 表示宏观曲面形貌，被视为
    需要去除的形状项。

    `u_um` is the chord coordinate along the curvature direction. Cylindrical
    sag is the macroscopic form component to remove.
    """
    if np.max(np.abs(u_um)) >= radius_um:
        raise ValueError("scan width is larger than the selected cylinder radius")
    return sign * (radius_um - np.sqrt(radius_um * radius_um - u_um * u_um))


def roughness_metrics(z_flat_um: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute areal roughness metrics from a flattened height vector.

    指标包括 Sa、Sq、Sp、Sv、Sz、Ssk 和 Sku，同时给出 nm 换算值。

    Metrics include Sa, Sq, Sp, Sv, Sz, Ssk, and Sku, with nanometre conversions.
    """
    residual = z_flat_um - float(np.mean(z_flat_um))
    sq = float(np.sqrt(np.mean(residual * residual)))
    metrics = {
        f"{prefix}Sa_um": float(np.mean(np.abs(residual))),
        f"{prefix}Sq_um": sq,
        f"{prefix}Sp_um": float(np.max(residual)),
        f"{prefix}Sv_um": float(-np.min(residual)),
        f"{prefix}Sz_um": float(np.max(residual) - np.min(residual)),
        f"{prefix}mean_um": float(np.mean(z_flat_um)),
    }
    if sq > 0:
        metrics[f"{prefix}Ssk"] = float(np.mean(residual**3) / sq**3)
        metrics[f"{prefix}Sku"] = float(np.mean(residual**4) / sq**4)
    else:
        metrics[f"{prefix}Ssk"] = 0.0
        metrics[f"{prefix}Sku"] = 0.0
    metrics[f"{prefix}Sa_nm"] = metrics[f"{prefix}Sa_um"] * 1000.0
    metrics[f"{prefix}Sq_nm"] = metrics[f"{prefix}Sq_um"] * 1000.0
    metrics[f"{prefix}Sz_nm"] = metrics[f"{prefix}Sz_um"] * 1000.0
    return metrics


def grid_from_points(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reshape scattered rows into a rectangular grid when possible.

    Gaussian 二维滤波和热力图绘制需要完整矩形网格。若点云缺点或顺序异常，
    函数返回 `None`。

    Two-dimensional Gaussian filtering and heatmaps require a complete
    rectangular grid. The function returns `None` for missing or irregular rows.
    """
    xs = np.unique(x)
    ys = np.unique(y)
    if len(xs) * len(ys) != len(values):
        return None
    order = np.lexsort((x, y))
    x_sorted = x[order]
    y_sorted = y[order]
    if not (np.array_equal(np.tile(xs, len(ys)), x_sorted) and np.array_equal(np.repeat(ys, len(xs)), y_sorted)):
        return None
    grid = values[order].reshape(len(ys), len(xs))
    return xs, ys, grid


def profile_metrics(lines: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute line-wise Ra/Rq/Rz summaries across grid rows or columns.

    这些指标用于辅助判断 x 方向和 y 方向轮廓粗糙度是否存在明显差异。

    These metrics help compare roughness behaviour along x and y profile
    directions.
    """
    centered = lines - np.mean(lines, axis=1, keepdims=True)
    ra = np.mean(np.abs(centered), axis=1)
    rq = np.sqrt(np.mean(centered * centered, axis=1))
    rz = np.max(centered, axis=1) - np.min(centered, axis=1)
    return {
        f"{prefix}Ra_mean_um": float(np.mean(ra)),
        f"{prefix}Ra_std_um": float(np.std(ra)),
        f"{prefix}Rq_mean_um": float(np.mean(rq)),
        f"{prefix}Rz_mean_um": float(np.mean(rz)),
        f"{prefix}center_Ra_um": float(ra[len(ra) // 2]),
        f"{prefix}center_Rq_um": float(rq[len(rq) // 2]),
        f"{prefix}center_Rz_um": float(rz[len(rz) // 2]),
    }


def gaussian_kernel_1d(cutoff_um: float, spacing_um: float, truncate_cutoffs: float) -> np.ndarray:
    """Build a one-dimensional Gaussian weighting kernel.

    `cutoff_um` 为截止波长，`truncate_cutoffs` 控制核半宽。核归一化后用于
    反射边界卷积。

    `cutoff_um` is the cutoff wavelength and `truncate_cutoffs` controls kernel
    half-width. The normalized kernel is used with reflect-padding convolution.
    """
    if cutoff_um <= 0:
        raise ValueError("--gaussian-cutoff-um must be positive")
    if spacing_um <= 0:
        raise ValueError("grid spacing must be positive")
    if truncate_cutoffs <= 0:
        raise ValueError("--gaussian-truncate-cutoffs must be positive")

    half_width = max(1, int(math.ceil(truncate_cutoffs * cutoff_um / spacing_um)))
    x = np.arange(-half_width, half_width + 1, dtype=float) * spacing_um
    kernel = np.exp(-math.pi * (x / (ISO_GAUSSIAN_ALPHA * cutoff_um)) ** 2)
    kernel /= float(np.sum(kernel))
    return kernel


def convolve_reflect_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Apply one-dimensional convolution along an axis with reflect padding.

    反射边界减少滤波边缘处的突变，适合有限扫描窗口内的表面高度数据。

    Reflect padding reduces edge jumps and suits finite scanning windows.
    """
    pad = len(kernel) // 2

    def convolve_line(line: np.ndarray) -> np.ndarray:
        padded = np.pad(line, pad, mode="reflect")
        return np.convolve(padded, kernel, mode="valid")

    return np.apply_along_axis(convolve_line, axis, values)


def gaussian_waviness_roughness(
    primary_grid_um: np.ndarray,
    dx_um: float,
    dy_um: float,
    cutoff_um: float,
    truncate_cutoffs: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Separate Gaussian waviness and roughness surfaces.

    先沿 x 方向卷积，再沿 y 方向卷积得到波纹面；粗糙度面为去形残余面
    减去波纹面，并重新去均值。

    The waviness surface is obtained by x-axis then y-axis convolution. The
    roughness surface is the form-removed residual minus waviness, then
    re-centered.
    """
    x_kernel = gaussian_kernel_1d(cutoff_um, dx_um, truncate_cutoffs)
    y_kernel = gaussian_kernel_1d(cutoff_um, dy_um, truncate_cutoffs)
    waviness = convolve_reflect_axis(primary_grid_um, x_kernel, axis=1)
    waviness = convolve_reflect_axis(waviness, y_kernel, axis=0)
    roughness = primary_grid_um - waviness
    roughness = roughness - float(np.mean(roughness))
    return waviness, roughness, len(x_kernel), len(y_kernel)


def write_key_value_csv(path: Path, rows: Iterable[tuple[str, object]]) -> None:
    """Write summary values as a simple `key,value` CSV.

    键值格式便于人工阅读，也便于批量汇总脚本按字段名读取。

    A key-value format is readable by humans and easy for the batch summary
    script to parse by field name.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in rows:
            writer.writerow([key, value])


def write_xlsx(path: Path, summary: dict[str, object], comparison_rows: list[dict[str, object]]) -> None:
    """Write optional Excel summaries when `openpyxl` is available.

    Excel 文件供人工核对。若环境缺少 openpyxl，脚本跳过该文件，不影响 CSV
    和图像输出。

    The Excel file is for manual review. If `openpyxl` is unavailable, this
    export is skipped without affecting CSV or image outputs.
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["key", "value"])
    for key, value in summary.items():
        ws.append([key, value])

    ws2 = wb.create_sheet("method_comparison")
    if comparison_rows:
        headers = list(comparison_rows[0].keys())
        ws2.append(headers)
        for row in comparison_rows:
            ws2.append([row.get(header) for header in headers])

    for sheet in wb.worksheets:
        for column in sheet.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
            sheet.column_dimensions[column[0].column_letter].width = min(max(max_len + 2, 12), 46)
    wb.save(path)


def color_ramp(t: np.ndarray, diverging: bool) -> np.ndarray:
    """Map normalized scalar values to RGB colors for heatmaps.

    发散色带用于残余高度，顺序色带用于原始高度和形貌项。

    Diverging colors are used for residual heights; sequential colors are used
    for raw height and form components.
    """
    t = np.clip(t, 0.0, 1.0)
    if diverging:
        blue = np.array([49, 97, 159], dtype=float)
        white = np.array([248, 248, 248], dtype=float)
        red = np.array([178, 24, 43], dtype=float)
        rgb = np.empty((*t.shape, 3), dtype=float)
        lower = t <= 0.5
        upper = ~lower
        rgb[lower] = blue + (white - blue) * (t[lower, None] / 0.5)
        rgb[upper] = white + (red - white) * ((t[upper, None] - 0.5) / 0.5)
        return rgb.astype(np.uint8)

    stops = np.array(
        [
            [35, 55, 95],
            [33, 122, 145],
            [117, 183, 119],
            [246, 220, 122],
        ],
        dtype=float,
    )
    scaled = t * (len(stops) - 1)
    idx = np.floor(scaled).astype(int)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = scaled - idx
    rgb = stops[idx] + (stops[idx + 1] - stops[idx]) * frac[..., None]
    return rgb.astype(np.uint8)


def jet_ramp(t: np.ndarray) -> np.ndarray:
    """Map normalized values to a profiler-like multi-color ramp.

    该色带用于三维表面渲染，帮助模拟轮廓仪软件常见的峰谷显示风格。

    This ramp is used for 3D surface rendering and mimics common profiler
    peak/valley coloring.
    """
    t = np.clip(t, 0.0, 1.0)
    stops = np.array(
        [
            [0, 0, 150],
            [0, 92, 255],
            [0, 220, 220],
            [90, 220, 50],
            [255, 220, 0],
            [255, 90, 0],
            [180, 0, 0],
        ],
        dtype=float,
    )
    scaled = t * (len(stops) - 1)
    idx = np.floor(scaled).astype(int)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = scaled - idx
    rgb = stops[idx] + (stops[idx + 1] - stops[idx]) * frac[..., None]
    return rgb.astype(np.uint8)


def write_heatmap(path: Path, grid: np.ndarray, diverging: bool = False) -> None:
    """Render a 2D heatmap image from a height grid.

    图像采用百分位裁剪，减少少量极端点对颜色范围的影响。

    Percentile clipping is used so a few extreme points do not dominate the
    color scale.
    """
    try:
        from PIL import Image
    except ImportError:
        return

    values = np.asarray(grid, dtype=float)
    if diverging:
        limit = float(np.percentile(np.abs(values - np.mean(values)), 99.0))
        limit = max(limit, 1e-12)
        normalized = (np.clip(values, -limit, limit) + limit) / (2.0 * limit)
    else:
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            high = low + 1e-12
        normalized = (np.clip(values, low, high) - low) / (high - low)
    image = Image.fromarray(color_ramp(normalized, diverging=diverging), mode="RGB")
    scale = max(1, int(math.ceil(900 / max(image.size))))
    image = image.resize((image.size[0] * scale, image.size[1] * scale), Image.Resampling.NEAREST)
    image.save(path)


def strided_indices(length: int, stride: int) -> np.ndarray:
    """Return down-sampling indices while retaining the last grid point.

    三维表面图不需要绘制每个网格点；保留末点可避免边界缺口。

    3D surface plots do not need every grid point; retaining the final point
    avoids boundary gaps.
    """
    stride = max(1, int(stride))
    idx = list(range(0, length, stride))
    if idx[-1] != length - 1:
        idx.append(length - 1)
    return np.array(idx, dtype=int)


def load_font(size: int):
    """Load a common system font for PIL-rendered labels.

    若指定字体缺失，则回退到 PIL 默认字体，保证图像仍可生成。

    If preferred fonts are missing, the PIL default font is used so rendering
    can continue.
    """
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def format_um(value: float) -> str:
    """Format micrometre values with precision adapted to magnitude.

    小数值保留更多有效位，大数值减少小数位，避免图例标签拥挤。

    Small values keep more precision; large values use fewer decimals to avoid
    crowded labels.
    """
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.4f}um"
    if magnitude >= 10:
        return f"{value:.3f}um"
    if magnitude >= 1:
        return f"{value:.4f}um"
    return f"{value:.5f}um"


def write_3d_surface_plot(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    z_grid: np.ndarray,
    title: str,
    stride: int = 2,
    width: int = 2400,
    height: int = 1400,
) -> None:
    """Render a lightweight 3D-like surface image using PIL.

    该函数不依赖 Matplotlib 三维轴，直接把网格投影到二维画布。这样可在
    简化环境中生成用于报告核对的三维表面图。

    This function avoids Matplotlib 3D axes and projects the grid directly onto
    a 2D canvas. It provides review images in lightweight environments.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    x_idx = strided_indices(len(xs), stride)
    y_idx = strided_indices(len(ys), stride)
    xs_s = xs[x_idx]
    ys_s = ys[y_idx]
    z_s = z_grid[np.ix_(y_idx, x_idx)]

    z_min = float(np.min(z_grid))
    z_max = float(np.max(z_grid))
    z_span = max(z_max - z_min, 1e-12)

    x_norm = (xs_s - float(xs_s.min())) / max(float(xs_s.max() - xs_s.min()), 1e-12) - 0.5
    y_norm = (ys_s - float(ys_s.min())) / max(float(ys_s.max() - ys_s.min()), 1e-12) - 0.5
    xx, yy = np.meshgrid(x_norm, y_norm)
    zz = ((z_s - z_min) / z_span - 0.5) * 0.72

    azimuth = math.radians(-45.0)
    elevation = math.radians(34.0)
    x_rot = math.cos(azimuth) * xx - math.sin(azimuth) * yy
    y_rot = math.sin(azimuth) * xx + math.cos(azimuth) * yy
    proj_x = x_rot
    proj_y = y_rot * math.cos(elevation) - zz * math.sin(elevation)
    depth = y_rot * math.sin(elevation) + zz * math.cos(elevation)

    plot_left, plot_right = 360, width - 120
    plot_top, plot_bottom = 170, height - 110
    px_span = max(float(proj_x.max() - proj_x.min()), 1e-12)
    py_span = max(float(proj_y.max() - proj_y.min()), 1e-12)
    scale = min((plot_right - plot_left) / px_span, (plot_bottom - plot_top) / py_span) * 0.92
    center_x = (plot_left + plot_right) / 2.0
    center_y = (plot_top + plot_bottom) / 2.0
    proj_center_x = (float(proj_x.max()) + float(proj_x.min())) / 2.0
    proj_center_y = (float(proj_y.max()) + float(proj_y.min())) / 2.0
    screen_x = center_x + (proj_x - proj_center_x) * scale
    screen_y = center_y - (proj_y - proj_center_y) * scale

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_small = load_font(26)
    font_medium = load_font(32)
    font_title = load_font(38)

    if title:
        draw.text((360, 48), title, fill=(20, 20, 20), font=font_title)

    # Draw a light base outline before the surface.
    # 先绘制浅色底面轮廓，给三维投影提供空间参照。
    base_z = np.full((2, 2), -0.5 * 0.72)
    base_xx = np.array([[-0.5, 0.5], [-0.5, 0.5]])
    base_yy = np.array([[-0.5, -0.5], [0.5, 0.5]])
    bx_rot = math.cos(azimuth) * base_xx - math.sin(azimuth) * base_yy
    by_rot = math.sin(azimuth) * base_xx + math.cos(azimuth) * base_yy
    bpx = bx_rot
    bpy = by_rot * math.cos(elevation) - base_z * math.sin(elevation)
    bsx = center_x + (bpx - proj_center_x) * scale
    bsy = center_y - (bpy - proj_center_y) * scale
    base_corners = [
        (float(bsx[0, 0]), float(bsy[0, 0])),
        (float(bsx[0, 1]), float(bsy[0, 1])),
        (float(bsx[1, 1]), float(bsy[1, 1])),
        (float(bsx[1, 0]), float(bsy[1, 0])),
    ]
    draw.line(base_corners + [base_corners[0]], fill=(205, 205, 205), width=2)

    cells = []
    ny, nx = z_s.shape
    gy, gx = np.gradient(zz)
    shade = 0.78 + 0.22 * np.clip(1.0 - 4.0 * np.hypot(gx, gy), 0.0, 1.0)
    for i in range(ny - 1):
        for j in range(nx - 1):
            pts = [
                (float(screen_x[i, j]), float(screen_y[i, j])),
                (float(screen_x[i, j + 1]), float(screen_y[i, j + 1])),
                (float(screen_x[i + 1, j + 1]), float(screen_y[i + 1, j + 1])),
                (float(screen_x[i + 1, j]), float(screen_y[i + 1, j])),
            ]
            z_avg = float(np.mean(z_s[i : i + 2, j : j + 2]))
            t = (z_avg - z_min) / z_span
            rgb = jet_ramp(np.array(t, dtype=float)).astype(float)
            cell_shade = float(np.mean(shade[i : i + 2, j : j + 2]))
            rgb = np.clip(rgb * cell_shade, 0, 255).astype(np.uint8)
            cell_depth = float(np.mean(depth[i : i + 2, j : j + 2]))
            cells.append((cell_depth, pts, tuple(int(v) for v in rgb)))

    for _, pts, rgb in sorted(cells, key=lambda item: item[0]):
        draw.polygon(pts, fill=rgb)

    # Draw top outline after the surface.
    # 表面绘制完成后再画顶面外框，避免外框被色块遮挡。
    outline = [
        (float(screen_x[0, 0]), float(screen_y[0, 0])),
        (float(screen_x[0, -1]), float(screen_y[0, -1])),
        (float(screen_x[-1, -1]), float(screen_y[-1, -1])),
        (float(screen_x[-1, 0]), float(screen_y[-1, 0])),
    ]
    draw.line(outline + [outline[0]], fill=(170, 170, 170), width=2)

    # Corner labels, matching the profiler-style visual cue without crowding the image.
    # 角点标签模拟轮廓仪软件的视觉提示，同时控制文字密度。
    label_color = (25, 25, 25)
    draw.text((outline[0][0] - 8, outline[0][1] + 18), "0um", fill=label_color, font=font_small)
    draw.text((outline[1][0] - 20, outline[1][1] + 18), f"{xs.max():.4f}", fill=label_color, font=font_small)
    draw.text((outline[3][0] - 90, outline[3][1] + 16), f"{ys.max():.4f}um", fill=label_color, font=font_small)

    peak_idx = np.unravel_index(int(np.argmax(z_s)), z_s.shape)
    draw.text(
        (float(screen_x[peak_idx]) + 10, float(screen_y[peak_idx]) - 42),
        format_um(z_max),
        fill=label_color,
        font=font_small,
    )

    # Colorbar.
    # 色标使用同一 jet-like 色带，便于读者把颜色与高度范围对应起来。
    cb_x, cb_y, cb_w, cb_h = 62, 78, 18, 315
    for row in range(cb_h):
        t = 1.0 - row / max(cb_h - 1, 1)
        color = tuple(int(v) for v in jet_ramp(np.array(t, dtype=float)))
        draw.line((cb_x, cb_y + row, cb_x + cb_w, cb_y + row), fill=color)
    draw.rectangle((cb_x, cb_y, cb_x + cb_w, cb_y + cb_h), outline=(245, 245, 245), width=1)
    for tick in np.linspace(z_max, z_min, 6):
        y_tick = cb_y + (z_max - float(tick)) / z_span * cb_h
        draw.line((cb_x + cb_w + 4, y_tick, cb_x + cb_w + 14, y_tick), fill=(70, 70, 70), width=1)
        draw.text((cb_x + cb_w + 20, y_tick - 16), format_um(float(tick)), fill=label_color, font=font_medium)

    image.save(path)


def main() -> None:
    """Run the full curved-surface roughness-processing workflow.

    主流程包括：读取点云、确定样件半径、自动或手动确定曲率方向、去除
    圆柱形貌和平面倾斜、计算去形残余面、可选 Gaussian 滤波、写出表格、
    报告、追踪文件和诊断图。

    The workflow reads the point cloud, determines sample radius, estimates or
    accepts curvature direction, removes cylindrical form and plane tilt,
    computes the residual surface, optionally applies Gaussian filtering, and
    writes tables, reports, trace files, and diagnostic images.
    """
    args = parse_args()
    input_csv = args.input_csv if args.input_csv.is_absolute() else (Path.cwd() / args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    # Infer or read the machined surface radius before form removal. The radius
    # fixes the macroscopic cylindrical component that should not be counted as
    # roughness.
    # 去形前先确定加工表面半径。该半径决定宏观圆柱形貌，宏观形貌不纳入
    # 粗糙度起伏。
    sample = args.sample or infer_sample_from_path(input_csv)
    radius_mm, radius_meta = surface_radius_mm(
        sample,
        args.surface_radius_mm,
        args.planned_ae_mm,
        args.actual_ae_mm,
    )
    radius_um = radius_mm * 1000.0

    data = maybe_crop_grid(read_point_cloud(input_csv), args.crop_border_points)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    intensity = data[:, 3] if data.shape[1] >= 4 else np.full(len(data), np.nan)

    # Center x and y before fitting. This improves numerical conditioning and
    # makes plane/quadratic coefficients easier to interpret.
    # 拟合前对 x、y 去中心，改善最小二乘数值条件，也让平面和二次项系数更稳定。
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    auto_direction, auto_second_derivative, auto_radius_um, quadratic_beta = fit_quadratic_orientation(xc, yc, z)

    # Curvature direction can come from the automatic quadratic fit or from an
    # explicit command-line angle. The chosen direction defines chord coordinate
    # `u` for cylindrical sag removal.
    # 曲率方向可由二次拟合自动识别，也可由命令行角度指定。选定方向后，
    # 弦向坐标 u 用于计算圆柱 sag。
    if args.curvature_angle_deg is None:
        curvature_direction = auto_direction
        curvature_angle_deg = angle_from_direction(curvature_direction)
        angle_source = "auto quadratic fit"
    else:
        curvature_direction = direction_from_angle(args.curvature_angle_deg)
        curvature_angle_deg = args.curvature_angle_deg
        angle_source = "explicit --curvature-angle-deg"

    if args.curvature_sign == "auto":
        sign = -1.0 if auto_second_derivative < 0 else 1.0
        sign_source = "auto quadratic fit"
    else:
        sign = 1.0 if args.curvature_sign == "positive" else -1.0
        sign_source = f"explicit --curvature-sign {args.curvature_sign}"

    # Rotate centered coordinates into curvature direction `u` and cylinder-axis
    # direction `v`, then compute the known-radius cylindrical form.
    # 将去中心坐标旋转到曲率方向 u 和圆柱轴向 v，再计算固定半径圆柱形貌。
    u = xc * curvature_direction[0] + yc * curvature_direction[1]
    v = -xc * curvature_direction[1] + yc * curvature_direction[0]
    sag = cylindrical_sag(u, radius_um, sign)

    # Fit scanner tilt after subtracting cylindrical sag. The removed form is
    # the sum of the fitted plane and cylindrical sag.
    # 先扣除圆柱 sag 后拟合扫描倾斜平面。最终剔除形貌为平面项与圆柱项之和。
    plane_beta = fit_plane(xc, yc, z - sag)
    plane = plane_beta[0] + plane_beta[1] * xc + plane_beta[2] * yc
    form = plane + sag
    residual = z - form
    residual = residual - float(np.mean(residual))

    # Arc coordinate is retained in the flattened point cloud to preserve the
    # geometric meaning of the curved direction after form removal.
    # 去形后的点云仍保留曲率方向弧长坐标，便于后续追溯曲面几何位置。
    arc_coord = radius_um * np.arcsin(u / radius_um)

    # Alternative residuals are kept for method comparison: plane-only removal
    # and free quadratic removal. They are not the retained Chapter 4 result.
    # 方法对比保留仅去平面和自由二次曲面去形结果；第 4 章保留结果使用固定
    # 半径圆柱去形和 Gaussian 粗糙度面。
    plane_only_beta = fit_plane(xc, yc, z)
    plane_only_residual = z - (
        plane_only_beta[0] + plane_only_beta[1] * xc + plane_only_beta[2] * yc
    )
    quadratic_design = np.column_stack([np.ones_like(xc), xc, yc, xc * xc, xc * yc, yc * yc])
    quadratic_residual = z - quadratic_design @ quadratic_beta

    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_columns = np.column_stack(
        [
            x,
            y,
            z,
            intensity,
            u,
            v,
            arc_coord,
            form,
            residual,
        ]
    )
    flat_header = ",".join(
        [
            "x_um",
            "y_um",
            "z_um",
            "intensity",
            "curvature_chord_um",
            "cylinder_axis_coord_um",
            "curvature_arc_flat_um",
            "removed_form_z_um",
            "flattened_z_residual_um",
        ]
    )
    flattened_path = output_dir / "flattened_point_cloud.csv"
    np.savetxt(flattened_path, flat_columns, delimiter=",", header=flat_header, comments="", fmt="%.10g")

    # Summary values combine provenance, geometric settings, fitting diagnostics,
    # and retained roughness metrics. The batch script reads this file later.
    # 摘要信息合并来源、几何参数、拟合诊断和粗糙度指标，后续批量汇总脚本会读取它。
    summary: dict[str, object] = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "method": "best-fit plane plus fixed-radius cylindrical form removal",
        "unit_xyz": "um",
        "point_count": int(len(data)),
        "x_min_um": float(np.min(x)),
        "x_max_um": float(np.max(x)),
        "y_min_um": float(np.min(y)),
        "y_max_um": float(np.max(y)),
        "z_min_um": float(np.min(z)),
        "z_max_um": float(np.max(z)),
        "crop_border_points": int(args.crop_border_points),
        **radius_meta,
        "surface_radius_um": radius_um,
        "curvature_angle_deg": curvature_angle_deg,
        "curvature_angle_source": angle_source,
        "curvature_sign": "positive" if sign > 0 else "negative",
        "curvature_sign_source": sign_source,
        "auto_quadratic_second_derivative_1_per_um": auto_second_derivative,
        "auto_quadratic_radius_estimate_um": auto_radius_um,
        "plane_intercept_um": float(plane_beta[0]),
        "plane_slope_x_um_per_um": float(plane_beta[1]),
        "plane_slope_y_um_per_um": float(plane_beta[2]),
    }
    summary.update(roughness_metrics(residual, "form_removed_unfiltered_"))

    grid = grid_from_points(x, y, residual)
    raw_grid = grid_from_points(x, y, z)
    form_grid = grid_from_points(x, y, form)
    gaussian_waviness_grid = None
    gaussian_roughness_grid = None
    if grid is not None:
        # Grid spacing is inferred from unique x and y coordinates. The profiler
        # data used in the thesis are regular 250 by 250 grids.
        # 网格间距由唯一 x、y 坐标推断。论文保留数据为规则 250 x 250 网格。
        xs, ys, residual_grid = grid
        dx_um = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0
        dy_um = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.0
        summary.update(
            {
                "grid_x_count": int(len(xs)),
                "grid_y_count": int(len(ys)),
                "grid_dx_um": dx_um,
                "grid_dy_um": dy_um,
            }
        )
        summary.update(profile_metrics(residual_grid, "x_profile_"))
        summary.update(profile_metrics(residual_grid.T, "y_profile_"))
        if not args.disable_gaussian_filter:
            # Gaussian filtering splits the form-removed surface into waviness
            # and roughness. Chapter 4 keeps the roughness component as the main
            # surface for Sa/Sq/Sz.
            # Gaussian 滤波把去形残余面分离为波纹面和粗糙度面。第 4 章采用
            # 粗糙度面计算 Sa/Sq/Sz。
            gaussian_waviness_grid, gaussian_roughness_grid, kernel_x_count, kernel_y_count = (
                gaussian_waviness_roughness(
                    residual_grid,
                    dx_um,
                    dy_um,
                    args.gaussian_cutoff_um,
                    args.gaussian_truncate_cutoffs,
                )
            )
            summary.update(
                {
                    "gaussian_filter_enabled": True,
                    "gaussian_filter_standard": "ISO 16610-style Gaussian weighting; alpha=sqrt(ln(2)/pi), 50 percent transmission at cutoff",
                    "gaussian_cutoff_um": float(args.gaussian_cutoff_um),
                    "gaussian_alpha": ISO_GAUSSIAN_ALPHA,
                    "gaussian_truncate_cutoffs": float(args.gaussian_truncate_cutoffs),
                    "gaussian_kernel_x_count": int(kernel_x_count),
                    "gaussian_kernel_y_count": int(kernel_y_count),
                    "gaussian_filter_boundary": "reflect padding",
                }
            )
            summary.update(roughness_metrics(gaussian_waviness_grid.ravel(), "gaussian_waviness_"))
            summary.update(roughness_metrics(gaussian_roughness_grid.ravel(), "gaussian_roughness_"))
            summary.update(profile_metrics(gaussian_roughness_grid, "gaussian_x_profile_"))
            summary.update(profile_metrics(gaussian_roughness_grid.T, "gaussian_y_profile_"))

            grid_xx, grid_yy = np.meshgrid(xs, ys)
            filtered_columns = np.column_stack(
                [
                    grid_xx.ravel(),
                    grid_yy.ravel(),
                    residual_grid.ravel(),
                    gaussian_waviness_grid.ravel(),
                    gaussian_roughness_grid.ravel(),
                ]
            )
            filtered_header = ",".join(
                [
                    "x_um",
                    "y_um",
                    "form_removed_unfiltered_z_um",
                    "gaussian_waviness_z_um",
                    "gaussian_roughness_z_um",
                ]
            )
            np.savetxt(
                output_dir / "gaussian_filtered_point_cloud.csv",
                filtered_columns,
                delimiter=",",
                header=filtered_header,
                comments="",
                fmt="%.10g",
            )
        else:
            summary["gaussian_filter_enabled"] = False

    if gaussian_roughness_grid is not None:
        # Prefer Gaussian roughness as the retained result when filtering is
        # available; otherwise fall back to the unfiltered form-removed surface.
        # 若 Gaussian 粗糙度面可用，则作为保留结果；否则使用未滤波的去形残余面。
        summary.update(roughness_metrics(gaussian_roughness_grid.ravel()))
        summary["main_result_surface"] = "gaussian_roughness"
    else:
        summary.update(roughness_metrics(residual))
        summary["main_result_surface"] = "form_removed_unfiltered"

    comparison_rows = []
    # Method comparison keeps alternative form-removal routes visible in the
    # output folder. This helps audit whether the retained fixed-radius approach
    # changes roughness scale relative to simpler baselines.
    # 方法对比表保留不同去形路线的指标，便于核对固定半径去形相对简单基线
    # 对粗糙度量级的影响。
    for method, values in [
        ("plane_only", plane_only_residual),
        ("free_quadratic_form", quadratic_residual),
        ("fixed_radius_cylinder_unfiltered", residual),
    ]:
        row: dict[str, object] = {"method": method}
        row.update(roughness_metrics(values))
        comparison_rows.append(row)
    if gaussian_waviness_grid is not None and gaussian_roughness_grid is not None:
        for method, values in [
            ("gaussian_waviness", gaussian_waviness_grid.ravel()),
            ("gaussian_roughness", gaussian_roughness_grid.ravel()),
        ]:
            row = {"method": method}
            row.update(roughness_metrics(values))
            comparison_rows.append(row)

    summary_path = output_dir / "roughness_summary.csv"
    write_key_value_csv(summary_path, summary.items())
    write_xlsx(output_dir / "roughness_summary.xlsx", summary, comparison_rows)

    comparison_path = output_dir / "method_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    report_path = output_dir / "processing_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Curved surface roughness processing report\n")
        f.write("==========================================\n\n")
        f.write(f"Input: {input_csv}\n")
        f.write(f"Method: {summary['method']}\n")
        f.write("Form removed: scanner tilt plane + known-radius single-curvature cylinder.\n")
        if summary.get("gaussian_filter_enabled"):
            f.write(
                "Gaussian separation: roughness = form-removed surface - Gaussian waviness surface.\n"
            )
            f.write(
                "Gaussian weighting: alpha=sqrt(ln(2)/pi), giving 50 percent transmission at cutoff.\n"
            )
            f.write(f"Gaussian cutoff: {summary['gaussian_cutoff_um']} um\n\n")
        else:
            f.write("Gaussian roughness/waviness filter: disabled.\n\n")
        f.write("Radius\n")
        f.write("------\n")
        for key in radius_meta:
            f.write(f"{key}: {radius_meta[key]}\n")
        f.write(f"surface_radius_um: {radius_um}\n\n")
        f.write("Main roughness result\n")
        f.write("---------------------\n")
        f.write(f"main_result_surface: {summary['main_result_surface']}\n")
        for key in ["Sa_um", "Sq_um", "Sp_um", "Sv_um", "Sz_um", "Ssk", "Sku", "Sa_nm", "Sq_nm", "Sz_nm"]:
            f.write(f"{key}: {summary[key]}\n")
        f.write("\nUnfiltered form-removed surface\n")
        f.write("-------------------------------\n")
        for key in [
            "form_removed_unfiltered_Sa_um",
            "form_removed_unfiltered_Sq_um",
            "form_removed_unfiltered_Sz_um",
        ]:
            f.write(f"{key}: {summary[key]}\n")
        if summary.get("gaussian_filter_enabled"):
            f.write("\nGaussian waviness surface\n")
            f.write("-------------------------\n")
            for key in [
                "gaussian_waviness_Sa_um",
                "gaussian_waviness_Sq_um",
                "gaussian_waviness_Sz_um",
            ]:
                f.write(f"{key}: {summary[key]}\n")
        f.write("\nDiagnostics\n")
        f.write("-----------\n")
        for key in [
            "curvature_angle_deg",
            "curvature_sign",
            "auto_quadratic_second_derivative_1_per_um",
            "auto_quadratic_radius_estimate_um",
            "plane_slope_x_um_per_um",
            "plane_slope_y_um_per_um",
        ]:
            f.write(f"{key}: {summary[key]}\n")

    if not args.no_images:
        # Diagnostic images are optional. Tables remain the authoritative
        # numerical output, while images support visual inspection of fitting
        # and filtering effects.
        # 诊断图可选生成。表格是正式数值输出，图像用于目视检查去形和滤波效果。
        if raw_grid is not None:
            write_heatmap(output_dir / "raw_height_heatmap.png", raw_grid[2], diverging=False)
            write_3d_surface_plot(
                output_dir / "raw_height_3d.png",
                raw_grid[0],
                raw_grid[1],
                raw_grid[2],
                "Raw height surface",
                stride=args.surface_plot_stride,
            )
        if form_grid is not None:
            write_heatmap(output_dir / "removed_form_heatmap.png", form_grid[2], diverging=False)
        if grid is not None:
            write_heatmap(output_dir / "flattened_residual_heatmap.png", grid[2], diverging=True)
            write_3d_surface_plot(
                output_dir / "flattened_residual_3d.png",
                grid[0],
                grid[1],
                grid[2],
                "Form-removed primary surface",
                stride=args.surface_plot_stride,
            )
        if gaussian_waviness_grid is not None and gaussian_roughness_grid is not None and grid is not None:
            write_heatmap(output_dir / "gaussian_waviness_heatmap.png", gaussian_waviness_grid, diverging=True)
            write_heatmap(output_dir / "gaussian_roughness_heatmap.png", gaussian_roughness_grid, diverging=True)
            write_3d_surface_plot(
                output_dir / "flattened_roughness_3d.png",
                grid[0],
                grid[1],
                gaussian_roughness_grid,
                f"Gaussian roughness surface, cutoff {args.gaussian_cutoff_um:g} um",
                stride=args.surface_plot_stride,
            )
        elif grid is not None:
            write_3d_surface_plot(
                output_dir / "flattened_roughness_3d.png",
                grid[0],
                grid[1],
                grid[2],
                "Form-removed roughness surface",
                stride=args.surface_plot_stride,
            )

    trace_path = output_dir / "processing_trace.md"
    rendered_outputs = [
        "roughness_summary.csv",
        "roughness_summary.xlsx",
        "method_comparison.csv",
        "processing_report.txt",
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
    with trace_path.open("w", encoding="utf-8") as f:
        # The trace file records the exact command and retained outputs for one
        # surface, making batch results easier to audit later.
        # 追踪文件记录单个加工面的运行命令和保留文件，便于后续审计批量结果。
        f.write(f"# {output_dir.name} Surface Roughness Processing Trace\n\n")
        f.write("## Scope\n\n")
        f.write(f"- Input point cloud: `{input_csv}`\n")
        f.write(f"- Output folder: `{output_dir}`\n")
        f.write(f"- Main retained script: `{Path(__file__).resolve()}`\n\n")
        f.write("## Processing Command\n\n")
        command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
        f.write("```bash\n")
        f.write(command)
        f.write("\n```\n\n")
        f.write("## Key Parameters\n\n")
        for key in [
            "sample",
            "nominal_mid_radius_mm",
            "thickness_mm",
            "planned_ae_mm",
            "actual_ae_mm",
            "surface_radius_mm",
            "gaussian_cutoff_um",
            "gaussian_filter_boundary",
            "curvature_angle_deg",
            "curvature_sign",
        ]:
            if key in summary:
                f.write(f"- {key}: `{summary[key]}`\n")
        f.write("\n## Current Main Result\n\n")
        for key in ["Sa_um", "Sq_um", "Sz_um", "Sa_nm", "Sq_nm", "Sz_nm", "Ssk", "Sku"]:
            f.write(f"- `{key} = {summary[key]}`\n")
        f.write("\n## Retained Outputs\n\n")
        for name in rendered_outputs:
            if (output_dir / name).exists():
                f.write(f"- `{name}`\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {flattened_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
