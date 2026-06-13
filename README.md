# Thesis Data Processing Scripts

本目录整理本科论文中承担主要数据处理任务的脚本副本。原始 ODB、位移导出表、三坐标工作簿、三维轮廓仪点云、实验照片和论文 DOCX 未放入本目录。脚本中的数据路径已经泛化为示意路径，开源使用时按同名结构放入等价数据即可。

## 目录结构

```text
open_source_data_processing_scripts/
├── README.md
├── requirements.txt
├── scripts/
│   ├── process_curved_surface_roughness.py
│   ├── summarize_surface_roughness_batch.py
│   └── abaqus/
│       ├── case_config.yml / case_config.json
│       ├── common.py
│       ├── build_displacement_dataset.py
│       ├── upgrade_odb_files.py
│       ├── extract_cut_progress_status.py
│       ├── adjust_cut_progress_windows.py
│       ├── extract_true_thickness_fullfield_odb.py
│       ├── extract_true_thickness_all.py
│       ├── extract_true_thickness_cut_milestones.py
│       ├── build_thickness_dataset.py
│       ├── build_actual_cut_progress_views.py
│       ├── export_thesis_english_figures.py
│       ├── export_v2_extra_figures.py
│       ├── extract_force_stress_metrics.py
│       ├── export_force_stress_figures.py
│       ├── build_thickness_seed_tables.py
│       ├── extract_true_thickness_odb.py
│       └── export_thesis_figures.py
└── legacy_chapter4_assets/
    └── analysis/
        └── build_thesis_assets.py
```

## 路径示意

| 数据类型 | 示例路径 |
|---|---|
| Abaqus 位移导出 Excel | `data/abaqus/displacement_exports/S1/01/S1-01-001.xlsx` |
| 原始 ODB | `data/abaqus/odb/S1.odb` 至 `data/abaqus/odb/S6.odb` |
| 升级后 ODB | `output/odb_upgraded/S1_upg.odb` 至 `output/odb_upgraded/S6_upg.odb` |
| 旧壁厚种子工作簿 | `data/abaqus/wall_thickness/legacy_thickness.xlsx` |
| 三维轮廓仪点云 | `data/profilometer/TXF1-1.csv` |
| 粗糙度批处理 manifest | `output/surface_roughness/surface_roughness_batch_manifest.csv` |
| 三坐标壁厚工作簿 | `data/experimental/cmm_wall_thickness.xlsx` |
| 实验照片 | `data/photos/*.png` |

`case_config.yml` 和 `case_config.json` 保存六个示例工况的曲率半径、名义壁厚、标准帧、测线位置、ODB 文件名和 S3 中段路径保留比例。Abaqus 自带 Python 若缺少 PyYAML，可读取同名 JSON 配置。

## 脚本说明

| 脚本 | 处理的数据 | 主要处理 | 输出 |
|---|---|---|---|
| `scripts/abaqus/common.py` | `case_config.yml/json`、位移导出目录结构 | 读取配置，解析工况、测线、帧号、名义半径和极角 | 被其他 Abaqus 脚本导入 |
| `scripts/abaqus/build_displacement_dataset.py` | S1 至 S6 位移导出 Excel，每张表含两侧表面位移 | 计算中面位移 `w_mid_mm`、两侧位移差 `delta_t_mm`、帧级位移能量，识别主响应窗口；S3 按 `path_keep_fraction` 裁剪中段有效路径 | `raw_displacement_metrics.csv.gz`、`aligned_displacement_long.csv.gz`、`aligned_displacement_grid.csv.gz`、`ks_ke_summary.csv`、`aligned_displacement.xlsx`、`qc_energy_windows.png` |
| `scripts/abaqus/upgrade_odb_files.py` | 示例原始 ODB | 调用 `abaqus upgrade`，生成升级副本，不改写原始文件 | `output/odb_upgraded/*_upg.odb` |
| `scripts/abaqus/extract_cut_progress_status.py` | 升级后 ODB 的 `STATUS` 场 | 提取逐帧失效单元数，识别实际切削开始、四分位进程和切削完成帧 | `cut_progress_all_frames.csv`、`cut_progress_milestones.csv/json` |
| `scripts/abaqus/adjust_cut_progress_windows.py` | STATUS 逐帧表、里程碑表、工况配置 | 对设置了 `path_keep_fraction` 的算例重写正式比较窗口，保留原始 STATUS 里程碑字段 | 修正后的 `cut_progress_all_frames.csv`、`cut_progress_milestones.csv/json` |
| `scripts/abaqus/extract_true_thickness_fullfield_odb.py` | 单个升级后 ODB、工况配置、目标帧 | 在初始构型下按极角和轴向位置配对内外表面节点，逐帧计算变形后的真壁厚 | `true_thickness_fullfield_<case>.csv.gz`、`true_thickness_frame_summary_<case>.csv`、`surface_pair_map_<case>.csv.gz`、提取日志 |
| `scripts/abaqus/extract_true_thickness_all.py` | 六个升级后 ODB、工况配置 | 批量调用全场真壁厚提取脚本 | 六个工况的全场真壁厚和帧级统计 |
| `scripts/abaqus/extract_true_thickness_cut_milestones.py` | 实际切削里程碑、升级后 ODB、工况配置 | 将标准帧与 cut_start/q1/q2/q3/cut_done 合并后补提真壁厚 | 覆盖或补全关键帧真壁厚 CSV |
| `scripts/abaqus/build_thickness_dataset.py` | 真壁厚全场表、帧级统计、位移窗口、STATUS 里程碑 | 合并六个工况，生成按实际切削进程对齐的壁厚表和切削完成帧厚度场 | `true_thickness_summary_all_cases.csv`、`aligned_true_thickness_summary.csv`、`true_thickness_end_field.csv.gz`、`true_thickness_key_metrics.json` |
| `scripts/abaqus/build_actual_cut_progress_views.py` | 位移指标、壁厚汇总、STATUS 逐帧表和里程碑 | 将能量、测线位移和真壁厚统一映射到实际切削进程 `eta` | `aligned_energy_by_cut_progress.csv`、`aligned_line_history_by_cut_progress.csv`、`aligned_true_thickness_by_cut_progress.csv` |
| `scripts/abaqus/export_thesis_english_figures.py` | 位移、切削进程、真壁厚、终止帧厚度场 | 导出第 3 章最终英文图，并写出算例与测线指标表 | `fig03_01` 至 `fig03_12` 图件、`thesis_case_metrics.csv/json`、`thesis_line_metrics.csv/json` |
| `scripts/abaqus/export_v2_extra_figures.py` | `thesis_case_metrics`、`thesis_line_metrics` | 导出峰值位移、厚度保持率、响应时序和测线梯度等补充判读图 | `output/figures/abaqus_v2_extra/*.png/pdf`、`v2_extra_summary.json` |
| `scripts/abaqus/extract_force_stress_metrics.py` | 升级后 ODB、实际切削里程碑、算例指标表 | 提取 RF1/RF2/RF3、等效反力、Mises 应力分位值和接触压力统计 | `force_history_by_cut_progress.csv`、`stress_contact_by_cut_progress.csv`、`force_stress_summary.csv/json` |
| `scripts/abaqus/export_force_stress_figures.py` | 反力、应力、接触压力 CSV | 绘制实际切削进程下的反力曲线、阶段均值、应力分位和接触压力图 | `fig03_15_force_response_combo.*`、`fig03_16_stress_contact_combo.*` |
| `scripts/abaqus/build_thickness_seed_tables.py` | 旧壁厚种子工作簿 | 将缺陷点、控制点、旧近似厚度公式和帧号整理成 CSV，用于早期点位式真壁厚核查 | `thickness_point_seeds.csv`、`legacy_formula_audit.csv` |
| `scripts/abaqus/extract_true_thickness_odb.py` | 单个 ODB、旧壁厚种子表、工况配置 | 按种子点附近的极角和轴向位置提取点位真壁厚，是早期点位式流程 | `surface_pair_map_<case>.csv.gz`、`true_thickness_profiles_<case>.csv.gz`、`true_thickness_points_<case>.csv` |
| `scripts/abaqus/export_thesis_figures.py` | 早期位移、种子点和点位真壁厚表 | 导出旧版 A1 至 A9 图件；用于追溯早期图件口径 | `figA1` 至 `figA9`、`captions.md` |
| `scripts/process_curved_surface_roughness.py` | 三维轮廓仪点云 CSV，字段为 `x,y,z[,intensity]` | 去除单曲率圆柱形貌和扫描倾斜平面，计算 Sa/Sq/Sp/Sv/Sz/Ssk/Sku，并可用 Gaussian 权重分离波纹面和粗糙度面 | 去形点云、Gaussian 粗糙度点云、`roughness_summary.csv/xlsx`、方法对比表、热力图、三维表面图 |
| `scripts/summarize_surface_roughness_batch.py` | 单个加工面的粗糙度输出目录和批处理 manifest | 汇总每个加工面的 Gaussian 粗糙度指标，检查文件完整性、半径、样件编号、滤波参数和网格规模 | `batch_roughness_results.csv`、`batch_validation_report.csv/txt` |
| `legacy_chapter4_assets/analysis/build_thesis_assets.py` | 三坐标壁厚工作簿、批量粗糙度结果、Gaussian 粗糙度点云、实验照片 | 生成第 4 章壁厚误差、重复性、粗糙度、壁厚误差与 Sa 对应关系、轮廓高度图和照片压缩件 | 第 4 章图件、`thickness_measurements_long.csv`、壁厚统计表、粗糙度统计表、`thesis_asset_manifest.json` |

## 运行顺序

普通 Python 环境：

```bash
python -m pip install -r requirements.txt
python scripts/abaqus/build_displacement_dataset.py
```

Abaqus ODB 相关脚本需要在 Abaqus 命令可用的环境中运行：

```bash
python scripts/abaqus/upgrade_odb_files.py --input-dir data/abaqus/odb --abaqus-command abaqus
abaqus python scripts/abaqus/extract_cut_progress_status.py
python scripts/abaqus/adjust_cut_progress_windows.py
python scripts/abaqus/extract_true_thickness_all.py --abaqus-command abaqus
python scripts/abaqus/extract_true_thickness_cut_milestones.py --abaqus-command abaqus
python scripts/abaqus/build_thickness_dataset.py
python scripts/abaqus/build_actual_cut_progress_views.py
python scripts/abaqus/export_thesis_english_figures.py
abaqus python scripts/abaqus/extract_force_stress_metrics.py
python scripts/abaqus/export_force_stress_figures.py
python scripts/abaqus/export_v2_extra_figures.py
```

第 4 章实验数据链：

```bash
python scripts/process_curved_surface_roughness.py data/profilometer/TXF1-1.csv --sample S1 --output-dir output/surface_roughness/TXF1-1
python scripts/summarize_surface_roughness_batch.py
python legacy_chapter4_assets/analysis/build_thesis_assets.py
```

## 其它说明

本目录只保留脚本、配置模板和处理口径说明。真实论文文档、原始测量数据、商业软件工程文件、求解结果和个人过程文件不在开源包内。若要复现实验数值，需要按上方示意路径提供等价数据。

