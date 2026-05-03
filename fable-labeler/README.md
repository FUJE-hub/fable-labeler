# Fable Labeler

A local, offline, lightweight image annotation tool with built-in **color verification**, **anomaly detection**, and **3D RGB point cloud visualization**.

> A community contribution project by a freshman CS student, built using tokens from the **Xiaomi MiMo Orbit — 百万亿Token创造者激励计划**.

[English](#features) | [中文](#中文)

## Background

Existing annotation tools — LabelImg, LabelMe, CVAT, among others — excel at bounding box creation, but provide no mechanism to evaluate annotation quality. After annotating hundreds of images, practitioners are left with unanswered questions:

- Are my annotations consistent across the dataset?
- Did I accidentally draw a box on the wrong region?
- What color distribution do my bounding boxes actually contain?

Fable Labeler addresses this gap by combining standard annotation capabilities with **built-in quality verification** — enabling users to detect errors, outliers, and inconsistencies before the dataset enters the training pipeline.

## Who Is This For?

Fable Labeler is designed for **small teams and individual researchers** who need to annotate image datasets and care about data quality. It is particularly suited for:

- **Computer vision researchers** — Building custom datasets for object detection or classification, and want to catch annotation errors before training
- **Data quality engineers** — Auditing annotation consistency across a dataset before it enters the model training pipeline
- **Small lab teams (2–5 people)** — Requiring more than a basic drawing tool, but without the operational overhead of a full platform like CVAT
- **Offline / air-gapped environments** — Where installing Docker or deploying web services is not feasible
- **Students and educators** — Learning data annotation pipelines and understanding what constitutes high-quality training data
- **Domain-specific datasets** — Medical imaging, satellite remote sensing, industrial inspection — where color anomalies in annotations often indicate real labeling errors

**Example workflow**: A dataset of 200 images is annotated with bounding boxes. Before training a YOLO model, the user runs RGB verification and anomaly detection. The tool reports: "Annotation #47 on image 13 has a color distribution that deviates significantly from other annotations sharing the same label — the bounding box may have been drawn on the wrong region." After correction, the dataset is exported to YOLO format for training.

### What RGB Detection Can and Cannot Do

Fable Labeler's verification relies on **color distribution analysis** (RGB + HSV). This is a statistical heuristic, not a semantic correctness check. Understanding its boundaries is essential to using it effectively.

**Well-suited scenarios:**

- **Uniform-color objects** — Stop signs (red), traffic lights (red/green/yellow), vegetation (green), sky (blue). Color is a strong discriminator, and mismatches reliably indicate errors.
- **Background mislabeling detection** — If a bbox accidentally covers background instead of the target object, the color profile will almost always differ from correct annotations of the same label.
- **Cross-annotation consistency auditing** — When you have 50 annotations of "car" and 1 of them has a completely different color distribution, that outlier is likely an error.
- **Domain-specific uniform datasets** — Medical imaging (tissue color), industrial inspection (surface defects), satellite imagery (land cover types) — where color correlates strongly with the category.

**Not well-suited scenarios:**

- **Visually diverse categories** — "Person", "dog", "cat" can have vastly different colors (skin tone, fur pattern). RGB outlier detection will produce false positives because the natural variance within the category is high.
- **Lighting-sensitive datasets** — The same object photographed under different illumination (day/night, indoor/outdoor) will have different RGB distributions. The tool may flag legitimate annotations as anomalous.
- **Grayscale or low-saturation images** — When the image is predominantly gray, color information is minimal and the feature vector loses discriminative power.
- **Precise boundary evaluation** — RGB detection can tell you if a bbox is on the wrong *region*, but cannot tell you if the bbox boundary is 5 pixels off from the optimal edge.
- **Shape, texture, or semantic correctness** — A green bbox on green grass and a green bbox on a green car will look identical to the color analyzer. Context and shape are beyond its scope.

**In summary**: RGB verification is best used as a **pre-training sanity check filter** — it catches obvious errors cheaply and quickly, but it does not replace human review or model-based validation for precise annotation quality.

## Development Process

### Phase 1: Core Annotation (MVP)

The first goal was a working annotation tool with the minimum viable feature set:

1. **Tkinter canvas** — Chose Tkinter over PyQt for zero-dependency deployment. Built a custom `CanvasWidget` class handling mouse events for draw, drag, resize, and select.

2. **Data model** — Designed `Annotation` (bbox, label, attributes) and `Project` (image_annotations dict, labels list, save/load JSON) as pure Python classes with no UI dependency.

3. **Undo/Redo** — Implemented dict-based snapshots instead of deep-copying objects. Each action pushes a serialized snapshot to a 50-step stack. This was chosen over command pattern for simplicity.

4. **Dark theme** — Built a centralized `THEME` dict in `utils.py` with factory functions `make_button()` and `make_label()` to ensure consistent styling across all UI components.

### Phase 2: Color Verification (Differentiation)

This is what makes Fable Labeler unique — the ability to inspect what's actually inside your bounding boxes:

1. **RGB extraction** — `extract_bbox_pixels()` crops the bbox region from the PIL image and returns a flat numpy array of RGB values. All coordinates are relative (0-1 normalized) to support images of any resolution.

2. **HSV dual-space analysis** — Pure numpy vectorized conversion from RGB to HSV. No OpenCV dependency. The key insight is that HSV separates color (hue) from brightness (value), making it easier to detect anomalies like a "green" annotation that's actually mostly dark pixels.

3. **Dominant color extraction** — Quantizes pixels into HSV bins and counts with `Counter`. Returns top 5 colors with hex codes and ratios.

4. **Anomaly detection** — The core innovation. A 12-dimensional feature vector (6 RGB stats + 6 HSV stats) is computed per annotation, then three independent algorithms vote:

   - **Z-score**: flags annotations whose features deviate more than σ from the group mean
   - **IQR**: uses interquartile range for robust outlier detection (less sensitive to extreme values)
   - **Mahalanobis distance**: accounts for correlations between features (e.g., high red usually means low blue)

   An annotation is flagged as anomalous only if **2 out of 3 algorithms agree** — this reduces false positives significantly compared to any single method.

5. **Cross-image consistency** — After verifying all annotations for a label, the tool aggregates scores across images. If "cat" annotations have consistent colors in 9 images but look completely different in 1 image, that image is flagged.

6. **Threshold calibration** — A dialog with sliders lets users adjust Z-score, IQR, and Mahalanobis thresholds. The key design decision was making thresholds **configurable per-dataset**, because what's anomalous for medical images is different from what's anomalous for street photos.

### Phase 3: 3D Point Cloud Visualization

To help users intuitively understand color distributions:

1. **RGB space mapping** — Each pixel maps to a 3D point where R=x, G=y, B=z (0-255 range). The point is colored by its actual RGB value, so you literally see the color space.

2. **Sampling** — For large bounding boxes, uniform sampling reduces point count to a configurable limit (default 3000) while preserving the overall distribution shape.

3. **Export** — Point cloud data can be exported to CSV (for Excel/R) or JSON (for further processing).

### Phase 4: Quality Hardening

After the features worked, the focus shifted to reliability:

1. **PIL memory management** — Identified 5 points where PIL images could leak memory (load, switch, cache, consume, exception). Added `safe_close_pil()` wrapper with `try/finally` patterns at each point.

2. **Incremental status tracking** — Replaced O(n) full-scan status updates with O(1) delta tracking. `_labeled_count` and `_total_anns_count` are updated incrementally when switching images, not recalculated from scratch.

3. **LRU preload cache** — Adjacent images (±2 range) are preloaded in background. Cache is bounded to 6 entries with LRU eviction to prevent memory explosion.

4. **Auto-save** — A 30-second timer checks a dirty flag and saves only if changes occurred. This avoids the data loss risk of manual-only save.

5. **Operation logger** — Tracks 13 event types (session, image open/leave, annotation CRUD, verify, export, undo/redo) for audit and debugging.

6. **Unit tests** — 58 tests covering all models/ modules. No external test framework dependency — uses Python's built-in `unittest`.

7. **Type annotations** — Full type coverage across all models/ files with `from __future__ import annotations` for forward compatibility.

### Phase 5: Polish & Optimization

Final round of improvements:

1. **HSV DRY refactoring** — Extracted shared `_compute_hsv_from_rgb()` helper used by both vectorized HSV conversion and dominant color extraction.

2. **Canvas drag optimization** — Added `_cached_scale` / `_cached_photo` to avoid recomputing PhotoImage during drag operations. Redraw only happens on drag end.

3. **matplotlib consolidation** — Moved `matplotlib.use("TkAgg")` to `main.py` entry point, removed duplicate calls from panel modules.

4. **Sub-function extraction** — Split `_update_status()` into `_sync_labeled_count()`, `_refresh_listbox_color()`, and `_update_sidebar_color()` for better readability and testability.

5. **Bug fixes** — Fixed `_sync_labeled_count` logic error and PIL image memory leak in `load_image_at`.

## Features

### Annotation

- Bounding box creation, selection, move, resize, delete
- Multi-label management with sidebar panel
- Undo / Redo (50-step stack)
- Click-through: click on empty area to deselect, click on box to select
- Label switching via sidebar click

### Color Verification (Unique)

- Per-annotation RGB color statistics (mean, std, histogram)
- HSV dual-space analysis (hue, saturation, value)
- Dominant color extraction (top 5, hex + ratio)
- Interactive color histogram (R/G/B channels)
- 12-dim feature vector for anomaly detection

### Anomaly Detection (Unique)

- **Triple-algorithm voting**: Z-score + IQR + Mahalanobis distance
- Configurable thresholds via calibration dialog (sliders)
- Cross-image consistency analysis across entire dataset
- Severity classification: HIGH / MEDIUM / NONE
- Batch correction suggestions

### 3D RGB Point Cloud (Unique)

- 3D scatter plot (R=x, G=y, B=z) colored by actual RGB values
- Per-annotation point cloud extraction
- Configurable sampling size for performance
- Export to CSV / JSON

### Export Formats

- COCO JSON
- Pascal VOC XML
- YOLO TXT + classes.txt
- Custom JSON / CSV

### Other

- Auto-save every 30 seconds
- Operation log tracking (13 event types)
- LRU preload cache (adjacent images)
- Dark theme UI
- Keyboard shortcuts for all common operations

## Requirements

- Python 3.9+
- Pillow >= 9.0
- numpy >= 1.21
- matplotlib >= 3.5

Tkinter is included with Python by default on most systems.

## Installation

```bash
git clone https://github.com/FUJE-hub/fable-labeler.git
cd fable-labeler
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. Click **Open Folder** and select a directory containing images
2. Images are listed in the left sidebar
3. Click an image to load it
4. Draw bounding boxes on the canvas
5. Enter label name in the input field and press Enter or click Add
6. Use **RGB Verify** to check annotation color quality
7. Export annotations via the toolbar

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+O` | Open image directory |
| `Ctrl+S` | Save project |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Delete selected annotation |
| `Escape` | Deselect / cancel drawing |
| `Left` / `Right` | Previous / next image |
| `[` / `]` | Jump to previous / next unlabeled image |
| `Home` | Reset zoom and pan to default |
| `1` – `9` | Quick-assign label to selected annotation |
| Mouse wheel | Zoom in / out |
| Right-click drag | Pan image |

## Project Structure

```
fable-labeler/
├── main.py                     # Entry point
├── utils.py                    # Theme, fonts, factories, utilities
├── models/
│   ├── annotation.py           # Annotation + Project data model
│   ├── color_extractor.py      # RGB/HSV extraction + anomaly detection
│   ├── config.py               # Runtime configuration
│   ├── exporter.py             # COCO / VOC / YOLO exporters
│   ├── logger.py               # Operation logger
│   └── point_cloud.py          # 3D RGB point cloud generation
├── ui/
│   ├── canvas_widget.py        # Annotation canvas (draw/drag/zoom)
│   ├── main_window.py          # Main window and orchestration
│   ├── pointcloud_panel.py     # Point cloud panel
│   ├── rgb_panel.py            # RGB verification panel
│   └── sidebar.py              # Image list + label manager
└── tests/
    └── test_core.py            # Unit tests (58 tests)
```

## Running Tests

```bash
python -m unittest tests.test_core -v
```

## How It Works

### Annotation Flow

```
Open folder → Select image → Draw bbox → Add label → Repeat
                                                     ↓
                                          Save (auto every 30s)
```

### Verification Flow

```
Select annotation → RGB Verify → Color stats computed
                                      ↓
                              Anomaly detection (triple-algorithm)
                                      ↓
                              Threshold calibration (optional)
                                      ↓
                              Cross-image consistency analysis
```

### Data Storage

All user data is stored in the selected image directory:

| File | Content |
|------|---------|
| `.labeler_meta.json` | Annotations, labels, version |
| `.labeler_config.json` | Threshold configuration |
| `.labeler_logs/` | Operation logs (JSON) |

No data is sent to any external server. Everything stays local.

## Why Fable Labeler?

Existing tools (LabelImg, LabelMe, CVAT) focus on **annotation creation**. Fable Labeler focuses on **annotation quality**:

- Are annotations consistent across images?
- Are there outlier annotations that deviate from the expected distribution?
- What do the color distributions inside bounding boxes actually look like?

If you care about data quality, not just data quantity, this tool is for you.

### Technical Contributions

1. **Triple-algorithm anomaly voting** — Combines Z-score, IQR, and Mahalanobis distance with a majority-vote consensus. This multi-algorithm approach reduces false positives compared to single-method detection.

2. **12-dim RGB+HSV feature space** — By combining both color spaces into a unified feature vector, the detection captures anomalies that RGB-only or HSV-only methods would miss.

3. **Cross-image consistency analysis** — Extends anomaly detection from single-image to dataset-level, catching systematic annotation errors that per-image analysis cannot.

4. **Threshold calibration dialog** — Acknowledges that "anomaly" is dataset-dependent. Users can tune sensitivity without modifying code.

5. **3D RGB point cloud** — Provides intuitive visual understanding of color distributions in annotation regions, bridging the gap between numerical statistics and human perception.

## About

This project was built using tokens from the **Xiaomi MiMo Orbit — 百万亿Token创造者激励计划**. As a freshman CS student, I received tokens from the program and wanted to give back to the community by building something useful. The tool works, but has not been exhaustively tested across all edge cases. Issues and PRs are welcome.

## License

[MIT License](LICENSE)

---

# 中文

一个本地、离线、轻量的图像标注工具，内置**颜色验证**、**异常检测**和 **3D RGB 点云可视化**。

> 计算机科学与技术专业大一学生开发，使用 **Xiaomi MiMo Orbit — 百万亿Token创造者激励计划** 获得的 Token 构建，作为社区回馈项目。

## 项目背景

现有的标注工具（LabelImg、LabelMe、CVAT）只管画框，不管标注质量。在实际使用中，我们发现以下问题无法回答：

- 标注在整个数据集中是否一致？
- 有没有不小心画错区域？
- 标注框里的颜色分布是什么样的？

Fable Labeler 就是为了解决这个问题——不只是画框，还能帮你**验证标注质量**。

## 适用场景

Fable Labeler 面向**小型团队和个人研究者**，特别适合以下场景：

- **计算机视觉研究者** — 构建自定义目标检测或分类数据集，训练前希望排查标注错误
- **数据质量工程师** — 在数据集进入模型训练流程之前，审计标注的一致性与准确性
- **小型实验室团队（2–5 人）** — 需要超越基础画框工具的能力，但不需要 CVAT 等重型平台的部署与运维成本
- **离线 / 封闭网络环境** — 无法安装 Docker 或部署 Web 服务，需要纯本地运行的标注工具
- **学生与教育者** — 学习数据标注流程，理解高质量训练数据的构成标准
- **领域专用数据集** — 医学影像、卫星遥感、工业质检等，标注框内的颜色异常往往意味着真实的标注错误

**典型工作流**：对一个包含 200 张图片的数据集完成框选标注后，在训练 YOLO 模型之前，运行 RGB 验证和异常检测。工具会报告："第 13 张图的第 47 个标注，其颜色分布与同标签的其他标注存在显著偏差——可能框选了错误的区域。" 修正后导出 YOLO 格式，以更干净的数据投入训练。

### RGB 检测能做什么、不能做什么

Fable Labeler 的验证依赖**颜色分布分析**（RGB + HSV）。这是一种统计启发式方法，而非语义正确性判断。理解它的边界是有效使用它的前提。

**适合的场景：**

- **颜色单一的目标物体** — 停车标志（红色）、交通信号灯（红/绿/黄）、植被（绿色）、天空（蓝色）。颜色是强判别特征，颜色不匹配通常能可靠指示错误。
- **误标背景检测** — 标注框意外覆盖了背景而非目标物体时，颜色特征几乎总是与同标签的正确标注不同。
- **标注间一致性审查** — 当 50 个 "car" 标注中有 1 个颜色分布截然不同时，这个离群点很可能是错误标注。
- **领域专用的均匀数据集** — 医学影像（组织颜色）、工业质检（表面缺陷）、卫星遥感（地物覆盖类型）——颜色与类别强相关。

**不适合的场景：**

- **视觉差异大的类别** — "人"、"狗"、"猫"的颜色变化极大（肤色、毛色、花纹）。RGB 离群检测会产生大量误报，因为类别内的自然方差本身就很高。
- **光照敏感的数据集** — 同一物体在不同光照条件下（白天/夜晚、室内/室外）的 RGB 分布完全不同。工具可能将合法标注误判为异常。
- **灰度或低饱和度图像** — 当图像以灰色为主时，颜色信息极少，特征向量失去区分能力。
- **精确边界评估** — RGB 检测能判断标注框是否在错误的**区域**，但无法判断边界是否偏离最优边缘 5 个像素。
- **形状、纹理或语义正确性** — 绿色草地上的绿色框和绿色汽车上的绿色框，在颜色分析器看来完全相同。上下文和形状超出其分析范围。

**总结**：RGB 验证最适合作为**训练前的预检过滤器**——它能廉价、快速地捕获明显错误，但不能替代人工审查或基于模型的精确质量验证。

## 功能特性

### 标注功能

- 框选标注：创建、选择、拖动、缩放、删除
- 多标签管理，侧边栏面板
- 撤销 / 重做（50 步栈）
- 点击空白取消选择，点击标注框选中
- 侧边栏点击切换标签

### 颜色验证（独有）

- 每个标注框内的 RGB 颜色统计（均值、标准差、直方图）
- HSV 双空间分析（色相、饱和度、明度）
- 主色提取（前 5 种颜色，含十六进制色码和占比）
- 交互式颜色直方图（R/G/B 三通道）
- 12 维特征向量用于异常检测

### 异常检测（独有）

- **三算法投票**：Z-score + IQR + 马氏距离
- 可调阈值校准对话框（滑块调节）
- 跨图一致性分析（全数据集级别）
- 严重程度分级：ANOMALY / SUSPICIOUS / NORMAL
- 批量纠错建议

### 3D RGB 点云（独有）

- 3D 散点图（R=x, G=y, B=z），点的颜色即实际 RGB 值
- 每个标注区域独立点云提取
- 可配置采样大小，平衡精度与性能
- 支持导出 CSV / JSON

### 导出格式

- COCO JSON
- Pascal VOC XML
- YOLO TXT + classes.txt
- 自定义 JSON / CSV

### 其他

- 每 30 秒自动保存
- 操作日志追踪（13 种事件类型）
- LRU 预加载缓存（相邻图片）
- 深色主题界面
- 快捷键支持

## 环境要求

- Python 3.9+
- Pillow >= 9.0
- numpy >= 1.21
- matplotlib >= 3.5

Tkinter 随 Python 自带，无需额外安装。

## 安装

```bash
git clone https://github.com/FUJE-hub/fable-labeler.git
cd fable-labeler
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py
```

1. 点击 **打开文件夹**，选择包含图片的目录
2. 图片列表显示在左侧边栏
3. 点击图片加载到画布
4. 在画布上绘制标注框
5. 在输入框输入标签名称，按回车或点击添加
6. 使用 **RGB 验证** 检查标注框内的颜色质量
7. 通过工具栏导出标注结果

## 快捷键

| 按键 | 功能 |
|------|------|
| `Ctrl+O` | 打开图片目录 |
| `Ctrl+S` | 保存项目 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+Y` | 重做 |
| `Delete` | 删除选中标注 |
| `Escape` | 取消选择 / 取消绘制 |
| `左` / `右` | 上一张 / 下一张图片 |
| `[` / `]` | 跳转到上一张 / 下一张未标注图片 |
| `Home` | 重置缩放和平移到默认状态 |
| `1` – `9` | 快速为选中标注分配标签 |
| 鼠标滚轮 | 放大 / 缩小 |
| 右键拖动 | 平移图像 |

## 项目结构

```
fable-labeler/
├── main.py                     # 入口
├── utils.py                    # 主题、字体、工厂函数、工具
├── models/
│   ├── annotation.py           # 标注 + 项目数据模型
│   ├── color_extractor.py      # RGB/HSV 提取 + 异常检测
│   ├── config.py               # 运行时配置
│   ├── exporter.py             # COCO / VOC / YOLO 导出
│   ├── logger.py               # 操作日志
│   └── point_cloud.py          # 3D RGB 点云生成
├── ui/
│   ├── canvas_widget.py        # 标注画布（绘制/拖动/缩放）
│   ├── main_window.py          # 主窗口与调度
│   ├── pointcloud_panel.py     # 点云面板
│   ├── rgb_panel.py            # RGB 验证面板
│   └── sidebar.py              # 图片列表 + 标签管理
└── tests/
    └── test_core.py            # 单元测试（58 个测试）
```

## 运行测试

```bash
python -m unittest tests.test_core -v
```

## 工作原理

### 标注流程

```
打开文件夹 → 选择图片 → 绘制标注框 → 添加标签 → 重复
                                               ↓
                                    自动保存（每 30 秒）
```

### 验证流程

```
选中标注 → RGB 验证 → 计算颜色统计
                           ↓
                    异常检测（三算法投票）
                           ↓
                    阈值校准（可选）
                           ↓
                    跨图一致性分析
```

### 数据存储

所有用户数据存储在所选图片目录内：

| 文件 | 内容 |
|------|------|
| `.labeler_meta.json` | 标注数据、标签列表、版本号 |
| `.labeler_config.json` | 阈值配置 |
| `.labeler_logs/` | 操作日志（JSON） |

不上传任何数据到外部服务器，完全本地运行。

## 为什么做 Fable Labeler？

现有工具（LabelImg、LabelMe、CVAT）专注于**绘制标注框**。Fable Labeler 专注于**标注质量验证**：

- 标注在不同图片之间是否一致？
- 是否存在偏离整体分布的异常标注？
- 标注框内的颜色分布特征是什么？

如果关注的不仅是数据数量，更是数据质量，这个工具会适合你。

### 技术贡献

1. **三算法异常投票** — 组合 Z-score、IQR 和马氏距离，通过多数投票达成共识，相比单一算法显著降低误报率
2. **12 维 RGB+HSV 特征空间** — 将两个色彩空间融合为统一特征向量，捕捉单一空间检测不到的异常
3. **跨图一致性分析** — 将异常检测从单图扩展到数据集级别，发现逐图分析无法捕获的系统性标注错误
4. **可调阈值校准** — 承认"异常"因数据集而异，用户可通过滑块自行调节灵敏度，无需修改代码
5. **3D RGB 点云** — 直观展示标注区域的颜色分布，弥合数值统计与人类感知之间的认知鸿沟

## 关于

本项目使用 **Xiaomi MiMo Orbit — 百万亿Token创造者激励计划** 获得的 Token 构建。作者为计算机科学与技术专业大一学生，获得 Token 后希望做一些有用的东西回馈社区。工具可用，但尚未经过所有边界条件的全面测试。欢迎 Issue 和 PR。

## 协议

[MIT License](LICENSE)
