# PlantHSI Studio (PHS Studio)

面向植物 / 作物高光谱影像的桌面向导式分析软件。以“项目（Project）”为顶层容器，
所有导入、分割、光谱提取、预处理、建模、深度学习、可解释性、导出与报告都记录为可追溯的
`RunRecord`，产物以内容 hash 复用。

- 两条文献流程：PlantSpecLab（ROI 光谱 + 传统机器学习）与 DC²Net（3D 图块 + 深度网络）
- 数据格式：`.mat`（v5 / v7.3）、ENVI（`.hdr/.raw/.bil/.bsq`）、GeoTIFF（`.tif/.tiff`）
- 工程存储：Zarr v2（立方体）+ SQLite/WAL（运行与产物元数据）+ JSON（项目元信息）
- 界面：PySide6 桌面应用，支持 2D 假彩色、像元光谱联动、分割 / 预处理 / 建模 / DC²Net 面板

---

## 1. 干净机器安装

以下步骤在 Debian/Ubuntu 上验证（Linux x86_64）。Python 需 3.11–3.14。

```bash
# 1) 系统库（Qt 运行时；无桌面环境也需要这些 .so）
sudo apt-get update
sudo apt-get install -y \
  libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libfontconfig1 libxcb-cursor0 libxcb-xinerama0

# 2) 获取代码并创建虚拟环境
cd /workspace/plant-hsi-studio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 3) 安装（含 IO / 传统 ML / PyTorch / 开发工具）
pip install -e ".[io,ml,dl,dev]"
```

> 不需要深度学习时可用 `pip install -e ".[io,ml]"`；`dl` 额外会装 CPU 版之外的
> PyTorch 需要按官方源安装，本仓库只在 CPU 上做过验收。

### 验证安装

```bash
source .venv/bin/activate
python -c "import plant_hsi, PySide6, zarr, sklearn; print('ok')"

# 全量单元测试 + UI 冒烟（无显示器）
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:warnings
```

### 启动

```bash
phs                     # 或 python -m plant_hsi
```

无显示器 / CI 环境请设置 `QT_QPA_PLATFORM=offscreen`，程序会进入 headless 模式并正常退出。

---

## 2. 第一次使用（冒烟路径）

1. **File → New Project…** 选择空目录，填写项目名。
2. **Data → Import RAW Dataset…** 选择 `.mat` / `.hdr` / `.tif`。缺少波长时，导入对话框
   会要求确认（Zenodo 番茄数据为 400–950 nm、550 波段、1 nm 间隔）。
3. 左侧资源树双击 CUBE，打开假彩色视图；右侧“分割 / Segmentation”面板可设置 RA/DE
   指数、阈值算法（Otsu/ISODATA/K-means/GMM/Li/MinError）、形态学与连通域，运行后保存掩膜。
4. “提取 ROI”生成 ROI 均值谱，联动到“光谱预处理”和“建模与报告”面板。
5. 预处理链（SNV/MSC/FOD/SG/导数/归一化等）→ 选择模型与交叉验证 → 训练 → 查看指标与
   混淆矩阵 / 散点图。
6. **Model → Build HTML Report** 生成项目报告；模型、指标、报告均登记在 `runs/` 与
   `artifacts/` 下，可在资源树中追溯。
7. 深度学习：绑定 CUBE 后构建图块集、训练 DC²Net、滑窗推理、波长重要性。

---

## 3. 复现与质量门槛

```bash
cd /workspace/plant-hsi-studio
ruff check . && ruff format --check .
python -m mypy src
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:warnings
```

- 大立方体导入基准（默认 10 GB 逻辑尺寸，峰值内存约 120 MB）：

  ```bash
  python scripts/large_import_bench.py --shape 10000 10000 25 --out /tmp/phs_bench
  ```

- 关键界面截图：

  ```bash
  python scripts/stage6_capture.py     # 输出到 /workspace/reports/stage6/
  ```

- Stage 6 一键验收（持久化 / 内存 / 全量测试 / 文档与截图）：

  ```bash
  python scripts/stage6_acceptance.py
  ```

---

## 4. 目录结构

```text
plant-hsi-studio/
├─ src/plant_hsi/
│  ├─ core/          Project、SQLite/WAL 元数据、Step 引擎、Artifact
│  ├─ io/            MAT / ENVI / GeoTIFF 读取，分块写 Zarr
│  ├─ preprocess/    预处理链（SNV/MSC/FOD/SG/导数/归一化）
│  ├─ segment/       RA/DE 指数、阈值、形态学、连通域
│  ├─ features/      ROI 光谱、图块样本集
│  ├─ modeling/      8 个传统模型、交叉验证、指标、模型卡
│  ├─ dl/            DC²Net、训练器、滑窗推理
│  ├─ explain/       波长重要性
│  ├─ predict/       逐像元推理
│  ├─ report/        HTML 报告 + run manifest
│  ├─ export/        GeoTIFF / CSV
│  └─ ui/            主窗口与各分析面板
├─ tests/unit/       pytest 单元测试与 UI 冒烟
├─ scripts/          大文件基准、截图、Stage 6 验收
└─ docs/             设计文档（位于 /workspace/docs/design）
```

## 5. 已知限制

见 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。

## 6. 许可

Apache-2.0。
