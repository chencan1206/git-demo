# PHS Studio 用户手册（v1.0）

本手册面向首次安装和使用 PHS Studio 的科研用户。软件定位、数据格式与算法实现见
[README](../README.md)；已知限制见 [KNOWN_LIMITATIONS](../KNOWN_LIMITATIONS.md)。

---

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux x86_64（已验证）、macOS / Windows 理论可用 |
| Python | 3.11 – 3.14 |
| 内存 | 常规影像 ≥ 8 GB；10 GB 级立方体导入建议 ≥ 16 GB |
| 显示 | 桌面环境；无显示器时可用 `QT_QPA_PLATFORM=offscreen` 做冒烟测试 |
| 可选 | NVIDIA GPU + CUDA 版 PyTorch（深度学习训练更快） |

---

## 2. 安装

### 2.1 从源码一键安装（推荐）

```bash
cd plant-hsi-studio
./scripts/install.sh --with-qt          # 包含 Qt 系统库（Debian/Ubuntu）
# 需要深度学习时：
./scripts/install.sh --dl
```

脚本会创建 `.venv/`、安装全部运行时依赖，并注册 `phs` 命令。可重复执行。

### 2.2 手动安装

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 \
  libfontconfig1 libxcb-cursor0 libxcb-xinerama0
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .                        # 运行时依赖；加 .[dl] 安装 PyTorch
```

### 2.3 从 wheel 安装（离线分发）

```bash
pip install plant_hsi_studio-1.0.0-py3-none-any.whl
```

> wheel 的默认依赖已覆盖 GUI、导入、传统建模与报告；只有 PyTorch 需要额外
> `pip install torch`（或 `pip install "plant_hsi_studio-...whl[dl]"`）。

### 2.4 验证

```bash
python -c "import plant_hsi; print(plant_hsi.__version__)"   # 1.0.0
QT_QPA_PLATFORM=offscreen phs                               # 无显示器冒烟启动
```

---

## 3. 启动与界面

```bash
source .venv/bin/activate
phs
```

界面由四部分组成：

- **菜单栏**：File / Data / Segment / Model / Help。
- **左侧资源树**：数据集、掩膜、光谱表、模型、报告等 artifact，双击打开。
- **中间标签页**：2D 假彩色影像、分类图、掩膜；与像元光谱联动。
- **右侧停靠面板**：分割、光谱预处理、建模与报告、深度学习（DC²Net）。
- **状态栏 / 日志**：每次运行的状态、耗时与 `run_xxxx` 编号。

菜单速查：

| 菜单 | 动作 |
|---|---|
| File | New Project、Open Project、Save、Quit |
| Data | Import RAW Dataset、Open in View |
| Segment | Save Mask、Extract ROI、Export |
| Model | Train、Build HTML Report、Open Report |
| Help | About（显示版本） |

---

## 4. 项目与可追溯性

一个项目就是一个目录：

```text
my-project/
├─ project.json     项目元信息（名称、创建时间、schema_version）
├─ project.db       SQLite（WAL）：运行记录与产物索引
├─ artifacts/       Zarr 立方体、掩膜、光谱表、模型、报告
├─ runs/            每次运行的输出与日志
├─ datasets/        派生数据
└─ logs/            运行日志
```

- 每次“运行”都会写入一条 `RunRecord`：Step 名、参数快照、随机种子、输入/输出哈希、耗时、错误。
- 相同 Step + 参数 + 输入 + 种子会命中缓存，直接复用既有产物。
- 所有产物带内容 hash，可在资源树中查看依赖关系（`parents`）。
- 断电或强杀进程后重新打开项目会执行 SQLite 完整性检查；`project.json` 采用原子写。

---

## 5. 数据导入

**Data → Import RAW Dataset…**，支持：

| 格式 | 说明 |
|---|---|
| `.mat` v5 | 自动探测第一个三维数值变量；整文件读入内存 |
| `.mat` v7.3 | 通过 h5py 读取 |
| ENVI | `.hdr` + `.raw/.bil/.bsq`，可分块流式读 |
| GeoTIFF | `.tif/.tiff`，未压缩时可内存映射分块读 |
| `.npz` | 单个三维数组 |

导入时注意：

1. **波长**：源文件没有波长坐标时必须确认。Zenodo 番茄数据为 400–950 nm、550 波段、1 nm 间隔。
2. **维度**：约定 `(行, 列, 波段)`；GeoTIFF 的波段轴会自动判断。
3. **内存**：ENVI / 未压缩 TIFF 按行分块写入 Zarr，峰值内存与块大小相关，与文件总量无关；MAT v5 因压缩无法随机访问，仍需整文件载入。
4. **去重**：重复导入同一文件会按内容 hash 识别。

---

## 6. 分割与 ROI

右侧 **分割 / Segmentation** 面板：

1. 选择指数：
   - **RA**：光谱角（Spectral Angle），对整体亮度不敏感；
   - **DE**：光谱信息散度（SID）。
2. 选择参考光谱：来自框选区域、已有 ROI 或手工光谱。
3. 选择阈值算法：Otsu、ISODATA、K-means、GMM、Li、MinError；可叠加形态学开/闭运算。
4. 预览满意后运行，生成 `MASK`；再用连通域提取多个目标（设置最小面积、连通性、排序）。
5. **Segment → Extract ROI** 生成每个目标的平均光谱表（样本 × 波段），并保留面积、质心等元数据。

> 对弱对比数据，建议先用叶片区域的平均光谱作参考，再对光谱角图做 Otsu 分割。

---

## 7. 光谱预处理

**光谱预处理 / Preprocessing** 面板支持链式处理，顺序即执行顺序：

- SNV、MSC、一阶/二阶导数、Savitzky–Golay（窗口、多项式阶数、导数阶）、
  FOD（分数阶微分，截断 G-L）、均值中心、标准化、Min-Max、矢量归一化、小波阈值。

每条链可保存为模板；训练模型时链快照会写入模型卡，保证复现。

---

## 8. 建模与报告

**建模与报告 / Modeling** 面板：

1. 选择任务：分类或回归。
2. 选择模型：
   - 分类：RF、SVM、LDA、KNN、BPNN；
   - 回归：RFR、PLSR、SVR、LR、LightGBM。
3. 设置划分比例、交叉验证折数、随机种子与超参数。
4. 训练后查看指标（Acc/P/R/F1 或 R²/MAE/RMSE）、混淆矩阵或散点图。
5. **Model → Build HTML Report** 生成单文件报告，内含项目信息、参数、指标、图与运行清单。

> 样本量过小时指标波动很大。分类任务建议每类 ≥ 20 个独立 ROI，并按叶片/日期分组划分。

---

## 9. 深度学习（DC²Net）

**深度学习 / DC²Net** 面板：

1. 绑定一个 CUBE，设置 patch 大小、步长、训练/验证/测试比例。
2. 划分时按叶片或日期分组，避免同叶泄漏；随机种子会记录。
3. 选择 DC²Net 预设，设置轮数、批大小、学习率、早停；支持暂停/续训。
4. 训练完成后可滑窗推理生成分类图，或计算通道注意力/波长重要性。
5. 可导出远程训练包，在带 GPU 的机器上训练后回传权重。

CPU 训练仅适合小数据冒烟；正式训练建议 CUDA 环境。

---

## 10. 推理与导出

- **推理**：选择模型与新影像，输出逐像元分类图/置信度图，或 ROI 预测表。
- **导出**：
  - `export.csv`：光谱表 / 预测表（UTF-8-BOM，Excel 可直接打开）；
  - `export.geotiff`：掩膜、分类图；
  - 报告：`report.html` + `run_manifest.json`；
  - 模型：`.pkl` / `.pt` + `model_card.json`。

所有导出都登记在 `runs/` 下，可在资源树中定位到对应运行记录。

---

## 11. 演示项目

没有真实数据时，可生成一个合成演示项目：

```bash
source .venv/bin/activate
python scripts/make_demo_project.py --out /workspace/phs_demo --force
```

生成内容：

- `/workspace/phs_demo/demo-project`：4 个立方体、8 个 ROI、2 类；
- `/workspace/phs_demo/demo_report.html`：含模型指标的报告。

用 **File → Open Project** 打开 `demo-project` 即可体验完整流程。

---

## 12. 复现与质量检查

```bash
ruff check . && ruff format --check .
python -m mypy src
QT_QPA_PLATFORM=offscreen python -m pytest -o addopts="" -q
```

- 10 GB 级导入基准：

  ```bash
  python scripts/large_import_bench.py --shape 10000 10000 25 --out /tmp/phs_bench
  ```

- 真实数据端到端验收（Zenodo 番茄数据）：

  ```bash
  PYTHONPATH=src python scripts/stage7_e2e_zenodo.py
  ```

---

## 13. 故障排查

| 现象 | 处理 |
|---|---|
| `qt.qpa.plugin: could not load the Qt platform plugin "xcb"` | 安装第 2 节的 Qt 系统库；或设 `QT_QPA_PLATFORM=offscreen` |
| `libxcb-cursor.so.0: cannot open shared object file` | `sudo apt-get install libxcb-cursor0` |
| 启动即退出，无窗口 | 检查 `DISPLAY`；无显示器时属正常 headless 行为 |
| 导入大 MAT 时内存暴涨 | MAT v5 为压缩格式，需整文件解压；改用 ENVI/TIFF 或分块数据 |
| 训练报“类别样本过少” | 增加 ROI 数量，或减小交叉验证折数 |
| 报告里没有混淆矩阵 | 分类任务且测试集需同时包含各类别 |
| `ModuleNotFoundError: zarr / sklearn / tifffile` | 重新安装运行时依赖：`pip install -e .` |
| 中文显示为方框 | 安装中文字体（如 `fonts-noto-cjk`）后重启 |

---

## 14. 已知限制

完整列表见 [KNOWN_LIMITATIONS](../KNOWN_LIMITATIONS.md)。v1.0 的主要限制：

- 单用户、单写者；同一项目不要同时被两个进程打开。
- 不支持撤销/重做；误操作可重新运行 Step 覆盖。
- 项目打包格式 `.phsproj` 尚未实现。
- 报告仅 HTML，未提供 PDF。
- MAT v5 导入与 DC²Net 全量训练受内存限制。
- GPU 需手动安装 CUDA 版 PyTorch，软件不做自动选择。
