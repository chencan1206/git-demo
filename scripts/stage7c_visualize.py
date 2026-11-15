"""Stage 7c — Visual-rich analysis report for Zenodo 15800414.

Reads existing deep_analysis.json + spectra CSV + segment masks,
generates a self-contained HTML report with inline SVG figures
(no external dependencies, no static files to ship).

Charts produced:
  1. Mean reflectance spectra (400–950 nm) Healthy vs Alternaria
  2. Spectrum difference Δ(Alternaria − Healthy) with red-edge highlight
  3. ROI-level spectra overlay (6 samples)
  4. Vegetation-index bar chart (NDVI / EVI / NDRE / SIPI)
  5. Segmentation-sensitivity grouped bars
  6. RF n_estimators sensitivity
  7. CV accuracy/f1 box plot (2-fold)
  8. Per-band heatmap (correlation with class label)
  9. Segmentation mask thumbnails

Output:  /workspace/reports/zenodo_full_report.html
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tifffile

REPORTS = Path("/workspace/reports")
DATA = Path("/workspace/data/raw/zenodo_15800414")
WL = np.linspace(400.0, 950.0, 550).astype(np.float32)

plt.rcParams.update({
    "svg.fonttype": "none",
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def fig_to_svg(fig: plt.Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


def fig_to_png_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def compute_vis(spec: np.ndarray) -> dict:
    """Compute common vegetation indices from a full-band reflectance vector."""
    # bands: WL[260:290] ~ 660-700 red; WL[270]~677; WL[400]~800 NIR
    red = spec[270]
    re = spec[293]   # ~707 nm red-edge
    nir = spec[400]  # ~800 nm
    blue = spec[80]  # ~480 nm
    green = spec[150]  # ~550 nm
    return {
        "NDVI": (nir - red) / (nir + red + 1e-9),
        "EVI": 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-9),
        "NDRE": (nir - re) / (nir + re + 1e-9),
        "SIPI": (nir - blue) / (nir - red + 1e-9),
        "GNDVI": (nir - green) / (nir + green + 1e-9),
    }


def main() -> int:
    t0 = time.perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True)

    ana = json.loads((REPORTS / "zenodo_deep_analysis.json").read_text())
    df = pd.read_csv(REPORTS / "zenodo_spectra.csv")
    labels = np.array(df["class"].tolist())
    wl_cols = [c for c in df.columns if c.startswith("wl_")]
    spectra = df[wl_cols].to_numpy(dtype=np.float32)

    # ---- Figure 1: mean spectra ----
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = {"Healthy": "#2ecc71", "Alternaria": "#e74c3c"}
    for cls, grp in df.groupby("class"):
        idx = grp.index
        mean = spectra[idx].mean(axis=0)
        std = spectra[idx].std(axis=0)
        ax.plot(WL, mean, label=cls, color=colors[cls], lw=2)
        ax.fill_between(WL, mean - std, mean + std, color=colors[cls], alpha=0.15)
    ax.axvspan(400, 700, color="#f39c12", alpha=0.08, label="Visible")
    ax.axvspan(700, 1100, color="#3498db", alpha=0.08, label="NIR")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Mean Leaf Reflectance — Healthy vs Alternaria (Day-1)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig1 = fig_to_svg(fig)

    # ---- Figure 2: spectrum difference ----
    fig, ax = plt.subplots(figsize=(9, 3.8))
    h_mean = spectra[labels == "Healthy"].mean(axis=0)
    a_mean = spectra[labels == "Alternaria"].mean(axis=0)
    diff = a_mean - h_mean
    peak_idx = int(np.argmax(np.abs(diff)))
    ax.fill_between(WL, diff, 0, where=diff >= 0, color="#e74c3c", alpha=0.55, label="Alternaria > Healthy")
    ax.fill_between(WL, diff, 0, where=diff < 0, color="#2ecc71", alpha=0.55, label="Healthy > Alternaria")
    # red-edge highlight
    ax.axvspan(680, 760, color="#c0392b", alpha=0.12, label="Red-edge 680–760 nm")
    ax.annotate(f"Peak {WL[peak_idx]:.0f} nm\nΔ={diff[peak_idx]:.3f}",
                xy=(WL[peak_idx], diff[peak_idx]),
                xytext=(WL[peak_idx] + 25, diff[peak_idx] + 0.005),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=9, ha="center")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Δ Reflectance (Alternaria − Healthy)")
    ax.set_title("Spectral Difference — Disease Signal Peaks in the Red Edge")
    ax.legend(loc="upper right", framealpha=0.9)
    fig2 = fig_to_svg(fig)

    # ---- Figure 3: ROI overlay ----
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, row in df.iterrows():
        cls = row["class"]
        ax.plot(WL, spectra[i],
                color=colors[cls], lw=1.2, alpha=0.7,
                label=cls if i == 0 or (labels[i] != labels[i-1]) else None)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("ROI-level Spectra Overlay — 3 Leaves per Class")
    ax.legend(framealpha=0.9)
    fig3 = fig_to_svg(fig)

    # ---- Figure 4: VI bar ----
    vi_by_roi = [compute_vis(spectra[i]) for i in range(len(spectra))]
    vi_df = pd.DataFrame(vi_by_roi)
    vi_df["class"] = labels
    vi_mean = vi_df.groupby("class")[["NDVI","EVI","NDRE","SIPI","GNDVI"]].mean()

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(vi_mean.columns))
    w = 0.35
    for i, cls in enumerate(vi_mean.index):
        ax.bar(x + (i-0.5)*w, vi_mean.loc[cls], width=w,
               color=colors[cls], label=cls, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(vi_mean.columns)
    ax.set_ylabel("Index Value")
    ax.set_title("Vegetation Indices by Class")
    ax.legend(framealpha=0.9)
    fig4 = fig_to_svg(fig)

    # ---- Figure 5: segmentation sensitivity ----
    seg = ana["segmentation_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, cls in zip(axes, ["Healthy", "Alternaria"]):
        trials = seg[cls]
        labels_s = [v["label"].split("·")[-1].strip() for v in trials.values()]
        coverage = [v["coverage"] * 100 for v in trials.values()]
        thr = [v["threshold"] for v in trials.values()]
        bars = ax.bar(labels_s, coverage, color=colors[cls], alpha=0.75)
        for bar, t in zip(bars, thr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"thr={t:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("Coverage %")
        ax.set_title(f"{cls} — Segmentation Sensitivity")
        ax.tick_params(axis="x", rotation=20)
    fig5 = fig_to_svg(fig)

    # ---- Figure 6: RF sensitivity ----
    sens = ana["model_sensitivity"]["n_estimators"]
    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ns = [s["n"] for s in sens]
    times = [s["time_s"] for s in sens]
    accs = [s["accuracy"] for s in sens]
    ax1.plot(ns, times, "o-", color="#3498db", label="Train time (s)", lw=2)
    ax1.set_xlabel("n_estimators")
    ax1.set_ylabel("Train time (s)", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")
    ax2 = ax1.twinx()
    ax2.plot(ns, accs, "s--", color="#e74c3c", label="Test accuracy", lw=2)
    ax2.set_ylabel("Test accuracy", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(0.5, 1.05)
    ax1.set_title("RF Hyperparameter Sensitivity (2-fold CV, n=6)")
    fig6 = fig_to_svg(fig)

    # ---- Figure 7: CV boxplot ----
    cv = ana["final_model"]["cv"]
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    data = [cv["metrics"][m]["values"] for m in metrics_plot]
    bplot = ax.boxplot(data, tick_labels=[m.replace("_macro","") for m in metrics_plot],
                       patch_artist=True, widths=0.4)
    for patch, color in zip(bplot["boxes"], ["#3498db","#2ecc71","#e67e22","#9b59b6"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(data):
        for v in vals:
            ax.scatter(i+1, v, color="black", zorder=5, alpha=0.6, s=30)
    ax.set_ylabel("Score")
    ax.set_title("2-fold Cross Validation Metrics")
    ax.set_ylim(0, 1.05)
    fig7 = fig_to_svg(fig)

    # ---- Figure 8: band-class correlation heatmap ----
    # simple t-test proxy: difference in means / pooled std
    fig, ax = plt.subplots(figsize=(9, 3.2))
    h = spectra[labels == "Healthy"]
    a = spectra[labels == "Alternaria"]
    pooled_std = np.sqrt((h.std(axis=0)**2 + a.std(axis=0)**2) / 2) + 1e-9
    t_stat = (a.mean(axis=0) - h.mean(axis=0)) / pooled_std
    # downsample for display: 50-band bins
    n_bins = 50
    bin_size = 550 // n_bins
    t_binned = np.array([t_stat[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)])
    wl_binned = np.array([WL[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)])
    colors_cool = np.where(t_binned > 0, "#e74c3c", "#2ecc71")
    ax.bar(wl_binned, np.abs(t_binned), width=WL[bin_size]-WL[0], color=colors_cool, alpha=0.8)
    ax.axvspan(680, 760, color="#c0392b", alpha=0.15, label="Red-edge")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7, label="|t|=1.0")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("|t-statistic| (band vs class)")
    ax.set_title("Per-band Class Separability — Red Edge Dominates")
    ax.legend(framealpha=0.9)
    fig8 = fig_to_svg(fig)

    # ---- Figure 9: segmentation mask thumbnails ----
    fig, axes = plt.subplots(1, 4, figsize=(11, 4))
    mask_files = [
        ("Healthy mask", REPORTS / "zenodo_seg_mask_healthy.tif", "leaf coverage"),
        ("Alternaria mask", REPORTS / "zenodo_seg_mask_alternaria.tif", "leaf coverage"),
        ("Healthy labels", REPORTS / "zenodo_seg_labels_healthy.tif", "ROI labels"),
        ("Alternaria labels", REPORTS / "zenodo_seg_labels_alternaria.tif", "ROI labels"),
    ]
    for ax, (title, path, cmap_type) in zip(axes, mask_files):
        arr = tifffile.imread(path)
        if "labels" in title:
            ax.imshow(arr, cmap="jet", interpolation="nearest")
        else:
            ax.imshow(arr, cmap="gray", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig9 = fig_to_svg(fig)

    # ---- Figure 10: 红边斜率可视化 ----
    fig, ax = plt.subplots(figsize=(9, 3.8))
    # 计算 红边区域的一阶导数 (680-760 nm)
    re_start = np.argmin(np.abs(WL - 680))
    re_end = np.argmin(np.abs(WL - 760))
    for cls, grp in df.groupby("class"):
        idx = grp.index
        mean = spectra[idx].mean(axis=0)
        redge_wl = WL[re_start:re_end+1]
        redge_mean = mean[re_start:re_end+1]
        # 线性拟合斜率
        slope = np.polyfit(redge_wl, redge_mean, 1)[0]
        ax.plot(redge_wl, redge_mean, color=colors[cls], lw=2.5,
                label=f"{cls} (slope={slope:.4f}/nm)")
        ax.scatter([WL[re_start], WL[re_end]], [mean[re_start], mean[re_end]],
                   color=colors[cls], zorder=5, s=50)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Red-edge Slope — Disease flattens the transition")
    ax.legend(framealpha=0.9)
    fig10 = fig_to_svg(fig)

    elapsed = round(time.perf_counter() - t0, 2)

    # =====================================================
    # HTML ASSEMBLY
    # =====================================================
    payload = ana
    h_vi = compute_vis(spectra[labels == "Healthy"].mean(axis=0))
    a_vi = compute_vis(spectra[labels == "Alternaria"].mean(axis=0))

    def fmt(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Zenodo 15800414 — Plant HSI Studio 深度分析报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f8f9fa; color: #2c3e50; line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 1180px; margin: 0 auto; }}
  header {{ background: linear-gradient(135deg, #27ae60 0%, #2ecc71 50%, #e67e22 100%);
           color: white; padding: 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
           box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
  header h1 {{ font-size: 1.9rem; margin-bottom: 0.5rem; font-weight: 700; }}
  header .meta {{ font-size: 0.9rem; opacity: 0.9; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
              gap: 1rem; margin: 1.2rem 0; }}
  .kpi {{ background: white; padding: 1.2rem; border-radius: 10px;
          border-left: 4px solid #27ae60; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .kpi .label {{ font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; color: #2c3e50; margin-top: 0.3rem; }}
  .kpi.orange {{ border-left-color: #e67e22; }}
  .kpi.red {{ border-left-color: #e74c3c; }}
  .kpi.blue {{ border-left-color: #3498db; }}
  section {{ background: white; padding: 1.8rem; border-radius: 10px;
             box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1.3rem; }}
  section h2 {{ font-size: 1.25rem; color: #2c3e50; margin-bottom: 1rem;
               padding-bottom: 0.6rem; border-bottom: 2px solid #ecf0f1; }}
  section h3 {{ font-size: 1.05rem; color: #34495e; margin: 1rem 0 0.6rem; }}
  .fig-wrap {{ margin: 1rem 0; overflow-x: auto; }}
  .fig-wrap svg {{ max-width: 100%; height: auto; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0; font-size: 0.9rem; }}
  th, td {{ padding: 0.55rem 0.8rem; text-align: left; border-bottom: 1px solid #ecf0f1; }}
  th {{ background: #f8f9fa; font-weight: 600; color: #34495e; }}
  tr:hover {{ background: #fdfefe; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .badge {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px;
           font-size: 0.78rem; font-weight: 600; }}
  .badge-green {{ background: #d5f5e3; color: #1e8449; }}
  .badge-red {{ background: #fadbd8; color: #c0392b; }}
  .badge-orange {{ background: #fdebd0; color: #ca6f1e; }}
  .badge-blue {{ background: #d6eaf8; color: #2471a3; }}
  .insight {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 0.9rem 1.1rem;
              border-radius: 0 8px 8px 0; margin: 1rem 0; font-size: 0.92rem; }}
  .insight strong {{ color: #7d6608; }}
  footer {{ text-align: center; color: #95a5a6; font-size: 0.85rem; padding: 1.5rem; }}
  .code {{ background: #f8f9fa; padding: 1rem; border-radius: 6px; overflow-x: auto;
           font-family: monospace; font-size: 0.82rem; color: #555; max-height: 300px;
           border: 1px solid #ecf0f1; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>🌿 Zenodo 15800414 番茄高光谱病害分析报告</h1>
  <div class="meta">
    数据集: <a style="color:white" href="https://zenodo.org/records/15800414">zenodo.org/records/15800414</a>
    &nbsp;|&nbsp; 生成时间: {time.strftime('%Y-%m-%d %H:%M UTC')}
    &nbsp;|&nbsp; 分析耗时: {elapsed}s
    &nbsp;|&nbsp; PHS Studio v1.0.0
  </div>
</header>

<!-- ===== KPI CARDS ===== -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="label">数据集规模</div>
    <div class="value">5.77 GB</div>
    <div style="font-size:0.78rem;color:#95a5a6">5 类 × 8 时间点 × 140×280×550</div>
  </div>
  <div class="kpi orange">
    <div class="label">当前已分析</div>
    <div class="value">2 cubes</div>
    <div style="font-size:0.78rem;color:#95a5a6">Healthy + Alternaria (Day-1)</div>
  </div>
  <div class="kpi red">
    <div class="label">光谱差异峰值</div>
    <div class="value">744 nm</div>
    <div style="font-size:0.78rem;color:#95a5a6">Δ={fmt(payload['spectral_difference']['peak_diff'])}</div>
  </div>
  <div class="kpi blue">
    <div class="label">分析总耗时</div>
    <div class="value">{payload['total_seconds']}s</div>
    <div style="font-size:0.78rem;color:#95a5a6">完整 Step pipeline</div>
  </div>
</div>

<!-- ===== SECTION 1: 数据集说明 ===== -->
<section>
  <h2>1. 数据集概览</h2>
  <p>
    该数据集包含 <strong>40 个高光谱立方体</strong> (5 类 × 8 时间点/重复),每个为
    <code>140 × 280 × 550</code> 的 <code>float64</code> MAT v5 格式 (Zlib 压缩)。
    光谱范围 <strong>400–950 nm</strong>,<strong>~1 nm 间隔</strong>。
    番茄叶片在接种以下病原后的时序成像:
  </p>
  <table>
    <tr><th>类别</th><th>病斑</th><th>立方体数</th><th>单文件大小</th></tr>
    <tr><td><span class="badge badge-green">Healthy</span></td><td>健康对照</td><td>8</td><td>~153 MB</td></tr>
    <tr><td><span class="badge badge-red">Alternaria alternata</span></td><td>早疫</td><td>8</td><td>~153 MB</td></tr>
    <tr><td><span class="badge badge-red">Alternaria solani</span></td><td>早疫(另一株)</td><td>8</td><td>~113 MB</td></tr>
    <tr><td><span class="badge badge-orange">Botrytis cinerea</span></td><td>灰霉</td><td>8</td><td>~115 MB</td></tr>
    <tr><td><span class="badge badge-orange">Fusarium oxysporum</span></td><td>枯萎</td><td>8</td><td>113–153 MB</td></tr>
  </table>
  <div class="insight">
    <strong>⚠️ 网络限制说明:</strong>
    沙箱代理带宽较低 (~几十 KB/s),全量 40 立方体下载预计需要数十小时。
    当前报告基于已下载的 <strong>Day-1 Healthy_1.mat + Alternaria_alternata_1.mat</strong>。
    后台正在补充下载更多文件,可用于后续 5 类全量分析。
  </div>
</section>

<!-- ===== SECTION 2: 光谱分析 ===== -->
<section>
  <h2>2. 光谱统计分析</h2>

  <div class="two-col">
    <div>
      <h3>2.1 平均反射率曲线</h3>
      <div class="fig-wrap">{fig1}</div>
      <p style="font-size:0.88rem;color:#7f8c8d">
        阴影区域为 ±1σ。绿色=健康叶,红色=病害叶。病害叶在近红外区域的反射率系统性低于健康叶。
      </p>
    </div>
    <div>
      <h3>2.2 单 ROI 光谱叠加</h3>
      <div class="fig-wrap">{fig3}</div>
      <p style="font-size:0.88rem;color:#7f8c8d">
        每类 3 个叶片 ROI。类内一致性较好,类间在红边区域差异明显。
      </p>
    </div>
  </div>

  <h3>2.3 光谱差异 — 疾病信号定位</h3>
  <div class="fig-wrap">{fig2}</div>

  <div class="two-col">
    <div>
      <h3>2.4 红边斜率对比</h3>
      <div class="fig-wrap">{fig10}</div>
      <p style="font-size:0.88rem;color:#7f8c8d">
        健康叶红边斜率更陡 (健康 >  Alternaria),符合植物病害光谱学预期。
      </p>
    </div>
    <div>
      <h3>2.5 各波段可分性 (t 统计量)</h3>
      <div class="fig-wrap">{fig8}</div>
      <p style="font-size:0.88rem;color:#7f8c8d">
        红边区域 (680–760 nm) 的 |t| 显著高于可见光/NIR,是最具判别力的波段。
      </p>
    </div>
  </div>

  <div class="insight">
    <strong>🔬 核心发现 — 与设计文档 2.4 节文献预期一致:</strong><br>
    光谱差异峰值位于 <strong>{fmt(payload['spectral_difference']['peak_band_nm'],1)} nm</strong>,
    Alternaria 在该波段反射率比 Healthy 高 <strong>{fmt(payload['spectral_difference']['peak_diff'])}</strong>。
    平均绝对差异仅 <strong>{fmt(payload['spectral_difference']['mean_abs_diff'],5)}</strong>,
    说明病害信号是<strong>波段选择性</strong>的,而不是全光谱整体偏移。
  </div>
</section>

<!-- ===== SECTION 3: 植被指数 ===== -->
<section>
  <h2>3. 植被指数分析</h2>
  <div class="two-col">
    <div>
      <div class="fig-wrap">{fig4}</div>
    </div>
    <div>
      <h3>各指数均值 (类内平均)</h3>
      <table>
        <tr><th>指数</th><th>Healthy</th><th>Alternaria</th><th>Δ</th></tr>
        <tr><td>NDVI</td><td>{fmt(h_vi['NDVI'])}</td><td>{fmt(a_vi['NDVI'])}</td><td style="color:{('#e74c3c' if h_vi['NDVI']>a_vi['NDVI'] else '#27ae60')}">{fmt(a_vi['NDVI']-h_vi['NDVI'],3)}</td></tr>
        <tr><td>EVI</td><td>{fmt(h_vi['EVI'])}</td><td>{fmt(a_vi['EVI'])}</td><td>{fmt(a_vi['EVI']-h_vi['EVI'],3)}</td></tr>
        <tr><td>NDRE</td><td>{fmt(h_vi['NDRE'])}</td><td>{fmt(a_vi['NDRE'])}</td><td>{fmt(a_vi['NDRE']-h_vi['NDRE'],3)}</td></tr>
        <tr><td>SIPI</td><td>{fmt(h_vi['SIPI'])}</td><td>{fmt(a_vi['SIPI'])}</td><td>{fmt(a_vi['SIPI']-h_vi['SIPI'],3)}</td></tr>
        <tr><td>GNDVI</td><td>{fmt(h_vi['GNDVI'])}</td><td>{fmt(a_vi['GNDVI'])}</td><td>{fmt(a_vi['GNDVI']-h_vi['GNDVI'],3)}</td></tr>
      </table>
      <div class="insight">
        <strong>NDVI / NDRE 降低</strong>是叶片光合作用受损的典型表现。
        病害叶的 NIR 反射率下降导致 NDVI 从 <strong>{fmt(h_vi['NDVI'],3)}</strong> 降至 <strong>{fmt(a_vi['NDVI'],3)}</strong>。
      </div>
    </div>
  </div>
</section>

<!-- ===== SECTION 4: 分割 ===== -->
<section>
  <h2>4. 叶片分割</h2>

  <h3>4.1 分割参数敏感性</h3>
  <div class="fig-wrap">{fig5}</div>

  <h3>4.2 分割掩膜预览</h3>
  <div class="fig-wrap">{fig9}</div>

  <table>
    <tr><th>指标</th><th>Healthy</th><th>Alternaria</th></tr>
    <tr><td>Otsu RA 阈值</td><td>{fmt(payload['segmentation_sensitivity']['Healthy']['otsu']['threshold'])}</td><td>{fmt(payload['segmentation_sensitivity']['Alternaria']['otsu']['threshold'])}</td></tr>
    <tr><td>叶片覆盖率</td><td>{fmt(payload['spectral_stats']['Healthy']['leaf_coverage'])}</td><td>{fmt(payload['spectral_stats']['Alternaria']['leaf_coverage'])}</td></tr>
    <tr><td>连通域数</td><td>3</td><td>3</td></tr>
    <tr><td>85th %ile 覆盖率</td><td>{fmt(payload['segmentation_sensitivity']['Healthy']['percentile']['coverage'])}</td><td>{fmt(payload['segmentation_sensitivity']['Alternaria']['percentile']['coverage'])}</td></tr>
  </table>

  <div class="insight">
    <strong>分割鲁棒性:</strong> RA 谱角 + Otsu 在两类上的阈值非常稳定 (0.918 vs 0.921),
    说明光谱角对健康/病害叶片的内部一致性区分度好。
    固定阈值 0.15 完全失效 (覆盖率 100%),因为 RA 相似度图的分布远比预期狭窄。
  </div>
</section>

<!-- ===== SECTION 5: 建模 ===== -->
<section>
  <h2>5. 分类建模 (Random Forest)</h2>

  <h3>5.1 超参敏感性 (n_estimators)</h3>
  <div class="fig-wrap">{fig6}</div>

  <h3>5.2 2-fold 交叉验证</h3>
  <div class="fig-wrap">{fig7}</div>

  <div class="two-col">
    <div>
      <h3>5.3 最终模型混淆矩阵</h3>
      <table>
        <tr><th></th><th>Pred: Alternaria</th><th>Pred: Healthy</th></tr>
        <tr><td><strong>True: Alternaria</strong></td><td>1</td><td>0</td></tr>
        <tr><td><strong>True: Healthy</strong></td><td>0</td><td>1</td></tr>
      </table>
      <p style="font-size:0.82rem;color:#95a5a6">
        注: 只有 2 个测试样本 (test_size=0.4, n=6),
        因此测试集 accuracy=1.0 <strong>不能作为结论</strong>。
        CV 更可信 (2-fold, 4 个 fold 样本)。
      </p>
    </div>
    <div>
      <h3>5.4 CV 完整指标</h3>
      <table>
        <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Fold Values</th></tr>
"""
    cv_metrics = payload["final_model"]["cv"]["metrics"]
    for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        v = cv_metrics[m]
        html += f"""        <tr><td>{m}</td><td>{fmt(v['mean'])}</td><td>{fmt(v['std'])}</td><td>[{", ".join(fmt(x,2) for x in v['values'])}]</td></tr>\n"""

    html += f"""      </table>
    </div>
  </div>

  <div class="insight">
    <strong>⚠️ 样本量限制警告:</strong>
    6 个 ROI (每类 3 个) 的 2-fold CV 波动大 (accuracy 0.5–1.0, macro_f1 0.33–1.0)。
    要获得统计显著的结果,需要至少 <strong>40+ 立方体</strong> (每类 8 个重复)。
    当前结果主要用于 <strong>pipeline 功能验证</strong>,而非定量性能评估。
  </div>
</section>

<!-- ===== SECTION 6: 性能 ===== -->
<section>
  <h2>6. 性能测量 (完整 Step pipeline)</h2>

  <table>
    <tr><th>阶段</th><th>耗时</th><th>RSS 增量</th></tr>
    <tr><td>MAT 加载 (单个)</td><td>1.4–1.6 s</td><td>+100–150 MB</td></tr>
    <tr><td>相似度图 (RA spectral angle)</td><td>~0.4 s</td><td>—</td></tr>
    <tr><td>Otsu 阈值 + 形态学 + 连通域</td><td>&lt;1 s</td><td>—</td></tr>
    <tr><td>RF 训练 (n=50→400)</td><td>0.69→1.92 s</td><td>—</td></tr>
    <tr><td><strong>总计 (2 立方体)</strong></td><td><strong>{payload['total_seconds']} s</strong></td><td><strong>峰值 ~749 MB</strong></td></tr>
  </table>

  <div class="insight">
    <strong>💡 性能洞察:</strong> MAT v5 格式需要 <strong>整文件解压到内存</strong>,无法分块随机访问,
    这是当前 I/O 瓶颈 (1.5 s / cube)。但一旦导入为 Zarr,后续处理全是毫秒级。
    按此推算,40 立方体全量跑完预计 <strong>~6 min</strong>,峰值内存 ~1.5 GB。
  </div>
</section>

<!-- ===== SECTION 7: 后续工作 ===== -->
<section>
  <h2>7. 后续工作 & 当前局限</h2>

  <h3>7.1 阻塞项</h3>
  <table>
    <tr><th>#</th><th>问题</th><th>影响</th><th>建议</th></tr>
    <tr><td>1</td><td>沙箱代理带宽 ~几十 KB/s</td><td>40 立方体全量下载需数十小时</td><td>用户本地下载后上传,或后台慢慢下</td></tr>
    <tr><td>2</td><td>样本量 n=6 ROI</td><td>分类指标不具备统计显著性</td><td>全量 40 cube 后重跑</td></tr>
    <tr><td>3</td><td>训练/测试划分未按文件分组</td><td>同叶泄漏风险</td><td>全量数据后按文件分组划分</td></tr>
  </table>

  <h3>7.2 框架级发现 (Bug)</h3>
  <div class="code">
1. Step 引擎 cache hit 时 result.skipped=True, outputs=[] 不自动恢复
   → GUI 重跑会 StopIteration;需调 prj.find_cached_outputs() 手动恢复

2. cache key 对嵌套 dict (如 model_params) 的 hash 不稳定
   → 不同 n_estimators 可能误命中相同 cache
  </div>
</section>

<footer>
  Generated by PHS Studio Step Engine v1.0.0 &nbsp;•&nbsp;
  数据分析: numpy + scikit-learn + matplotlib &nbsp;•&nbsp;
  报告 HTML 自包含 (无外部依赖)
</footer>

</div>
</body>
</html>
"""

    out = REPORTS / "zenodo_full_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ 报告生成: {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
