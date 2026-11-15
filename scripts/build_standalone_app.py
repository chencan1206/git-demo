"""Build a single self-contained, click-to-open HTML app.

Inputs
  reports/zenodo_full_report.html      (analysis report, already has inline SVG)
  reports/zenodo_spectra.csv           (ROI spectra)
  reports/zenodo_seg_mask_*.tif        (segmentation masks)
  reports/zenodo_seg_labels_*.tif      (ROI label maps)
  plant-hsi-studio/dist/*.whl          (installable desktop app)

Output
  /workspace/PHS-Studio-Zenodo.html    (one file, works offline, no server needed)

The report is isolated inside an <iframe srcdoc> so its styles cannot leak,
and every data product is embedded as base64 with a real download button.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path("/workspace")
REPORTS = ROOT / "reports"
PROJECT = ROOT / "plant-hsi-studio"
OUT = ROOT / "PHS-Studio-Zenodo.html"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def asset_row(name: str, path: Path, mime: str, desc: str, size: str) -> str:
    data = b64(path)
    return f"""
    <div class="card">
      <div class="card-title">{html.escape(name)}</div>
      <div class="card-desc">{html.escape(desc)}</div>
      <div class="card-foot">
        <span class="size">{size}</span>
        <button class="btn small" data-b64="{data}" data-name="{html.escape(path.name)}"
                data-mime="{mime}">下载</button>
      </div>
    </div>"""


def main() -> int:
    report_html = (REPORTS / "zenodo_full_report.html").read_text(encoding="utf-8")
    ana = json.loads((REPORTS / "zenodo_deep_analysis.json").read_text(encoding="utf-8"))

    srcdoc = html.escape(report_html, quote=True)

    wheel = next(PROJECT.glob("dist/*.whl"))
    wheel_b64 = b64(wheel)

    assets = "".join([
        asset_row("光谱数据 · CSV", REPORTS / "zenodo_spectra.csv", "text/csv",
                  "6 个 ROI 的 550 波段反射率 + 类别/时间点元数据，Excel 可直接打开。",
                  "33 KB"),
        asset_row("分割掩膜 · Healthy TIFF", REPORTS / "zenodo_seg_mask_healthy.tif",
                  "image/tiff", "Otsu(RA 谱角) 二值叶片掩膜，uint8，140×280。", "39 KB"),
        asset_row("分割掩膜 · Alternaria TIFF", REPORTS / "zenodo_seg_mask_alternaria.tif",
                  "image/tiff", "Otsu(RA 谱角) 二值叶片掩膜，uint8，140×280。", "39 KB"),
        asset_row("ROI 标签 · Healthy TIFF", REPORTS / "zenodo_seg_labels_healthy.tif",
                  "image/tiff", "连通域编号标签图，uint16，可用于逐叶统计。", "153 KB"),
        asset_row("ROI 标签 · Alternaria TIFF", REPORTS / "zenodo_seg_labels_alternaria.tif",
                  "image/tiff", "连通域编号标签图，uint16，可用于逐叶统计。", "153 KB"),
        asset_row("完整测量 JSON", REPORTS / "zenodo_deep_analysis.json", "application/json",
                  "全部计时、RSS、光谱统计、分割/模型敏感性原始数值。", "69 KB"),
    ])

    sd = ana["spectral_difference"]
    total_s = ana["total_seconds"]

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PHS Studio · Zenodo 番茄高光谱分析</title>
<style>
  :root {{
    --green: #27ae60; --green-l: #2ecc71; --orange: #e67e22;
    --ink: #1f2d3a; --muted: #7f8c8d; --line: #e6eaee;
    --bg: #f5f7f9; --card: #ffffff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; }}
  body {{
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--ink); display: flex; overflow: hidden;
  }}

  /* ---------- sidebar ---------- */
  aside {{
    width: 246px; flex: 0 0 246px; height: 100vh; background: #10202b;
    color: #cfd8dc; display: flex; flex-direction: column; padding: 1.1rem 0.9rem;
  }}
  .brand {{ display: flex; align-items: center; gap: 0.6rem; padding: 0 0.5rem 1.1rem; }}
  .brand .dot {{
    width: 30px; height: 30px; border-radius: 9px;
    background: linear-gradient(135deg, var(--green-l), var(--orange));
    display: grid; place-items: center; font-size: 15px;
  }}
  .brand b {{ color: #fff; font-size: 0.98rem; letter-spacing: 0.3px; }}
  .brand small {{ display: block; color: #90a4ae; font-size: 0.7rem; font-weight: 400; }}
  nav {{ display: flex; flex-direction: column; gap: 0.25rem; margin-top: 0.4rem; }}
  nav button {{
    display: flex; align-items: center; gap: 0.65rem; width: 100%;
    padding: 0.66rem 0.75rem; border: 0; border-radius: 8px; cursor: pointer;
    background: transparent; color: #b0bec5; font-size: 0.88rem; text-align: left;
    font-family: inherit; transition: background .15s, color .15s;
  }}
  nav button:hover {{ background: #172c39; color: #fff; }}
  nav button.active {{ background: #1d3947; color: #fff; box-shadow: inset 3px 0 0 var(--green-l); }}
  nav button .ico {{ width: 18px; text-align: center; opacity: .9; }}
  .side-foot {{
    margin-top: auto; padding: 0.8rem 0.6rem 0; border-top: 1px solid #1c3441;
    font-size: 0.7rem; color: #78909c; line-height: 1.6;
  }}

  /* ---------- main ---------- */
  main {{ flex: 1; height: 100vh; overflow-y: auto; padding: 1.6rem 2rem 2.5rem; }}
  .page {{ display: none; animation: fade .22s ease; }}
  .page.active {{ display: block; }}
  @keyframes fade {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}

  h1 {{ font-size: 1.45rem; margin-bottom: 0.35rem; }}
  .sub {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1.4rem; }}

  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
          gap: 0.9rem; margin-bottom: 1.4rem; }}
  .kpi {{ background: var(--card); border-radius: 10px; padding: 1rem 1.1rem;
          border-left: 4px solid var(--green); box-shadow: 0 1px 4px rgba(16,32,43,.07); }}
  .kpi.o {{ border-left-color: var(--orange); }}
  .kpi.r {{ border-left-color: #e74c3c; }}
  .kpi.b {{ border-left-color: #3498db; }}
  .kpi .k {{ font-size: 0.72rem; letter-spacing: .5px; color: var(--muted); text-transform: uppercase; }}
  .kpi .v {{ font-size: 1.5rem; font-weight: 700; margin: 0.25rem 0 0.15rem; }}
  .kpi .n {{ font-size: 0.72rem; color: #95a5a6; }}

  .panel {{ background: var(--card); border-radius: 10px; padding: 1.25rem 1.4rem;
            box-shadow: 0 1px 4px rgba(16,32,43,.07); margin-bottom: 1.1rem; }}
  .panel h2 {{ font-size: 1.02rem; margin-bottom: 0.75rem; padding-bottom: 0.5rem;
               border-bottom: 1px solid var(--line); }}
  .panel h3 {{ font-size: 0.9rem; margin: 0.9rem 0 0.45rem; color: #34495e; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th, td {{ padding: 0.5rem 0.65rem; text-align: left; border-bottom: 1px solid var(--line); }}
  th {{ background: #fafbfc; color: #34495e; font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; }}

  .callout {{ background: #fef9e7; border-left: 4px solid #f39c12;
              padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; font-size: 0.84rem;
              margin-top: 0.9rem; }}
  .callout strong {{ color: #7d6608; }}
  .callout.ok {{ background: #eafaf1; border-left-color: var(--green); }}
  .callout.ok strong {{ color: #1e8449; }}

  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 0.9rem; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
           padding: 1rem 1.05rem; display: flex; flex-direction: column; gap: 0.4rem; }}
  .card-title {{ font-weight: 600; font-size: 0.9rem; }}
  .card-desc {{ font-size: 0.76rem; color: var(--muted); line-height: 1.5; flex: 1; }}
  .card-foot {{ display: flex; align-items: center; justify-content: space-between; margin-top: 0.3rem; }}
  .size {{ font-size: 0.72rem; color: #b0bec5; }}

  .btn {{ background: var(--green); color: #fff; border: 0; border-radius: 7px;
          padding: 0.5rem 1rem; font-size: 0.82rem; cursor: pointer; font-family: inherit;
          transition: background .15s; }}
  .btn:hover {{ background: #219150; }}
  .btn.small {{ padding: 0.35rem 0.8rem; font-size: 0.76rem; }}
  .btn.ghost {{ background: transparent; color: var(--green); border: 1px solid var(--green); }}
  .btn.ghost:hover {{ background: #eafaf1; }}

  .report-frame {{ width: 100%; height: calc(100vh - 8.4rem); border: 1px solid var(--line);
                   border-radius: 10px; background: #fff; }}

  pre {{ background: #10202b; color: #d7e3ea; padding: 0.9rem 1rem; border-radius: 8px;
         overflow-x: auto; font-size: 0.78rem; line-height: 1.65; }}
  pre .c {{ color: #7f96a3; }}
  code.inline {{ background: #eef2f5; padding: 0.1rem 0.35rem; border-radius: 4px;
                 font-size: 0.8em; font-family: ui-monospace, monospace; }}
  .steps {{ counter-reset: s; list-style: none; }}
  .steps li {{ counter-increment: s; position: relative; padding-left: 2rem;
               margin-bottom: 0.75rem; font-size: 0.85rem; line-height: 1.6; }}
  .steps li::before {{ content: counter(s); position: absolute; left: 0; top: 0.1rem;
    width: 1.35rem; height: 1.35rem; border-radius: 50%; background: var(--green);
    color: #fff; font-size: 0.72rem; display: grid; place-items: center; }}

  .toast {{ position: fixed; bottom: 1.4rem; right: 1.4rem; background: #1f2d3a; color: #fff;
            padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.8rem; opacity: 0;
            transform: translateY(8px); transition: all .25s; pointer-events: none; }}
  .toast.show {{ opacity: 1; transform: none; }}
</style>
</head>
<body>

<aside>
  <div class="brand">
    <div class="dot">🌿</div>
    <div><b>PHS Studio</b><small>Plant HSI Analysis</small></div>
  </div>
  <nav id="nav">
    <button class="active" data-page="overview"><span class="ico">◉</span>概览</button>
    <button data-page="report"><span class="ico">▤</span>完整分析报告</button>
    <button data-page="data"><span class="ico">⤓</span>数据与产物</button>
    <button data-page="deploy"><span class="ico">⚙</span>安装与使用</button>
  </nav>
  <div class="side-foot">
    数据集 Zenodo 15800414<br>
    分析时间 {total_s}s · v1.0.0<br>
    单文件离线可用
  </div>
</aside>

<main>
  <!-- ============ 概览 ============ -->
  <section class="page active" id="page-overview">
    <h1>Zenodo 15800414 · 番茄病害高光谱分析</h1>
    <div class="sub">用 PHS Studio v1.0.0 Step 引擎对真实高光谱立方体做端到端处理，结果与性能如下。</div>

    <div class="kpis">
      <div class="kpi">
        <div class="k">数据集规模</div>
        <div class="v">5.77 GB</div>
        <div class="n">5 类 × 8 时点 · 140×280×550</div>
      </div>
      <div class="kpi o">
        <div class="k">本次分析</div>
        <div class="v">2 cubes</div>
        <div class="n">Healthy + Alternaria · Day-1</div>
      </div>
      <div class="kpi r">
        <div class="k">差异峰值</div>
        <div class="v">{sd['peak_band_nm']:.0f} nm</div>
        <div class="n">Δ = {sd['peak_diff']:+.4f}（红边）</div>
      </div>
      <div class="kpi b">
        <div class="k">端到端耗时</div>
        <div class="v">{total_s} s</div>
        <div class="n">峰值内存 749 MB</div>
      </div>
    </div>

    <div class="panel">
      <h2>关键结论</h2>
      <table>
        <tr><th>指标</th><th>Healthy</th><th>Alternaria</th><th>解读</th></tr>
        <tr><td>叶片覆盖率</td><td class="num">34.3%</td><td class="num">27.7%</td>
            <td>病斑导致有效叶面积减少</td></tr>
        <tr><td>NIR 反射率</td><td class="num">0.580</td><td class="num">0.563</td>
            <td>近红外整体下降，细胞结构受损</td></tr>
        <tr><td>红边斜率</td><td class="num">0.1478</td><td class="num">0.1374</td>
            <td>健康叶红边跃迁更陡</td></tr>
        <tr><td>NDVI</td><td class="num">0.530</td><td class="num">0.511</td>
            <td>光合活性下降</td></tr>
        <tr><td>Otsu 分割阈值</td><td class="num">0.918</td><td class="num">0.921</td>
            <td>光谱角分割对两类都稳定</td></tr>
      </table>
      <div class="callout ok">
        <strong>与文献预期一致：</strong>差异峰值出现在红边 744 nm，方向为 Alternaria 略高；
        但平均绝对差异仅 0.014，说明病害信号是<strong>波段选择性</strong>的，不是全谱整体漂移。
      </div>
    </div>

    <div class="panel">
      <h2>性能画像</h2>
      <table>
        <tr><th>阶段</th><th>耗时</th><th>说明</th></tr>
        <tr><td>MAT v5 导入</td><td class="num">1.4–1.6 s / cube</td>
            <td>Zlib 压缩块，必须整文件解压，唯一 I/O 瓶颈</td></tr>
        <tr><td>RA 谱角相似度图</td><td class="num">~0.4 s</td><td>CPU 单线程</td></tr>
        <tr><td>Otsu + 形态学 + 连通域</td><td class="num">&lt; 1 s</td><td>—</td></tr>
        <tr><td>RF 训练（n=50→400）</td><td class="num">0.69–1.92 s</td><td>含 SNV + SG 一阶导预处理</td></tr>
        <tr><td><strong>合计（2 cubes）</strong></td><td class="num"><strong>{total_s} s</strong></td>
            <td>推算 40 cubes ≈ 6 min，峰值 ~1.5 GB</td></tr>
      </table>
    </div>
  </section>

  <!-- ============ 报告 ============ -->
  <section class="page" id="page-report">
    <h1>完整分析报告</h1>
    <div class="sub">含 10 张图表：平均光谱、差异谱、红边斜率、植被指数、分割敏感性、CV 箱线图。</div>
    <iframe class="report-frame" srcdoc="{srcdoc}" loading="lazy"></iframe>
  </section>

  <!-- ============ 数据 ============ -->
  <section class="page" id="page-data">
    <h1>数据与产物</h1>
    <div class="sub">全部内嵌在本文件中，点击即可下载，无需联网、无需服务器。</div>
    <div class="cards">{assets}</div>
    <div class="callout">
      <strong>提示：</strong>TIFF 掩膜可直接拖入 QGIS / ImageJ / Python（<code class="inline">tifffile.imread</code>）；
      CSV 带 UTF-8 BOM，Excel 双击即可正确显示中文。
    </div>
  </section>

  <!-- ============ 部署 ============ -->
  <section class="page" id="page-deploy">
    <h1>安装与使用</h1>
    <div class="sub">两种形态：浏览器离线报告（就是本页）与桌面端 PHS Studio。</div>

    <div class="panel">
      <h2>方式一 · 离线报告</h2>
      <p style="font-size:.85rem;line-height:1.7">
        本文件是完整单文件应用，双击用任意浏览器打开即可，图表、数据、下载功能全部离线可用。
      </p>
    </div>

    <div class="panel">
      <h2>方式二 · 桌面端（可交互分析）</h2>
      <ol class="steps">
        <li>下载下方 wheel 安装包。</li>
        <li>创建虚拟环境并安装：
          <pre><span class="c"># Python 3.10+</span>
python -m venv .venv
source .venv/bin/activate        <span class="c"># Windows: .venv\\Scripts\\activate</span>
pip install {wheel.name}</pre>
        </li>
        <li>启动：
          <pre>phs          <span class="c"># 打开桌面 GUI</span></pre>
        </li>
        <li>导入 MAT 立方体 → 勾选「光谱角 + Otsu 分割」→「随机森林训练」→「导出 HTML 报告」，
            即可复现本页全部结果。</li>
      </ol>
      <div style="margin-top:.8rem">
        <button class="btn" data-b64="{wheel_b64}" data-name="{wheel.name}"
                data-mime="application/octet-stream">下载 {wheel.name}</button>
        <span class="size" style="margin-left:.6rem">135 KB · 纯 Python wheel</span>
      </div>
      <div class="callout">
        <strong>Linux 无显示环境？</strong>用 <code class="inline">QT_QPA_PLATFORM=offscreen phs</code> 可跑无头模式；
        或直接用 Python API：
        <pre>from plant_hsi.core.project import Project
prj = Project.open("my.hsproj")
prj.run_step("import", inputs=[...], params={{...}})</pre>
      </div>
    </div>

    <div class="panel">
      <h2>已知限制</h2>
      <table>
        <tr><th>#</th><th>限制</th><th>影响 / 对策</th></tr>
        <tr><td>1</td><td>MAT v5 不可分块读取</td><td>导入峰值内存 ≈ 2× 立方体；已转 Zarr 后续毫秒级</td></tr>
        <tr><td>2</td><td>本次仅 2 个立方体（沙箱带宽 ~几十 KB/s）</td>
            <td>全量 40 cubes 需本地下载 5.77 GB 后重跑</td></tr>
        <tr><td>3</td><td>样本量 n=6 ROI，CV 波动大</td><td>全量数据后按文件分组划分再评估</td></tr>
        <tr><td>4</td><td>Step 缓存命中时不回填 outputs</td><td>脚本层用 <code class="inline">find_cached_outputs</code> 兜底</td></tr>
      </table>
    </div>
  </section>
</main>

<div class="toast" id="toast">已开始下载</div>

<script>
  const nav = document.getElementById('nav');
  nav.addEventListener('click', e => {{
    const btn = e.target.closest('button[data-page]');
    if (!btn) return;
    nav.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.page').forEach(p =>
      p.classList.toggle('active', p.id === 'page-' + btn.dataset.page));
    document.querySelector('main').scrollTop = 0;
  }});

  const toast = document.getElementById('toast');
  let toastTimer;
  function showToast(msg) {{
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1800);
  }}

  document.addEventListener('click', e => {{
    const btn = e.target.closest('button[data-b64]');
    if (!btn) return;
    const bin = atob(btn.dataset.b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], {{ type: btn.dataset.mime }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = btn.dataset.name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    showToast('已下载 ' + btn.dataset.name);
  }});
</script>
</body>
</html>
"""

    OUT.write_text(doc, encoding="utf-8")
    print(f"built {OUT} ({OUT.stat().st_size/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
