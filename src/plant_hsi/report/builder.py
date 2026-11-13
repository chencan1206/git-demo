"""Self-contained HTML report builder (no Jinja2 / Plotly dependency).

Charts are rendered as inline SVG so the report is a single portable file that
opens in any browser and prints cleanly.
"""

from __future__ import annotations

import contextlib
import html
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

_CSS = """
:root {
  --ink: #1b2430;
  --muted: #5c6b7f;
  --line: #d8e0ea;
  --card: #ffffff;
  --bg: #f3f6fa;
  --accent: #2f6fdb;
  --accent-soft: #e8f0fd;
  --ok: #1f9d63;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
               "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 14px; line-height: 1.6;
}
header {
  background: linear-gradient(120deg, #15233a 0%, #23477e 60%, #2f6fdb 100%);
  color: #fff; padding: 34px 44px 28px;
}
header h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: .3px; }
header .sub { opacity: .82; font-size: 13px; }
main { max-width: 1180px; margin: 0 auto; padding: 26px 24px 64px; }
section {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 2px rgba(22, 40, 70, .04);
}
section h2 {
  margin: 0 0 14px; font-size: 17px; border-left: 4px solid var(--accent);
  padding-left: 10px;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { background: #f7f9fc; color: var(--muted); font-weight: 600; }
tr:last-child td { border-bottom: none; }
code, pre {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
}
pre {
  background: #f7f9fc; border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
}
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
.metric {
  flex: 1 1 150px; background: var(--accent-soft); border-radius: 10px;
  padding: 12px 16px; border: 1px solid #d4e3fb;
}
.metric .v { font-size: 24px; font-weight: 700; color: var(--accent); }
.metric .k { font-size: 12px; color: var(--muted); }
.figures { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.figure { flex: 1 1 420px; min-width: 320px; }
.figure svg { width: 100%; height: auto; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.pill {
  display: inline-block; padding: 1px 9px; border-radius: 20px; font-size: 12px;
  background: #e8f6ee; color: var(--ok); border: 1px solid #c6ead7;
}
footer { text-align: center; color: var(--muted); font-size: 12px; padding: 24px 0 8px; }
@media print {
  body { background: #fff; }
  section { box-shadow: none; break-inside: avoid; }
  header { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _json_block(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return f"<code>{_esc(value)}</code>"
    return f"<pre>{_esc(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def confusion_matrix_svg(matrix: list[list[int]], labels: list[str], cell: int = 46) -> str:
    n = len(labels)
    if n == 0 or not matrix:
        return "<p class='note'>无混淆矩阵。</p>"
    max_value = max((max(row) for row in matrix), default=0) or 1
    left, top = 120, 26
    width = left + n * cell + 24
    height = top + n * cell + 44

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="混淆矩阵" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for i in range(n):
        parts.append(
            f'<text x="{left - 8}" y="{top + i * cell + cell / 2 + 4}" '
            f'text-anchor="end" font-size="12" fill="#5c6b7f">{_esc(labels[i])}</text>'
        )
        parts.append(
            f'<text x="{left + i * cell + cell / 2}" y="{top + n * cell + 18}" '
            f'text-anchor="middle" font-size="12" fill="#5c6b7f">{_esc(labels[i])}</text>'
        )
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            t = float(value) / max_value
            r = int(255 + t * (47 - 255))
            g = int(255 + t * (111 - 255))
            b = int(255 + t * (219 - 255))
            fill = f"rgb({r},{g},{b})"
            colour = "#ffffff" if t > 0.55 else "#1b2430"
            x = left + j * cell
            y = top + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" '
                f'stroke="#ffffff" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" '
                f'font-size="13" font-weight="600" fill="{colour}">{int(value)}</text>'
            )
    parts.append(f'<text x="{left}" y="16" font-size="12" fill="#5c6b7f">预测 →</text>')
    parts.append(f'<text x="14" y="{top + 12}" font-size="12" fill="#5c6b7f">真实 ↓</text>')
    parts.append("</svg>")
    return "".join(parts)


def scatter_svg(y_true: Any, y_pred: Any, width: int = 560, height: int = 400) -> str:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0 or yt.size != yp.size:
        return "<p class='note'>无预测散点数据。</p>"
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    pad = 54
    inner_w = width - pad - 24
    inner_h = height - pad - 36

    def sx(v: float) -> float:
        return pad + (v - lo) / (hi - lo) * inner_w

    def sy(v: float) -> float:
        return height - 36 - (v - lo) / (hi - lo) * inner_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="预测散点图" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{pad}" y1="{height - 36}" x2="{width - 24}" y2="{height - 36}" '
        f'stroke="#c8d2df"/>',
        f'<line x1="{pad}" y1="20" x2="{pad}" y2="{height - 36}" stroke="#c8d2df"/>',
        f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(hi)}" y2="{sy(hi)}" '
        f'stroke="#9fb4d2" stroke-dasharray="6 5"/>',
    ]
    for x, y in zip(yt, yp, strict=False):
        parts.append(
            f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="3.6" '
            f'fill="#2f6fdb" fill-opacity="0.72"/>'
        )
    parts.append(f'<text x="{pad}" y="{height - 14}" font-size="12" fill="#5c6b7f">实测值</text>')
    parts.append('<text x="12" y="30" font-size="12" fill="#5c6b7f">预测值</text>')
    parts.append(
        f'<text x="{width - 150}" y="30" font-size="12" fill="#5c6b7f">虚线 = 1:1 参考</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _metric_cards(task: str, metrics: dict[str, Any]) -> str:
    if task == "classification":
        specs = [
            ("accuracy", "准确率 Accuracy"),
            ("f1_macro", "F1（宏平均）"),
            ("precision_macro", "精确率（宏平均）"),
            ("recall_macro", "召回率（宏平均）"),
        ]
    else:
        specs = [
            ("r2", "决定系数 R²"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("mse", "MSE"),
        ]
    cards = "".join(
        f'<div class="metric"><div class="v">{_fmt_num(metrics.get(key), 4)}</div>'
        f'<div class="k">{_esc(label)}</div></div>'
        for key, label in specs
        if key in metrics
    )
    return f'<div class="cards">{cards}</div>'


def _per_class_table(metrics: dict[str, Any]) -> str:
    rows = metrics.get("per_class") or []
    if not rows:
        return ""
    body = [
        [
            _esc(r.get("label")),
            _fmt_num(r.get("precision")),
            _fmt_num(r.get("recall")),
            _fmt_num(r.get("f1")),
            _esc(r.get("support")),
        ]
        for r in rows
    ]
    return "<h3>分类别指标</h3>" + _table(["类别", "精确率", "召回率", "F1", "样本数"], body)


def _cv_table(cv: dict[str, Any] | None) -> str:
    if not cv or not cv.get("metrics"):
        return "<p class='note'>未进行交叉验证。</p>"
    rows = []
    for name, item in cv["metrics"].items():
        rows.append(
            [
                _esc(name),
                _fmt_num(item.get("mean")),
                _fmt_num(item.get("std")),
                _esc(cv.get("folds", "")),
            ]
        )
    return _table(["指标", "均值", "标准差", "折数"], rows)


def _chain_rows(chain: dict[str, Any] | None) -> list[list[Any]]:
    if not chain or not chain.get("ops"):
        return [["—", "未使用预处理链"]]
    rows = []
    for i, op in enumerate(chain["ops"], start=1):
        params = op.get("params") or {}
        body = ", ".join(f"{k}={v}" for k, v in params.items()) or "—"
        rows.append([i, _esc(op.get("kind", "")), _esc(body)])
    return rows


def _normalise_run(run: dict[str, Any]) -> dict[str, Any]:
    out = dict(run)
    for key in ("params", "output_artifacts"):
        value = out.get(key)
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                out[key] = json.loads(value)
    return out


def build_report(
    *,
    project: dict[str, Any],
    model: dict[str, Any],
    metrics_payload: dict[str, Any],
    predictions: pd.DataFrame | None,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    title: str = "PlantHSI Studio · 传统建模报告",
) -> str:
    """Render the full report HTML."""
    task = str(model.get("task") or metrics_payload.get("task") or "classification")
    algorithm = str(model.get("algorithm") or metrics_payload.get("algorithm") or "")
    metrics = metrics_payload.get("metrics") or model.get("metrics") or {}
    cv = metrics_payload.get("cv") or model.get("cv")

    project_rows = [
        ["项目名称", _esc(project.get("name", ""))],
        ["项目 ID", f"<code>{_esc(project.get('project_id', ''))}</code>"],
        ["创建时间", _esc(project.get("created_at", ""))],
        ["描述", _esc(project.get("description") or "—")],
        ["schema 版本", _esc(project.get("schema_version", ""))],
    ]

    model_rows = [
        ["模型 ID", f"<code>{_esc(model.get('model_id', ''))}</code>"],
        ["任务 / 算法", f"{_esc(task)} / {_esc(algorithm)}"],
        ["框架", _esc(model.get("framework", "scikit-learn"))],
        ["随机种子", _esc(model.get("seed", ""))],
        ["训练 / 测试样本", f"{model.get('n_train', '')} / {model.get('n_test', '')}"],
        [
            "特征数",
            _esc(len(model.get("feature_names") or metrics_payload.get("feature_names") or [])),
        ],
        ["类别", _esc(", ".join(model.get("classes") or []) or "—")],
        ["创建时间", _esc(model.get("created_at", ""))],
        ["运行环境", _esc(json.dumps(model.get("env") or {}, ensure_ascii=False))],
    ]

    split = model.get("split") or metrics_payload.get("split") or {}
    split_rows = [
        ["训练集", _esc(split.get("n_train", ""))],
        ["验证集", _esc(split.get("n_val", 0))],
        ["测试集", _esc(split.get("n_test", ""))],
        ["分层抽样", "是" if split.get("stratified") else "否"],
        ["种子", _esc(split.get("seed", ""))],
        ["测试比例", _fmt_num(split.get("test_fraction", ""), 3)],
    ]

    if task == "classification":
        figure = confusion_matrix_svg(
            metrics.get("confusion_matrix") or [],
            [str(v) for v in metrics.get("labels") or []],
        )
        figure_title = "混淆矩阵（测试集）"
    else:
        if predictions is not None and {"y_true", "y_pred"} <= set(predictions.columns):
            figure = scatter_svg(predictions["y_true"], predictions["y_pred"])
        else:
            figure = "<p class='note'>缺少预测文件，无法绘制散点图。</p>"
        figure_title = "预测值 vs 实测值（测试集）"

    run_rows = []
    for raw in runs:
        run = _normalise_run(raw)
        run_rows.append(
            [
                f"<code>{_esc(run.get('id', ''))}</code>",
                _esc(run.get("step", "")),
                f"<span class='pill'>{_esc(run.get('status', ''))}</span>",
                _esc(run.get("started_at", "")),
                _esc(run.get("finished_at", "")),
                _esc(run.get("seed", "")),
                _json_block(run.get("params") or {}),
            ]
        )

    artifact_rows = [
        [
            f"<code>{_esc(a.get('id', ''))}</code>",
            _esc(a.get("kind", "")),
            _esc(a.get("name", "")),
            f"<code>{_esc(a.get('uri', ''))}</code>",
            f"<code>{_esc((a.get('hash') or '')[:23])}</code>",
        ]
        for a in artifacts
    ]

    generated = datetime.now(UTC).isoformat(timespec="seconds")
    project_name = _esc(project.get("name", "未命名项目"))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="generator" content="PHS Studio"/>
<title>{_esc(title)} — {project_name}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{_esc(title)}</h1>
  <div class="sub">项目：{project_name} &nbsp;·&nbsp; 模型：{_esc(algorithm)} &nbsp;·&nbsp;
      生成时间：{_esc(generated)}</div>
</header>
<main>
  <section id="project">
    <h2>1 · 项目信息</h2>
    {_table(["项", "值"], project_rows)}
  </section>

  <section id="model">
    <h2>2 · 模型卡</h2>
    {_table(["项", "值"], model_rows)}
    <h3>数据划分</h3>
    {_table(["项", "值"], split_rows)}
    <h3>预处理链</h3>
    {_table(["序号", "算子", "参数"], _chain_rows(model.get("preprocess_chain")))}
  </section>

  <section id="metrics">
    <h2>3 · 测试集指标</h2>
    {_metric_cards(task, metrics)}
    <h3>交叉验证（训练集）</h3>
    {_cv_table(cv)}
    {_per_class_table(metrics)}
  </section>

  <section id="figures">
    <h2>4 · 关键图</h2>
    <div class="figures">
      <div class="figure">
        <h3>{_esc(figure_title)}</h3>
        {figure}
      </div>
    </div>
  </section>

  <section id="runs">
    <h2>5 · 运行记录</h2>
    {_table(["运行 ID", "Step", "状态", "开始", "结束", "种子", "参数"], run_rows)}
  </section>

  <section id="artifacts">
    <h2>6 · 产物清单</h2>
    {_table(["ID", "类型", "名称", "相对路径", "hash"], artifact_rows)}
  </section>

  <section id="reproduce">
    <h2>7 · 复现说明</h2>
    <p>运行清单（run manifest）与报告同目录生成，包含项目信息、模型卡、运行参数、
       产物 hash 与相对路径。换机器复现时，只需重定向项目根目录（保持
       <code>artifacts/</code> 下的相对路径不变），即可按记录重跑相同 Step 与参数。</p>
    <p class="note">所有指标由 PHS Studio 自动计算；固定数据 + 参数 + 种子可复现。</p>
  </section>
</main>
<footer>PlantHSI Studio · v0.1.0 · report generated {_esc(generated)}</footer>
</body>
</html>
"""
