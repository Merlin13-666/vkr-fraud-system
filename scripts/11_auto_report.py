from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _fmt(x: Any, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_html(index=False, escape=False)


def _norm_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def _rel_href(target: Path, report_dir: Path) -> str:
    try:
        rel = os.path.relpath(target.resolve(), start=report_dir.resolve())
    except Exception:
        rel = str(target)
    return rel.replace("\\", "/")


def _link(target: Path, label: Optional[str] = None, report_dir: Path = Path("reports")) -> str:
    href = _rel_href(target, report_dir=report_dir)
    txt = label or href
    return f'<a href="{href}">{txt}</a>'


def _git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _pip_freeze_short(max_lines: int = 50) -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more)"]
        return "\n".join(lines)
    except Exception:
        return "pip freeze unavailable"


@dataclass
class ModelBlock:
    key: str
    title: str
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    extra: Dict[str, Any]


def _extract_metrics(eval_dir: Path, key: str, title: str, file_name: str,
                     val_key: str = "val", test_key: str = "test",
                     extra: Optional[Dict[str, Any]] = None) -> Optional[ModelBlock]:
    path = eval_dir / file_name
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key=key,
        title=title,
        val_metrics=j.get(val_key, {}),
        test_metrics=j.get(test_key, {}),
        extra={"source": _norm_path(path), **(extra or {})},
    )


def _extract_metrics_fusion_external(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "fusion_metrics_external.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="fusion_external",
        title="Гибридная модель (Tabular + GNN), честный режим (future split)",
        val_metrics=j.get("val_external", {}),
        test_metrics=j.get("test_external", {}),
        extra={
            "source": _norm_path(path),
            "weights": j.get("fusion_weights", {}),
            "trained_on": j.get("trained_on", "external_val"),
            "used_gnn": j.get("used_gnn", "calibrated"),
        },
    )


def _extract_metrics_gnn_external(eval_dir: Path) -> Optional[ModelBlock]:
    val_path = eval_dir / "gnn_external_metrics_val.json"
    test_path = eval_dir / "gnn_external_metrics_test.json"
    jv = _read_json(val_path)
    jt = _read_json(test_path)
    if not jv or not jt:
        return None
    return ModelBlock(
        key="gnn_external",
        title="GNN (индуктивно, future split)",
        val_metrics=jv,
        test_metrics=jt,
        extra={"source_val": _norm_path(val_path), "source_test": _norm_path(test_path)},
    )


def _extract_metrics_fusion_internal(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "fusion_metrics_internal.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="fusion_internal",
        title="Fusion (internal, только для абляции)",
        val_metrics=j.get("val_internal", j.get("val", {})),
        test_metrics=j.get("test_internal", j.get("test", {})),
        extra={"source": _norm_path(path)},
    )


def _extract_metrics_gnn_internal(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "gnn_metrics.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="gnn_internal",
        title="GNN (internal split внутри train-графа, только для абляции)",
        val_metrics=j.get("val", {}),
        test_metrics=j.get("test", {}),
        extra={"source": _norm_path(path), "pos_weight": (j.get("class_balance") or {}).get("pos_weight")},
    )


def _extract_metrics_graph_metrics(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "graph_metrics_baseline_metrics.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="graph_metrics",
        title="Graph-metrics baseline (контрольная модель без GNN)",
        val_metrics=j.get("val", {}),
        test_metrics=j.get("test", {}),
        extra={"source": _norm_path(path), "features": j.get("features", [])},
    )


def _build_comparison_table(blocks: List[ModelBlock]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        rows.append(
            {
                "model_key": b.key,
                "model": b.title,
                "val_logloss": b.val_metrics.get("logloss"),
                "val_pr_auc": b.val_metrics.get("pr_auc"),
                "val_roc_auc": b.val_metrics.get("roc_auc"),
                "test_logloss": b.test_metrics.get("logloss"),
                "test_pr_auc": b.test_metrics.get("pr_auc"),
                "test_roc_auc": b.test_metrics.get("roc_auc"),
            }
        )
    df = pd.DataFrame(rows)
    for c in ["val_logloss", "val_pr_auc", "val_roc_auc", "test_logloss", "test_pr_auc", "test_roc_auc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["_is_main"] = (df["model_key"] == "fusion_external").astype(int)
    df = df.sort_values(
        by=["_is_main", "test_pr_auc", "test_logloss"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns=["_is_main"]).reset_index(drop=True)
    return df


def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    thr_dir = Path("artifacts/thresholds")

    reports_dir = Path("reports")
    assets_dir = reports_dir / "assets"
    tables_dir = reports_dir / "tables"
    _ensure_dir(reports_dir)
    _ensure_dir(assets_dir)
    _ensure_dir(tables_dir)

    # ---- Collect metrics
    blocks: List[ModelBlock] = []
    blocks.append(_extract_metrics_fusion_external(eval_dir))
    blocks.append(_extract_metrics(eval_dir, "tabular", "Tabular (LightGBM)", "tabular_metrics.json"))
    blocks.append(_extract_metrics_graph_metrics(eval_dir))
    blocks.append(_extract_metrics_gnn_internal(eval_dir))
    blocks.append(_extract_metrics_fusion_internal(eval_dir))
    blocks.append(_extract_metrics_gnn_external(eval_dir))
    blocks = [b for b in blocks if b is not None]

    if not blocks:
        raise RuntimeError("[A11] No metrics found in artifacts/evaluation. Run pipeline steps first.")

    comp = _build_comparison_table(blocks)
    comp_path = tables_dir / "model_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # ---- Copy figures
    figures_to_copy = [
        ("pr_curve_tabular.png", "pr_curve_tabular.png"),
        ("pr_curve_gnn.png", "pr_curve_gnn_internal.png"),
        ("pr_curve_fusion_internal.png", "pr_curve_fusion_internal.png"),
        ("pr_curve_gnn_external_val.png", "pr_curve_gnn_external_val.png"),
        ("pr_curve_fusion_external.png", "pr_curve_fusion_external.png"),
        ("pr_curve_graph_metrics.png", "pr_curve_graph_metrics.png"),

        ("zone_share_tabular_val.png", "zone_share_tabular_val.png"),
        ("zone_share_tabular_test.png", "zone_share_tabular_test.png"),
        ("zone_share_fusion_external_val.png", "zone_share_fusion_external_val.png"),
        ("zone_share_fusion_external_test.png", "zone_share_fusion_external_test.png"),

        ("graph_degree_tx_hist.png", "graph_degree_tx_hist.png"),
        ("graph_edges_by_dst_type.png", "graph_edges_by_dst_type.png"),
        ("graph_top_entities_degree.png", "graph_top_entities_degree.png"),
        ("graph_components_hist.png", "graph_components_hist.png"),

        ("graph_metrics_feature_importance.png", "graph_metrics_feature_importance.png"),
        ("shap_summary_bar.png", "shap_summary_bar.png"),
        ("shap_summary_beeswarm.png", "shap_summary_beeswarm.png"),
    ]

    copied_figs: List[str] = []
    for src_name, dst_name in figures_to_copy:
        src = eval_dir / src_name
        dst = assets_dir / dst_name
        if _safe_copy(src, dst):
            copied_figs.append(dst_name)

    # ---- Load thresholds/cost/decision tables
    thresholds_tab = _read_json(thr_dir / "thresholds_tabular.json")
    thresholds_fus = _read_json(thr_dir / "thresholds_fusion_external.json")

    cost_tab = _read_json(eval_dir / "cost_tabular_test.json")
    cost_fus = _read_json(eval_dir / "cost_fusion_external_test.json")

    def _read_csv_if_exists(p: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(p) if p.exists() else None

    dz_tab_val = _read_csv_if_exists(eval_dir / "decision_zones_tabular_val.csv")
    dz_tab_test = _read_csv_if_exists(eval_dir / "decision_zones_tabular_test.csv")
    dz_fus_val = _read_csv_if_exists(eval_dir / "decision_zones_fusion_external_val.csv")
    dz_fus_test = _read_csv_if_exists(eval_dir / "decision_zones_fusion_external_test.csv")

    # ---- Data/split block (VKR-friendly)
    split_info = _read_json(Path("data/splits/split_info.json")) or {}
    split_html = ""
    if split_info:
        split_html = f"""
        <div class="card" id="data">
          <h2>Данные и разбиение</h2>
          <div class="muted">
            Используется временное разбиение (future split): train → val → test по росту времени транзакций.
            Это снижает риск утечки и ближе к реальной эксплуатации антифрод-системы.
          </div>
          <div class="grid">
            <div>
              <b>Размеры</b><br/>
              train: <code>{split_info.get("train_size")}</code><br/>
              val: <code>{split_info.get("val_size")}</code><br/>
              test: <code>{split_info.get("test_size")}</code>
            </div>
            <div>
              <b>Fraud rate</b><br/>
              train: <code>{_fmt((split_info.get("train_fraud_rate") or 0.0), 6)}</code><br/>
              val: <code>{_fmt((split_info.get("val_fraud_rate") or 0.0), 6)}</code><br/>
              test: <code>{_fmt((split_info.get("test_fraud_rate") or 0.0), 6)}</code>
            </div>
          </div>
          <div class="muted small">
            source: {_link(Path("data/splits/split_info.json"), "data/splits/split_info.json", report_dir=reports_dir)}
          </div>
        </div>
        """

    # ---- HTML
    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; max-width: 1200px; }
      h1, h2, h3 { margin-top: 22px; }
      .muted { color: #555; }
      .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 14px 0; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; }
      th { background: #f7f7f7; position: sticky; top: 0; }
      img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 10px; }
      code { background: #f3f3f3; padding: 2px 5px; border-radius: 6px; }
      .small { font-size: 12px; }
      .ok { color: #0a7; font-weight: bold; }
      details > summary { cursor: pointer; font-weight: bold; margin: 6px 0; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .pill { display:inline-block; padding:2px 8px; border-radius: 999px; border: 1px solid #ddd; background:#fafafa; }
      .toc a { text-decoration: none; }
      .toc li { margin: 4px 0; }
    </style>
    """

    def _img(tag: str) -> str:
        return f'<img src="{_rel_href(assets_dir / tag, report_dir=reports_dir)}" alt="{tag}"/>'

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = _git_commit_short() or "n/a"
    freeze = _pip_freeze_short()

    meta_html = f"""
    <div class="card" id="meta">
      <h2>Воспроизводимость</h2>
      <div class="small muted">
        generated_at: <span class="mono">{gen_time}</span><br/>
        git_commit: <span class="mono">{git_hash}</span><br/>
        python: <span class="mono">{sys.version.split()[0]}</span><br/>
      </div>
      <details>
        <summary>pip freeze (первые строки)</summary>
        <pre class="mono small">{freeze}</pre>
      </details>
    </div>
    """

    fusion_ext = next((b for b in blocks if b.key == "fusion_external"), None)
    thresholds_fus = thresholds_fus or {}

    main_summary_html = ""
    if fusion_ext:
        w = fusion_ext.extra.get("weights") or {}
        main_summary_html = f"""
        <div class="card" id="main_result">
          <h2>Главный результат (для РПЗ / защиты)</h2>
          <div class="muted">
            Основной режим системы — <code>fusion_external</code>: гибрид Tabular + GNN с честной оценкой на future split
            (обучение на external VAL, тестирование на external TEST).
          </div>

          <div class="grid">
            <div>
              <b>Качество на TEST</b><br/>
              logloss: <code class="ok">{_fmt(fusion_ext.test_metrics.get("logloss"))}</code><br/>
              PR-AUC: <code class="ok">{_fmt(fusion_ext.test_metrics.get("pr_auc"))}</code><br/>
              ROC-AUC: <code class="ok">{_fmt(fusion_ext.test_metrics.get("roc_auc"))}</code>
              <div class="muted small">metrics: {_link(Path(fusion_ext.extra["source"]), "fusion_metrics_external.json", report_dir=reports_dir)}</div>
            </div>
            <div>
              <b>Политика порогов</b><br/>
              t_review: <code>{_fmt(thresholds_fus.get("t_review"))}</code><br/>
              t_deny: <code>{_fmt(thresholds_fus.get("t_deny"))}</code><br/>
              <span class="muted small">
                ограничения: max_fpr_deny={thresholds_fus.get("max_fpr_deny")} | max_review_share={thresholds_fus.get("max_review_share")}
              </span>
              <div class="muted small">source: {_link(thr_dir / "thresholds_fusion_external.json", "thresholds_fusion_external.json", report_dir=reports_dir)}</div>
            </div>
          </div>

          <div style="margin-top:10px;">
            <span class="pill small">fusion weights</span>
            <span class="small mono">w_tabular={_fmt(w.get("w_tabular"), nd=6)}, w_gnn={_fmt(w.get("w_gnn"), nd=6)}, bias={_fmt(w.get("bias"), nd=6)}</span>
          </div>

          <div class="muted small" style="margin-top:8px;">
            Комментарий для ВКР: веса fusion отражают вклад источников сигналов; в реальной системе они могут
            переобучаться/перекалибровываться по дрейфу данных.
          </div>
        </div>
        """

    blocks_html = []
    for b in blocks:
        extra_line = ""
        if b.key == "fusion_external":
            extra_line = f"trained_on={b.extra.get('trained_on')} | used_gnn={b.extra.get('used_gnn')}"
        if b.key == "gnn_internal" and b.extra.get("pos_weight") is not None:
            extra_line = f"pos_weight={_fmt(b.extra.get('pos_weight'), 3)}"

        blocks_html.append(
            f"""
            <div class="card">
              <h3>{b.title}</h3>
              <div class="muted small">key: <code>{b.key}</code>{(" | " + extra_line) if extra_line else ""}</div>
              <div class="grid">
                <div>
                  <b>VAL</b><br/>
                  logloss: <code>{_fmt(b.val_metrics.get("logloss"))}</code><br/>
                  PR-AUC: <code>{_fmt(b.val_metrics.get("pr_auc"))}</code><br/>
                  ROC-AUC: <code>{_fmt(b.val_metrics.get("roc_auc"))}</code>
                </div>
                <div>
                  <b>TEST</b><br/>
                  logloss: <code>{_fmt(b.test_metrics.get("logloss"))}</code><br/>
                  PR-AUC: <code>{_fmt(b.test_metrics.get("pr_auc"))}</code><br/>
                  ROC-AUC: <code>{_fmt(b.test_metrics.get("roc_auc"))}</code>
                </div>
              </div>
            </div>
            """
        )
    blocks_html = "\n".join(blocks_html)

    comp_html = f"""
    <div class='card' id='comparison'>
      <h2>Сравнение моделей</h2>
      <div class='muted small'>
        Таблица сохранена: {_link(comp_path, "model_comparison.csv", report_dir=reports_dir)}.
        Сортировка: основной режим <code>fusion_external</code> сверху, затем по <code>TEST PR-AUC</code> и <code>TEST logloss</code>.
      </div>
      {_df_to_html(comp, max_rows=50)}
    </div>
    """

    def _thr_block(name: str, thr: Optional[Dict[str, Any]], src_path: Path) -> str:
        if not thr:
            return f"<div class='card'><h3>{name}</h3><div class='muted'>Not found: {_norm_path(src_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{name}</h3>
          <div>t_review: <code>{_fmt(thr.get("t_review"))}</code></div>
          <div>t_deny: <code>{_fmt(thr.get("t_deny"))}</code></div>
          <div class='muted small'>constraints: max_fpr_deny={thr.get("max_fpr_deny")} | max_review_share={thr.get("max_review_share")}</div>
          <div class='muted small'>source: {_link(src_path, src_path.name, report_dir=reports_dir)}</div>
        </div>
        """

    thr_html = (
        "<div class='grid'>"
        + _thr_block("TABULAR thresholds", thresholds_tab, thr_dir / "thresholds_tabular.json")
        + _thr_block("FUSION_EXTERNAL thresholds", thresholds_fus, thr_dir / "thresholds_fusion_external.json")
        + "</div>"
    )

    def _dz_block(title: str, df: Optional[pd.DataFrame], file_path: Path) -> str:
        if df is None:
            return f"<div class='card'><h3>{title}</h3><div class='muted'>Not found: {_norm_path(file_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{title}</h3>
          <div class='muted small'>source: {_link(file_path, file_path.name, report_dir=reports_dir)}</div>
          {_df_to_html(df, max_rows=10)}
        </div>
        """

    dz_html = f"""
    <div class='grid'>
      {_dz_block("TABULAR decision zones (VAL)", dz_tab_val, eval_dir / "decision_zones_tabular_val.csv")}
      {_dz_block("TABULAR decision zones (TEST)", dz_tab_test, eval_dir / "decision_zones_tabular_test.csv")}
      {_dz_block("FUSION_EXTERNAL decision zones (VAL)", dz_fus_val, eval_dir / "decision_zones_fusion_external_val.csv")}
      {_dz_block("FUSION_EXTERNAL decision zones (TEST)", dz_fus_test, eval_dir / "decision_zones_fusion_external_test.csv")}
    </div>
    """

    def _cost_block(name: str, c: Optional[Dict[str, Any]], src_path: Path) -> str:
        if not c:
            return f"<div class='card'><h3>{name}</h3><div class='muted'>Not found: {_norm_path(src_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{name}</h3>
          <div class='muted small'>source: {_link(src_path, src_path.name, report_dir=reports_dir)}</div>
          <div>avg_cost_per_tx: <code>{_fmt(c.get("avg_cost_per_tx"), nd=6)}</code></div>
          <div>total_cost: <code>{_fmt(c.get("total_cost"), nd=3)}</code></div>
        </div>
        """

    cost_html = (
        "<div class='grid'>"
        + _cost_block("TABULAR cost (TEST)", cost_tab, eval_dir / "cost_tabular_test.json")
        + _cost_block("FUSION_EXTERNAL cost (TEST)", cost_fus, eval_dir / "cost_fusion_external_test.json")
        + "</div>"
    )

    # Graph section
    graph_html_path = assets_dir / "graph.html"
    if graph_html_path.exists():
        graph_section = f"""
        <div class="card" id="graph">
          <h2>Визуализация графа (pyvis)</h2>
          <div class="muted small">file: {_link(graph_html_path, "assets/graph.html", report_dir=reports_dir)}</div>
          <p><a href="{_rel_href(graph_html_path, report_dir=reports_dir)}" target="_blank">Открыть интерактивный граф в новой вкладке</a></p>
          <details>
            <summary>Встроенный просмотр (iframe)</summary>
            <iframe src="{_rel_href(graph_html_path, report_dir=reports_dir)}" style="width:100%; height:850px; border:1px solid #eee; border-radius:10px;"></iframe>
          </details>
        </div>
        """
    else:
        graph_section = """
        <div class="card" id="graph">
          <h2>Визуализация графа</h2>
          <div class="muted">Graph HTML не сгенерирован. Установи <code>pyvis</code> и запусти шаг <code>A11_graph_viz</code>.</div>
        </div>
        """

    # TOC
    toc = """
    <div class="card toc">
      <h2>Содержание</h2>
      <ul>
        <li><a href="#main_result">Главный результат</a></li>
        <li><a href="#data">Данные и разбиение</a></li>
        <li><a href="#comparison">Сравнение моделей</a></li>
        <li><a href="#graph">Визуализация графа</a></li>
        <li><a href="#meta">Воспроизводимость</a></li>
      </ul>
    </div>
    """

    pr_figs = []
    for f, title in [
        ("pr_curve_fusion_external.png", "PR-кривая: Гибрид (честный режим) — VAL"),
        ("pr_curve_tabular.png", "PR-кривая: Tabular (LightGBM) — VAL"),
        ("pr_curve_graph_metrics.png", "PR-кривая: Graph-metrics baseline — VAL"),
        ("pr_curve_gnn_external_val.png", "PR-кривая: GNN (индуктивно) — VAL"),
    ]:
        if (assets_dir / f).exists():
            pr_figs.append(f"<div class='card'><h3>{title}</h3>{_img(f)}</div>")
    pr_figs_html = "\n".join(pr_figs) if pr_figs else "<div class='muted'>No PR figures found.</div>"

    zone_figs = []
    for f, title in [
        ("zone_share_tabular_val.png", "Доли зон: Tabular — VAL"),
        ("zone_share_tabular_test.png", "Доли зон: Tabular — TEST"),
        ("zone_share_fusion_external_val.png", "Доли зон: Hybrid — VAL"),
        ("zone_share_fusion_external_test.png", "Доли зон: Hybrid — TEST"),
    ]:
        if (assets_dir / f).exists():
            zone_figs.append(f"<div class='card'><h3>{title}</h3>{_img(f)}</div>")
    zone_figs_html = "\n".join(zone_figs) if zone_figs else "<div class='muted'>No zone-share figures found.</div>"

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>VKR Fraud System — Auto Report</title>
        {css}
      </head>
      <body>
        <h1>VKR Fraud System — Auto Report (A11)</h1>
        <div class="muted">
          Отчёт собирается из сохранённых артефактов (без переобучения) и подходит для РПЗ/презентации.
        </div>

        {toc}

        {main_summary_html}

        {split_html}

        {comp_html}

        <h2>Метрики по моделям</h2>
        {blocks_html}

        <h2>PR-кривые</h2>
        <div class="grid">{pr_figs_html}</div>

        <h2>Политика порогов</h2>
        {thr_html}

        <h2>Decision zones (preview)</h2>
        {dz_html}

        <h2>Доли зон</h2>
        <div class="grid">{zone_figs_html}</div>

        <h2>Экономика (TEST)</h2>
        {cost_html}

        {graph_section}

        {meta_html}

        <div class="card">
          <h2>Примечания для ВКР</h2>
          <ul>
            <li><b>fusion_external</b> — основной режим (честная оценка на future split).</li>
            <li><b>internal</b>-метрики используются как абляция/отладка и не выдаются за «честный прод».</li>
            <li>Graph-metrics baseline — контрольная модель: показывает вклад одной только топологии без нейросетей.</li>
          </ul>
        </div>

        <div class="muted small">
          Generated: <code>reports/report.html</code>, <code>reports/assets/</code>, <code>reports/tables/</code>.
        </div>
      </body>
    </html>
    """

    out_html = reports_dir / "report.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[A11] Saved: {out_html}")
    print(f"[A11] Saved: {comp_path}")
    print(f"[A11] Copied figures: {len(copied_figs)}")


if __name__ == "__main__":
    main()