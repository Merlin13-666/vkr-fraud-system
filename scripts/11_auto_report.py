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


# =========================
# Utils
# =========================

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
    """
    Делает правильный href относительно reports/report.html.
    Работает и для artifacts/* (будет ../artifacts/...)
    """
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


def _pip_freeze_short(max_lines: int = 40) -> str:
    """
    Не идеальная, но полезная штука для ВКР: фиксируем окружение.
    """
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more)"]
        return "\n".join(lines)
    except Exception:
        return "pip freeze unavailable"


# =========================
# Model specs
# =========================

@dataclass
class ModelBlock:
    key: str
    title: str
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    extra: Dict[str, Any]


def _extract_metrics_tabular(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "tabular_metrics.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="tabular",
        title="Tabular (LightGBM)",
        val_metrics=j.get("val", {}),
        test_metrics=j.get("test", {}),
        extra={"source": _norm_path(path)},
    )


def _extract_metrics_gnn_internal(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "gnn_metrics.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="gnn_internal",
        title="GNN (internal split inside train graph)",
        val_metrics=j.get("val", {}),
        test_metrics=j.get("test", {}),
        extra={
            "source": _norm_path(path),
            "pos_weight": (j.get("class_balance") or {}).get("pos_weight"),
        },
    )


def _extract_metrics_fusion_internal(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "fusion_metrics_internal.json"
    j = _read_json(path)
    if not j:
        return None

    # internal fusion JSON uses keys: train / val_internal / test_internal
    val = j.get("val_internal", j.get("val", {}))
    test = j.get("test_internal", j.get("test", {}))

    return ModelBlock(
        key="fusion_internal",
        title="Fusion (internal, for ablation only)",
        val_metrics=val,
        test_metrics=test,
        extra={"source": _norm_path(path)},
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
        title="GNN (external inductive, future split)",
        val_metrics=jv,
        test_metrics=jt,
        extra={
            "source_val": _norm_path(val_path),
            "source_test": _norm_path(test_path),
        },
    )


def _extract_metrics_fusion_external(eval_dir: Path) -> Optional[ModelBlock]:
    path = eval_dir / "fusion_metrics_external.json"
    j = _read_json(path)
    if not j:
        return None
    return ModelBlock(
        key="fusion_external",
        title="Fusion (external honest, trained on external VAL, tested on external TEST)",
        val_metrics=j.get("val_external", {}),
        test_metrics=j.get("test_external", {}),
        extra={
            "source": _norm_path(path),
            "weights": j.get("fusion_weights", {}),
            "trained_on": j.get("trained_on", "external_val"),
            "used_gnn": j.get("used_gnn", "calibrated"),
        },
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
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Put fusion_external first (main system mode), then sort by test_pr_auc desc, test_logloss asc
    df["_is_main"] = (df["model_key"] == "fusion_external").astype(int)
    df = df.sort_values(
        by=["_is_main", "test_pr_auc", "test_logloss"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns=["_is_main"]).reset_index(drop=True)

    return df


# =========================
# Report builder
# =========================

def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    thr_dir = Path("artifacts/thresholds")

    reports_dir = Path("reports")
    assets_dir = reports_dir / "assets"
    tables_dir = reports_dir / "tables"

    _ensure_dir(reports_dir)
    _ensure_dir(assets_dir)
    _ensure_dir(tables_dir)

    # SHAP outputs are produced directly into reports/assets + reports/tables (by scripts/12_shap_tabular.py)
    shap_bar = assets_dir / "shap_summary_bar.png"
    shap_bees = assets_dir / "shap_summary_beeswarm.png"
    shap_csv = tables_dir / "shap_top_features.csv"
    shap_meta = tables_dir / "shap_meta.json"

    # ---------
    # Collect model metrics
    # ---------
    blocks: List[ModelBlock] = []
    for fn in [
        _extract_metrics_fusion_external,  # try main mode first
        _extract_metrics_tabular,
        _extract_metrics_gnn_internal,
        _extract_metrics_fusion_internal,
        _extract_metrics_gnn_external,
    ]:
        b = fn(eval_dir)
        if b is not None:
            blocks.append(b)

    if not blocks:
        raise RuntimeError("[A11] No metrics found in artifacts/evaluation. Run A3/A5/A6/A9/A10 first.")

    comp = _build_comparison_table(blocks)
    comp_path = tables_dir / "model_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # ---------
    # Copy key figures (if exist) to reports/assets
    # ---------
    figures_to_copy = [
        ("pr_curve_tabular.png", "pr_curve_tabular.png"),
        ("pr_curve_gnn.png", "pr_curve_gnn_internal.png"),
        ("pr_curve_fusion_internal.png", "pr_curve_fusion_internal.png"),
        ("pr_curve_gnn_external_val.png", "pr_curve_gnn_external_val.png"),
        ("pr_curve_fusion_external.png", "pr_curve_fusion_external.png"),

        ("zone_share_tabular_val.png", "zone_share_tabular_val.png"),
        ("zone_share_tabular_test.png", "zone_share_tabular_test.png"),
        ("zone_share_fusion_external_val.png", "zone_share_fusion_external_val.png"),
        ("zone_share_fusion_external_test.png", "zone_share_fusion_external_test.png"),
    ]

    copied_figs: List[str] = []
    for src_name, dst_name in figures_to_copy:
        src = eval_dir / src_name
        dst = assets_dir / dst_name
        if _safe_copy(src, dst):
            copied_figs.append(dst_name)

    # ---------
    # Load thresholds + decision tables + costs (tabular + fusion_external)
    # ---------
    thresholds_tab = _read_json(thr_dir / "thresholds_tabular.json")
    thresholds_fus = _read_json(thr_dir / "thresholds_fusion_external.json")

    dz_tab_val = eval_dir / "decision_zones_tabular_val.csv"
    dz_tab_test = eval_dir / "decision_zones_tabular_test.csv"
    dz_fus_val = eval_dir / "decision_zones_fusion_external_val.csv"
    dz_fus_test = eval_dir / "decision_zones_fusion_external_test.csv"

    cost_tab_path = eval_dir / "cost_tabular_test.json"
    cost_fus_path = eval_dir / "cost_fusion_external_test.json"
    cost_tab = _read_json(cost_tab_path)
    cost_fus = _read_json(cost_fus_path)

    def _read_csv_if_exists(p: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(p) if p.exists() else None

    dz_tab_val_df = _read_csv_if_exists(dz_tab_val)
    dz_tab_test_df = _read_csv_if_exists(dz_tab_test)
    dz_fus_val_df = _read_csv_if_exists(dz_fus_val)
    dz_fus_test_df = _read_csv_if_exists(dz_fus_test)

    # optional: show an example predict file with reasons, if exists
    pred_with_reasons_path = Path("artifacts/predict/test_tabular_with_reasons.parquet")
    pred_with_reasons_exists = pred_with_reasons_path.exists()

    # ---------
    # Titles
    # ---------
    FIG_TITLES = {
        "pr_curve_tabular.png": "PR-кривая: Tabular (LightGBM)",
        "pr_curve_gnn_internal.png": "PR-кривая: GNN (internal, ablation)",
        "pr_curve_fusion_internal.png": "PR-кривая: Fusion (internal, ablation)",
        "pr_curve_gnn_external_val.png": "PR-кривая: GNN (external, inductive) — VAL",
        "pr_curve_fusion_external.png": "PR-кривая: Fusion (external, честный режим) — VAL",

        "zone_share_tabular_val.png": "Доли зон: Tabular — VAL",
        "zone_share_tabular_test.png": "Доли зон: Tabular — TEST",
        "zone_share_fusion_external_val.png": "Доли зон: Fusion external — VAL",
        "zone_share_fusion_external_test.png": "Доли зон: Fusion external — TEST",
    }

    # ---------
    # HTML helpers
    # ---------
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
      .warn { color: #b45309; font-weight: bold; }
      .toc a { text-decoration: none; }
      .toc li { margin: 4px 0; }
      details > summary { cursor: pointer; font-weight: bold; margin: 6px 0; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .pill { display:inline-block; padding:2px 8px; border-radius: 999px; border: 1px solid #ddd; background:#fafafa; }
    </style>
    """

    def _img(tag: str) -> str:
        return f'<img src="{_rel_href(assets_dir / tag, report_dir=reports_dir)}" alt="{tag}"/>'

    # ---------
    # Header/meta block (VKR+)
    # ---------
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_hash = _git_commit_short() or "n/a"
    freeze = _pip_freeze_short()

    meta_html = f"""
    <div class="card">
      <h2 id="meta">Reproducibility / Metadata</h2>
      <div class="small muted">
        generated_at: <span class="mono">{gen_time}</span><br/>
        git_commit: <span class="mono">{git_hash}</span><br/>
        python: <span class="mono">{sys.version.split()[0]}</span><br/>
      </div>
      <details>
        <summary>pip freeze (first lines)</summary>
        <pre class="mono small">{freeze}</pre>
      </details>
    </div>
    """

    # ---------
    # Main system summary (fusion_external)
    # ---------
    fusion_ext = next((b for b in blocks if b.key == "fusion_external"), None)
    main_summary_html = ""
    if fusion_ext is not None:
        t_fus = thresholds_fus or {}
        w = fusion_ext.extra.get("weights") or {}
        main_summary_html = f"""
        <div class="card" id="main_result">
          <h2>Главный результат системы (для РПЗ)</h2>
          <div class="muted small">
            Main mode: <code>fusion_external</code> (честный режим, future split).<br/>
            trained_on: <code>{fusion_ext.extra.get("trained_on", "external_val")}</code>,
            used_gnn: <code>{fusion_ext.extra.get("used_gnn", "calibrated")}</code><br/>
            metrics source: {_link(eval_dir / "fusion_metrics_external.json", "fusion_metrics_external.json", report_dir=reports_dir)}
          </div>
          <div class="grid">
            <div>
              <b>TEST</b><br/>
              logloss: <code class="ok">{_fmt(fusion_ext.test_metrics.get("logloss"))}</code><br/>
              PR-AUC: <code class="ok">{_fmt(fusion_ext.test_metrics.get("pr_auc"))}</code><br/>
              ROC-AUC: <code class="ok">{_fmt(fusion_ext.test_metrics.get("roc_auc"))}</code>
            </div>
            <div>
              <b>Thresholds</b><br/>
              t_review: <code>{_fmt(t_fus.get("t_review"))}</code><br/>
              t_deny: <code>{_fmt(t_fus.get("t_deny"))}</code><br/>
              <span class="muted small">
                constraints: max_fpr_deny={t_fus.get("max_fpr_deny")} | max_review_share={t_fus.get("max_review_share")}
              </span>
              <div class="muted small">source: {_link(thr_dir / "thresholds_fusion_external.json", "thresholds_fusion_external.json", report_dir=reports_dir)}</div>
            </div>
          </div>

          <div style="margin-top:10px;">
            <span class="pill small">fusion weights</span>
            <span class="small mono">w_tabular={_fmt(w.get("w_tabular"), nd=6)}, w_gnn={_fmt(w.get("w_gnn"), nd=6)}, bias={_fmt(w.get("bias"), nd=6)}</span>
          </div>
        </div>
        """

    # ---------
    # Model blocks section
    # ---------
    blocks_html_list = []
    for b in blocks:
        extra_bits = []
        if b.key == "fusion_external":
            extra_bits.append(f"trained_on={b.extra.get('trained_on')}")
            extra_bits.append(f"used_gnn={b.extra.get('used_gnn')}")
        if b.key == "gnn_internal" and b.extra.get("pos_weight") is not None:
            extra_bits.append(f"pos_weight={_fmt(b.extra.get('pos_weight'), nd=3)}")

        extra_line = ""
        if extra_bits:
            extra_line = " | ".join(extra_bits)

        blocks_html_list.append(
            f"""
            <div class="card">
              <h3>{b.title}</h3>
              <div class="muted small">
                key: <code>{b.key}</code>
                {"&nbsp;|&nbsp;" + extra_line if extra_line else ""}
              </div>
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
    blocks_html = "\n".join(blocks_html_list)

    # ---------
    # Figures sections
    # ---------
    pr_order = [
        "pr_curve_tabular.png",
        "pr_curve_gnn_internal.png",
        "pr_curve_fusion_internal.png",
        "pr_curve_gnn_external_val.png",
        "pr_curve_fusion_external.png",
    ]
    pr_figs = []
    for f in pr_order:
        if f in copied_figs:
            title = FIG_TITLES.get(f, f)
            pr_figs.append(f"<div class='card'><h3>{title}</h3>{_img(f)}</div>")
    pr_figs_html = "\n".join(pr_figs)

    zone_order = [
        "zone_share_tabular_val.png",
        "zone_share_tabular_test.png",
        "zone_share_fusion_external_val.png",
        "zone_share_fusion_external_test.png",
    ]
    zone_figs = []
    for f in zone_order:
        if f in copied_figs:
            title = FIG_TITLES.get(f, f)
            zone_figs.append(f"<div class='card'><h3>{title}</h3>{_img(f)}</div>")
    zone_figs_html = "\n".join(zone_figs)

    # ---------
    # Threshold blocks
    # ---------
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

    # ---------
    # Decision zones blocks
    # ---------
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
      {_dz_block("TABULAR decision zones (VAL)", dz_tab_val_df, dz_tab_val)}
      {_dz_block("TABULAR decision zones (TEST)", dz_tab_test_df, dz_tab_test)}
      {_dz_block("FUSION_EXTERNAL decision zones (VAL)", dz_fus_val_df, dz_fus_val)}
      {_dz_block("FUSION_EXTERNAL decision zones (TEST)", dz_fus_test_df, dz_fus_test)}
    </div>
    """

    # ---------
    # Costs blocks
    # ---------
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
        + _cost_block("TABULAR cost (TEST)", cost_tab, cost_tab_path)
        + _cost_block("FUSION_EXTERNAL cost (TEST)", cost_fus, cost_fus_path)
        + "</div>"
    )

    # ---------
    # Comparison table block
    # ---------
    comp_html = f"""
    <div class='card' id='comparison'>
      <h2>Model comparison</h2>
      <div class='muted small'>
        Saved: {_link(comp_path, "model_comparison.csv", report_dir=reports_dir)}
        &nbsp;|&nbsp; Sort: <code>fusion_external</code> first, then by <code>TEST PR-AUC desc</code>, <code>TEST logloss asc</code>.
      </div>
      {_df_to_html(comp, max_rows=50)}
    </div>
    """

    # ---------
    # SHAP section
    # ---------
    shap_html = ""
    if shap_bar.exists():
        shap_html_parts: List[str] = []
        shap_html_parts.append("<h2 id='shap'>Интерпретация (SHAP, табличная модель)</h2>")
        shap_html_parts.append(
            "<div class='muted'>Глобальная важность признаков и их вклад в решение модели LightGBM. "
            "Для batch-predict доступна опция <code>--with-reasons</code>, которая добавляет <code>top_reasons</code> (top-k вкладов SHAP на транзакцию).</div>"
        )

        shap_html_parts.append("<div class='grid'>")
        shap_html_parts.append(
            f"<div class='card'><h3>SHAP summary (bar)</h3><img src='{_rel_href(shap_bar, report_dir=reports_dir)}' /></div>"
        )
        if shap_bees.exists():
            shap_html_parts.append(
                f"<div class='card'><h3>SHAP summary (beeswarm)</h3><img src='{_rel_href(shap_bees, report_dir=reports_dir)}' /></div>"
            )
        shap_html_parts.append("</div>")

        if shap_csv.exists():
            try:
                df_shap = pd.read_csv(shap_csv).head(30)
                shap_html_parts.append("<div class='card'>")
                shap_html_parts.append("<h3>Top features (mean |SHAP|)</h3>")
                shap_html_parts.append(_df_to_html(df_shap, max_rows=30))
                shap_html_parts.append(
                    f"<div class='muted small'>source: {_link(shap_csv, 'shap_top_features.csv', report_dir=reports_dir)}</div>"
                )
                if shap_meta.exists():
                    shap_html_parts.append(
                        f"<div class='muted small'>meta: {_link(shap_meta, 'shap_meta.json', report_dir=reports_dir)}</div>"
                    )
                shap_html_parts.append("</div>")
            except Exception:
                pass

        if pred_with_reasons_exists:
            shap_html_parts.append("<div class='card'>")
            shap_html_parts.append("<h3>Пример output с причинами</h3>")
            shap_html_parts.append(
                "<div class='muted'>Файл содержит колонку <code>top_reasons</code> — JSON-массив из top-k признаков "
                "с их значениями и вкладом SHAP (положительный → повышает риск, отрицательный → снижает).</div>"
            )
            shap_html_parts.append(
                f"<div class='muted small'>example: {_link(pred_with_reasons_path, pred_with_reasons_path.name, report_dir=reports_dir)}</div>"
            )
            shap_html_parts.append("</div>")

        shap_html = "\n".join(shap_html_parts)
    else:
        shap_html = "<div class='card'><h2 id='shap'>Интерпретация (SHAP)</h2><div class='muted'>SHAP artifacts not found.</div></div>"

    # ---------
    # Graph section (FIX: no duplicated <h2>)
    # ---------
    graph_html_path = assets_dir / "graph.html"
    if graph_html_path.exists():
        graph_section = f"""
        <div class="card" id="graph">
          <h2>Graph visualization</h2>
          <div class="muted small">file: {_link(graph_html_path, "assets/graph.html", report_dir=reports_dir)}</div>
          <p>
            <a href="{_rel_href(graph_html_path, report_dir=reports_dir)}" target="_blank">Open interactive graph (pyvis) in new tab</a>
          </p>
          <details>
            <summary>Embed preview (iframe)</summary>
            <iframe src="{_rel_href(graph_html_path, report_dir=reports_dir)}" style="width:100%; height:850px; border:1px solid #eee; border-radius:10px;"></iframe>
          </details>
        </div>
        """
    else:
        graph_section = """
        <div class="card" id="graph">
          <h2>Graph visualization</h2>
          <div class="muted">Graph HTML is not generated. Install <code>pyvis</code> and run step <code>A11_graph_viz</code>.</div>
        </div>
        """

    # ---------
    # TOC
    # ---------
    toc = """
    <div class="card toc">
      <h2>Contents</h2>
      <ul>
        <li><a href="#main_result">Main result</a></li>
        <li><a href="#comparison">Model comparison</a></li>
        <li><a href="#shap">SHAP interpretation</a></li>
        <li><a href="#graph">Graph visualization</a></li>
        <li><a href="#meta">Reproducibility / Metadata</a></li>
      </ul>
    </div>
    """

    # ---------
    # Page
    # ---------
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
          This report is generated from saved artifacts (no training). It is suitable for РПЗ / презентация.
        </div>

        {toc}

        {main_summary_html}

        {comp_html}

        <h2>Per-model metrics</h2>
        {blocks_html}

        <h2>PR curves</h2>
        <div class="grid">
          {pr_figs_html}
        </div>

        <h2>Thresholding policy</h2>
        {thr_html}

        <h2>Decision zones (preview)</h2>
        {dz_html}

        <h2>Zone share plots</h2>
        <div class="grid">
          {zone_figs_html}
        </div>

        <h2>Economics / cost (TEST)</h2>
        {cost_html}

        {shap_html}

        {graph_section}

        {meta_html}

        <h2>Notes for РПЗ</h2>
        <div class="card">
          <ul>
            <li><b>fusion_external</b> is the main system mode (honest evaluation on future split).</li>
            <li>Internal GNN/Fusion metrics are shown for ablation and debugging only.</li>
            <li>Thresholds are fitted on VAL (constraints: max FPR for deny + max share for review).</li>
          </ul>
        </div>

        <div class="muted small">
          Generated files: <code>reports/report.html</code>, <code>reports/assets/</code>, <code>reports/tables/</code>.
        </div>
      </body>
    </html>
    """

    out_html = reports_dir / "report.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[A11] Saved: {out_html}")
    print(f"[A11] Saved: {comp_path}")
    if copied_figs:
        print(f"[A11] Copied figures to: {assets_dir} ({len(copied_figs)} files)")
    else:
        print("[A11] No figures copied (some PNGs may be missing).")
    if shap_bar.exists():
        print("[A11] SHAP section: included (reports/assets/shap_*.png found).")
    if graph_html_path.exists():
        print("[A11] Graph section: included (reports/assets/graph.html found).")
    print("[A11] Done.")


if __name__ == "__main__":
    main()