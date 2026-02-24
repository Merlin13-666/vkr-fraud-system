from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
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


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/")


def _link(path: Path, label: Optional[str] = None) -> str:
    """HTML link to a local file (works when opening report from the repo folder)."""
    rel = _norm(path)
    txt = label or rel
    return f'<a href="{rel}">{txt}</a>'


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
        extra={"source": _norm(path)},
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
            "source": _norm(path),
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
        extra={"source": _norm(path)},
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
            "source_val": _norm(val_path),
            "source_test": _norm(test_path),
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
            "source": _norm(path),
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

    # Cosmetic #1:
    # Put fusion_external first (main system mode), then sort by test_pr_auc desc, test_logloss asc.
    is_main = (df["model_key"] == "fusion_external").astype(int)
    df["_is_main"] = is_main
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
    # Copy key figures (if exist)
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
      body { font-family: Arial, sans-serif; margin: 24px; }
      h1, h2, h3 { margin-top: 22px; }
      .muted { color: #555; }
      .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 14px 0; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; }
      th { background: #f7f7f7; }
      img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 10px; }
      code { background: #f3f3f3; padding: 2px 5px; border-radius: 6px; }
      .small { font-size: 12px; }
      .ok { color: #0a7; font-weight: bold; }
    </style>
    """

    def _img(tag: str) -> str:
        return f'<img src="assets/{tag}" alt="{tag}"/>'

    # ---------
    # Main system summary (fusion_external)
    # ---------
    fusion_ext = next((b for b in blocks if b.key == "fusion_external"), None)
    main_summary_html = ""
    if fusion_ext is not None:
        t_fus = thresholds_fus or {}
        main_summary_html = f"""
        <div class="card">
          <h2>Главный результат системы (для РПЗ)</h2>
          <div class="muted small">
            Main mode: <code>fusion_external</code> (честный режим, future split).<br/>
            trained_on: <code>{fusion_ext.extra.get("trained_on", "external_val")}</code>,
            used_gnn: <code>{fusion_ext.extra.get("used_gnn", "calibrated")}</code><br/>
            metrics source: {_link(eval_dir / "fusion_metrics_external.json", "fusion_metrics_external.json")}
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
              <div class="muted small">source: {_link(thr_dir / "thresholds_fusion_external.json", "thresholds_fusion_external.json")}</div>
            </div>
          </div>
        </div>
        """

    # ---------
    # Model blocks
    # ---------
    blocks_html = []
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

        blocks_html.append(
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
    blocks_html = "\n".join(blocks_html)

    # ---------
    # Figures sections
    # ---------
    pr_figs = []
    pr_order = [
        "pr_curve_tabular.png",
        "pr_curve_gnn_internal.png",
        "pr_curve_fusion_internal.png",
        "pr_curve_gnn_external_val.png",
        "pr_curve_fusion_external.png",
    ]
    for f in pr_order:
        if f in copied_figs:
            title = FIG_TITLES.get(f, f)
            pr_figs.append(f"<div class='card'><h3>{title}</h3>{_img(f)}</div>")
    pr_figs_html = "\n".join(pr_figs)

    zone_figs = []
    zone_order = [
        "zone_share_tabular_val.png",
        "zone_share_tabular_test.png",
        "zone_share_fusion_external_val.png",
        "zone_share_fusion_external_test.png",
    ]
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
            return f"<div class='card'><h3>{name}</h3><div class='muted'>Not found: {_norm(src_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{name}</h3>
          <div>t_review: <code>{_fmt(thr.get("t_review"))}</code></div>
          <div>t_deny: <code>{_fmt(thr.get("t_deny"))}</code></div>
          <div class='muted small'>constraints: max_fpr_deny={thr.get("max_fpr_deny")} | max_review_share={thr.get("max_review_share")}</div>
          <div class='muted small'>source: {_link(src_path, src_path.name)}</div>
        </div>
        """

    thr_html = (
        "<div class='grid'>"
        + _thr_block("TABULAR thresholds", thresholds_tab, thr_dir / "thresholds_tabular.json")
        + _thr_block("FUSION_EXTERNAL thresholds", thresholds_fus, thr_dir / "thresholds_fusion_external.json")
        + "</div>"
    )

    # ---------
    # Decision zones
    # ---------
    def _dz_block(title: str, df: Optional[pd.DataFrame], file_path: Path) -> str:
        if df is None:
            return f"<div class='card'><h3>{title}</h3><div class='muted'>Not found: {_norm(file_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{title}</h3>
          <div class='muted small'>source: {_link(file_path, file_path.name)}</div>
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
    # Costs
    # ---------
    def _cost_block(name: str, c: Optional[Dict[str, Any]], src_path: Path) -> str:
        if not c:
            return f"<div class='card'><h3>{name}</h3><div class='muted'>Not found: {_norm(src_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{name}</h3>
          <div class='muted small'>source: {_link(src_path, src_path.name)}</div>
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
    # Comparison table
    # ---------
    comp_html = f"""
    <div class='card'>
      <h2>Model comparison</h2>
      <div class='muted small'>
        Saved: {_link(comp_path, "model_comparison.csv")}
        &nbsp;|&nbsp; Sort: <code>fusion_external</code> first, then by <code>TEST PR-AUC desc</code>, <code>TEST logloss asc</code>.
      </div>
      {_df_to_html(comp, max_rows=50)}
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
    print("[A11] Done.")


if __name__ == "__main__":
    main()