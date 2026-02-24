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


def _norm(p: Path) -> str:
    """Normalize path for reports/json (Windows -> POSIX-like)."""
    return str(p).replace("\\", "/")


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


def _read_csv_if_exists(p: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(p) if p.exists() else None


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

    # Pretty formatting as floats where possible
    for c in ["val_logloss", "val_pr_auc", "val_roc_auc", "test_logloss", "test_pr_auc", "test_roc_auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values(["model_key"]).reset_index(drop=True)


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
        _extract_metrics_tabular,
        _extract_metrics_gnn_internal,
        _extract_metrics_fusion_internal,
        _extract_metrics_gnn_external,
        _extract_metrics_fusion_external,
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

    dz_tab_val_df = _read_csv_if_exists(dz_tab_val)
    dz_tab_test_df = _read_csv_if_exists(dz_tab_test)
    dz_fus_val_df = _read_csv_if_exists(dz_fus_val)
    dz_fus_test_df = _read_csv_if_exists(dz_fus_test)

    # ---------
    # Human titles for figures (COSMETIC FIX #1)
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
    # CSS
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
    </style>
    """

    # ---------
    # Build "Main system result" card (COSMETIC FIX #2)
    # ---------
    fusion_external_block = next((b for b in blocks if b.key == "fusion_external"), None)

    main_card_html = ""
    if fusion_external_block is not None:
        t_review = None
        t_deny = None
        if thresholds_fus:
            t_review = thresholds_fus.get("t_review")
            t_deny = thresholds_fus.get("t_deny")

        main_card_html = f"""
        <div class="card">
          <h2>Главный результат системы (для РПЗ)</h2>
          <div class="muted small">
            Main mode: <code>fusion_external</code> (честный режим, future split).<br/>
            trained_on: <code>{fusion_external_block.extra.get("trained_on")}</code>,
            used_gnn: <code>{fusion_external_block.extra.get("used_gnn")}</code>
          </div>
          <div class="grid">
            <div>
              <b>TEST</b><br/>
              logloss: <code>{_fmt(fusion_external_block.test_metrics.get("logloss"))}</code><br/>
              PR-AUC: <code>{_fmt(fusion_external_block.test_metrics.get("pr_auc"))}</code><br/>
              ROC-AUC: <code>{_fmt(fusion_external_block.test_metrics.get("roc_auc"))}</code>
            </div>
            <div>
              <b>Thresholds</b><br/>
              t_review: <code>{_fmt(t_review) if t_review is not None else "N/A"}</code><br/>
              t_deny: <code>{_fmt(t_deny) if t_deny is not None else "N/A"}</code><br/>
              <span class="muted small">
                constraints: max_fpr_deny={thresholds_fus.get("max_fpr_deny") if thresholds_fus else "N/A"}
                | max_review_share={thresholds_fus.get("max_review_share") if thresholds_fus else "N/A"}
              </span>
            </div>
          </div>
        </div>
        """
    else:
        main_card_html = """
        <div class="card">
          <h2>Главный результат системы (для РПЗ)</h2>
          <div class="muted">fusion_external not found. Run A10 first.</div>
        </div>
        """

    # ---------
    # Per-model blocks HTML
    # ---------
    blocks_html_parts: List[str] = []
    for b in blocks:
        blocks_html_parts.append(
            f"""
            <div class="card">
              <h3>{b.title}</h3>
              <div class="muted small">key: <code>{b.key}</code></div>
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
    blocks_html = "\n".join(blocks_html_parts)

    # ---------
    # Figures HTML
    # ---------
    def _img(tag: str) -> str:
        return f'<img src="assets/{tag}" alt="{tag}"/>'

    def _fig_card(filename: str) -> str:
        title = FIG_TITLES.get(filename, filename)
        return f"<div class='card'><h3>{title}</h3>{_img(filename)}</div>"

    pr_figs_html_parts: List[str] = []
    for f in [
        "pr_curve_tabular.png",
        "pr_curve_gnn_internal.png",
        "pr_curve_fusion_internal.png",
        "pr_curve_gnn_external_val.png",
        "pr_curve_fusion_external.png",
    ]:
        if f in copied_figs:
            pr_figs_html_parts.append(_fig_card(f))
    pr_figs_html = "\n".join(pr_figs_html_parts)

    zone_figs_html_parts: List[str] = []
    for f in [
        "zone_share_tabular_val.png",
        "zone_share_tabular_test.png",
        "zone_share_fusion_external_val.png",
        "zone_share_fusion_external_test.png",
    ]:
        if f in copied_figs:
            zone_figs_html_parts.append(_fig_card(f))
    zone_figs_html = "\n".join(zone_figs_html_parts)

    # ---------
    # Thresholds HTML
    # ---------
    def _thr_block(name: str, thr: Optional[Dict[str, Any]]) -> str:
        if not thr:
            return f"<div class='card'><h3>{name}</h3><div class='muted'>Not found</div></div>"
        return f"""
        <div class='card'>
          <h3>{name}</h3>
          <div>t_review: <code>{_fmt(thr.get("t_review"))}</code></div>
          <div>t_deny: <code>{_fmt(thr.get("t_deny"))}</code></div>
          <div class='muted small'>constraints: max_fpr_deny={thr.get("max_fpr_deny")} | max_review_share={thr.get("max_review_share")}</div>
        </div>
        """

    thr_html = "<div class='grid'>" + _thr_block("TABULAR thresholds", thresholds_tab) + _thr_block(
        "FUSION_EXTERNAL thresholds", thresholds_fus
    ) + "</div>"

    # ---------
    # Decision tables HTML
    # ---------
    def _dz_block(title: str, df: Optional[pd.DataFrame], file_path: Path) -> str:
        if df is None:
            return f"<div class='card'><h3>{title}</h3><div class='muted'>Not found: {_norm(file_path)}</div></div>"
        return f"""
        <div class='card'>
          <h3>{title}</h3>
          <div class='muted small'>source: <code>{_norm(file_path)}</code></div>
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
    # Cost HTML
    # ---------
    def _cost_block(name: str, c: Optional[Dict[str, Any]], src_path: Path) -> str:
        if not c:
            return f"<div class='card'><h3>{name}</h3><div class='muted'>Not found: {_norm(src_path)}</div></div>"

        n = c.get("n")
        costs = c.get("costs")
        n_html = f"<div>n: <code>{n}</code></div>" if n is not None else ""
        costs_html = f"<div class='muted small'>costs: {costs}</div>" if costs is not None else ""

        return f"""
        <div class='card'>
          <h3>{name}</h3>
          <div class='muted small'>source: <code>{_norm(src_path)}</code></div>
          <div>avg_cost_per_tx: <code>{_fmt(c.get("avg_cost_per_tx"), nd=6)}</code></div>
          <div>total_cost: <code>{_fmt(c.get("total_cost"), nd=3)}</code></div>
          {n_html}
          {costs_html}
        </div>
        """

    cost_html = "<div class='grid'>" + _cost_block(
        "TABULAR cost (TEST)", cost_tab, cost_tab_path
    ) + _cost_block(
        "FUSION_EXTERNAL cost (TEST)", cost_fus, cost_fus_path
    ) + "</div>"

    # ---------
    # Comparison table HTML
    # ---------
    comp_html = f"""
    <div class='card'>
      <h2>Model comparison</h2>
      <div class='muted small'>Saved: <code>{_norm(comp_path)}</code></div>
      {_df_to_html(comp, max_rows=50)}
    </div>
    """

    # ---------
    # Page HTML
    # ---------
    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>VKР Fraud System — Auto Report</title>
        {css}
      </head>
      <body>
        <h1>VKР Fraud System — Auto Report (A11)</h1>
        <div class="muted">
          This report is generated from saved artifacts (no training). It is suitable for RПЗ / презентация.
        </div>

        {main_card_html}

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

        <h2>Notes for RПЗ</h2>
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