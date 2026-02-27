from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# -------------------------
# Helpers
# -------------------------

def _run_module(module: str, args: Optional[List[str]] = None) -> None:
    """
    Run: python -m <module> [args...]
    Uses current interpreter (venv-safe).
    """
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)

    print(f"\n[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"[RUN] Failed: {module} (code={proc.returncode})")


def _exists_all(paths: List[Path]) -> bool:
    return all(p.exists() for p in paths)


def _p(*parts: str) -> Path:
    return Path(*parts)


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    outputs: List[Path]
    module: str
    module_args: List[str]


def _should_run(step: Step, force: bool) -> bool:
    if force:
        return True
    # if ANY output missing -> run
    return not _exists_all(step.outputs)


# -------------------------
# Pipeline definition
# -------------------------

def build_steps(skip_gnn: bool, skip_external: bool, skip_internal_fusion: bool, skip_report: bool) -> List[Step]:
    """
    Defines end-to-end 'one button' pipeline.
    We keep steps in a logical order and make each step skippable by artifact existence.
    """

    # A2
    s_prepare = Step(
        key="A2_prepare",
        title="Prepare data (merge + schema + time split)",
        outputs=[
            _p("data", "processed", "train.parquet"),
            _p("data", "processed", "val.parquet"),
            _p("data", "processed", "test.parquet"),
            _p("data", "splits", "split_info.json"),
            _p("artifacts", "evaluation", "columns.json"),
        ],
        module="scripts.00_prepare_data",
        module_args=[],
    )

    # A3
    s_tabular = Step(
        key="A3_tabular",
        title="Train tabular (LightGBM) + preds",
        outputs=[
            _p("artifacts", "tabular", "model.pkl"),
            _p("artifacts", "evaluation", "tabular_metrics.json"),
            _p("artifacts", "evaluation", "val_pred_tabular.parquet"),
            _p("artifacts", "evaluation", "test_pred_tabular.parquet"),
            _p("artifacts", "evaluation", "pr_curve_tabular.png"),
            _p("artifacts", "evaluation", "tabular_feature_spec.json"),
        ],
        module="scripts.01_train_tabular",
        module_args=[],
    )

    # A4.1
    s_build_graph = Step(
        key="A4_build_graph",
        title="Build graph artifacts (node_map + edges)",
        outputs=[
            _p("artifacts", "graph", "node_map.parquet"),
            _p("artifacts", "graph", "edges.parquet"),
            _p("artifacts", "graph", "graph_info.json"),
        ],
        module="scripts.02_build_graph",
        module_args=[],
    )

    # A4.2
    s_make_graph_data = Step(
        key="A4_make_graph_data",
        title="Convert to PyG HeteroData",
        outputs=[
            _p("artifacts", "graph", "graph_data.pt"),
            _p("artifacts", "graph", "tx_index.parquet"),
        ],
        module="scripts.03_make_graph_data",
        module_args=[],
    )

    # A5
    s_train_gnn = Step(
        key="A5_train_gnn",
        title="Train GNN (internal) + preds",
        outputs=[
            _p("artifacts", "graph", "gnn_model.pt"),
            _p("artifacts", "evaluation", "gnn_metrics.json"),
            _p("artifacts", "evaluation", "val_pred_gnn.parquet"),
            _p("artifacts", "evaluation", "test_pred_gnn.parquet"),
            _p("artifacts", "evaluation", "pr_curve_gnn.png"),
            _p("artifacts", "graph", "tx_scaler.json"),
        ],
        module="scripts.04_train_gnn",
        module_args=[],
    )

    # A6 (internal fusion) — полезно для ablation в отчёте, но можно скипнуть
    s_fusion_internal = Step(
        key="A6_fusion_internal",
        title="Train fusion internal (ablation/debug)",
        outputs=[
            _p("artifacts", "fusion", "fusion.pkl"),
            _p("artifacts", "evaluation", "fusion_metrics_internal.json"),
            _p("artifacts", "evaluation", "pr_curve_fusion_internal.png"),
            _p("artifacts", "evaluation", "val_pred_fusion_internal.parquet"),
            _p("artifacts", "evaluation", "test_pred_fusion_internal.parquet"),
        ],
        module="scripts.05_train_fusion",
        module_args=[],
    )

    # A9 external VAL
    s_gnn_external_val = Step(
        key="A9_gnn_external_val",
        title="Inductive GNN predict (external VAL)",
        outputs=[
            _p("artifacts", "evaluation", "val_pred_gnn_external.parquet"),
            _p("artifacts", "evaluation", "gnn_external_metrics_val.json"),
        ],
        module="scripts.08_predict_gnn_external",
        module_args=["--split", "val"],
    )

    # A9 external TEST
    s_gnn_external_test = Step(
        key="A9_gnn_external_test",
        title="Inductive GNN predict (external TEST)",
        outputs=[
            _p("artifacts", "evaluation", "test_pred_gnn_external.parquet"),
            _p("artifacts", "evaluation", "gnn_external_metrics_test.json"),
        ],
        module="scripts.08_predict_gnn_external",
        module_args=["--split", "test"],
    )

    # A9.3 calibration (temperature scaling)
    s_gnn_calibrate = Step(
        key="A9_calibrate_gnn",
        title="Calibrate external GNN (temperature scaling)",
        outputs=[
            _p("artifacts", "graph", "gnn_temperature.json"),
            _p("artifacts", "evaluation", "val_pred_gnn_external_calibrated.parquet"),
            _p("artifacts", "evaluation", "test_pred_gnn_external_calibrated.parquet"),
        ],
        module="scripts.09_calibrate_gnn",
        module_args=[],
    )

    # A10 fusion external (honest)
    s_fusion_external = Step(
        key="A10_fusion_external",
        title="Train fusion external (honest) + preds",
        outputs=[
            _p("artifacts", "fusion", "fusion_external.pkl"),
            _p("artifacts", "evaluation", "fusion_metrics_external.json"),
            _p("artifacts", "evaluation", "pr_curve_fusion_external.png"),
            _p("artifacts", "evaluation", "val_pred_fusion_external.parquet"),
            _p("artifacts", "evaluation", "test_pred_fusion_external.parquet"),
        ],
        module="scripts.10_train_fusion_external",
        module_args=[],
    )

    # A7 evaluate (thresholds/zones/cost) — запускаем, если нет fusion_external thresholds (или tabular thresholds)
    s_evaluate = Step(
        key="A7_evaluate",
        title="Evaluate + thresholds + decision zones + cost",
        outputs=[
            _p("artifacts", "thresholds", "thresholds_tabular.json"),
            _p("artifacts", "evaluation", "decision_zones_tabular_val.csv"),
            _p("artifacts", "evaluation", "decision_zones_tabular_test.csv"),
            _p("artifacts", "evaluation", "zone_share_tabular_val.png"),
            _p("artifacts", "evaluation", "zone_share_tabular_test.png"),
            _p("artifacts", "evaluation", "cost_tabular_test.json"),

            _p("artifacts", "thresholds", "thresholds_fusion_external.json"),
            _p("artifacts", "evaluation", "decision_zones_fusion_external_val.csv"),
            _p("artifacts", "evaluation", "decision_zones_fusion_external_test.csv"),
            _p("artifacts", "evaluation", "zone_share_fusion_external_val.png"),
            _p("artifacts", "evaluation", "zone_share_fusion_external_test.png"),
            _p("artifacts", "evaluation", "cost_fusion_external_test.json"),
        ],
        module="scripts.06_evaluate",
        module_args=[],
    )

    s_graph_viz = Step(
        key="A11_graph_viz",
        title="Build interactive graph viz (pyvis HTML)",
        outputs=[_p("reports", "assets", "graph.html")],
        module="scripts.16_build_graph_viz",
        module_args=[],
    )


    # A11 report
    s_report = Step(
        key="A11_auto_report",
        title="Auto report (HTML + tables + assets)",
        outputs=[
            _p("reports", "report.html"),
            _p("reports", "tables", "model_comparison.csv"),
        ],
        module="scripts.11_auto_report",
        module_args=[],
    )

    steps: List[Step] = [s_prepare, s_tabular, s_build_graph, s_make_graph_data]

    if not skip_gnn:
        steps.append(s_train_gnn)
        if not skip_internal_fusion:
            steps.append(s_fusion_internal)

    if not skip_external:
        # external inference logically requires trained GNN + node_map/graph_data + tabular preds
        steps.extend([s_gnn_external_val, s_gnn_external_test, s_gnn_calibrate, s_fusion_external])

    # evaluate should run after you have models/preds
    steps.append(s_evaluate)
    if not skip_report:
        steps.append(s_graph_viz)
        steps.append(s_report)

    return steps


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VKR Fraud System — One-click runner (A12)")
    parser.add_argument("--force", action="store_true", help="Re-run steps even if artifacts exist")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip internal GNN training step (A5)")
    parser.add_argument("--skip-external", action="store_true", help="Skip external inductive steps (A9/A10)")
    parser.add_argument("--skip-internal-fusion", action="store_true", help="Skip internal fusion (A6)")
    parser.add_argument("--skip-report", action="store_true", help="Skip auto report (A11)")

    parser.add_argument("--from-step", type=str, default=None, help="Start from step key (inclusive)")
    parser.add_argument("--to-step", type=str, default=None, help="Stop at step key (inclusive)")
    parser.add_argument("--shap", action="store_true", help="Run optional SHAP step (B1) if shap is installed")
    parser.add_argument("--start-api", action="store_true",
                        help="Start FastAPI service after pipeline (B2). Blocks terminal.")
    args = parser.parse_args()

    steps = build_steps(
        skip_gnn=args.skip_gnn,
        skip_external=args.skip_external,
        skip_internal_fusion=args.skip_internal_fusion,
        skip_report=args.skip_report,
    )

    s_shap = Step(
        key="B1_shap",
        title="SHAP for tabular (global plots + tables)",
        outputs=[
            _p("reports", "assets", "shap_summary_bar.png"),
            _p("reports", "tables", "shap_top_features.csv"),
        ],
        module="scripts.12_shap_tabular",
        module_args=[],
    )
    if args.shap:
        # Вставляем SHAP перед отчётом (если отчёт в списке)
        inserted = False
        for idx, st in enumerate(steps):
            if st.key == "A11_auto_report":
                steps.insert(idx, s_shap)
                inserted = True
                break
        if not inserted:
            steps.append(s_shap)

    # Slice by from/to
    if args.from_step:
        keys = [s.key for s in steps]
        if args.from_step not in keys:
            raise ValueError(f"--from-step unknown: {args.from_step}. Known: {keys}")
        steps = steps[keys.index(args.from_step):]

    if args.to_step:
        keys = [s.key for s in steps]
        if args.to_step not in keys:
            raise ValueError(f"--to-step unknown: {args.to_step}. Known: {keys}")
        steps = steps[: keys.index(args.to_step) + 1]

    print("VKR Fraud System: one-click runner (A12)")
    print(f"force={args.force}, skip_gnn={args.skip_gnn}, skip_external={args.skip_external}, "
          f"skip_internal_fusion={args.skip_internal_fusion}, skip_report={args.skip_report}")

    for step in steps:
        need = _should_run(step, force=args.force)
        status = "RUN" if need else "SKIP"
        print(f"\n[{status}] {step.key}: {step.title}")
        if not need:
            continue
        _run_module(step.module, step.module_args)

    # final check
    report_path = _p("reports", "report.html")
    if report_path.exists():
        print(f"\n[DONE] Report: {report_path}")
    else:
        print("\n[DONE] Pipeline finished (report not generated).")

    if args.start_api:
        _run_module("scripts.13_serve_api", [])


if __name__ == "__main__":
    main()