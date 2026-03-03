from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict


def _run_module(module: str, args: Optional[List[str]] = None) -> None:
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
    return True if force else (not _exists_all(step.outputs))


def _resolve_graph_pred_path(user_path: str) -> str:
    p = Path(user_path)
    if p.exists():
        return user_path
    fallback = Path("artifacts/evaluation/test_pred_fusion_external.parquet")
    if fallback.exists():
        return str(fallback).replace("\\", "/")
    return user_path


def build_steps(
    graph_mode: str,
    graph_tx_id: Optional[int],
    graph_entity_type: Optional[str],
    graph_entity_value: Optional[str],
    graph_pred_path: str,
    graph_hops: int,
    graph_max_nodes: int,
    graph_max_edges: int,
) -> List[Step]:
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

    s_make_graph_data = Step(
        key="A4_make_graph_data",
        title="Convert to PyG HeteroData",
        outputs=[_p("artifacts", "graph", "graph_data.pt"), _p("artifacts", "graph", "tx_index.parquet")],
        module="scripts.03_make_graph_data",
        module_args=[],
    )

    s_graph_stats = Step(
        key="A4_graph_stats",
        title="Graph topology stats (A4.3)",
        outputs=[
            _p("artifacts", "graph", "graph_stats.json"),
            _p("artifacts", "evaluation", "graph_degree_tx_hist.png"),
            _p("artifacts", "evaluation", "graph_edges_by_dst_type.png"),
            _p("artifacts", "evaluation", "graph_top_entities_degree.png"),
            _p("artifacts", "evaluation", "graph_components_hist.png"),
        ],
        module="scripts.17_graph_stats",
        module_args=[],
    )

    s_graph_metrics_baseline = Step(
        key="A6_1_graph_metrics_baseline",
        title="Graph-metrics baseline (A6.1)",
        outputs=[
            _p("artifacts", "evaluation", "graph_metrics_baseline_metrics.json"),
            _p("artifacts", "evaluation", "pr_curve_graph_metrics.png"),
            _p("artifacts", "evaluation", "val_pred_graph_metrics.parquet"),
            _p("artifacts", "evaluation", "test_pred_graph_metrics.parquet"),
            _p("artifacts", "evaluation", "graph_metrics_feature_importance.png"),
        ],
        module="scripts.18_train_graph_metrics_baseline",
        module_args=[],
    )

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

    s_gnn_external_val = Step(
        key="A9_gnn_external_val",
        title="Inductive GNN predict (external VAL)",
        outputs=[_p("artifacts", "evaluation", "val_pred_gnn_external.parquet"), _p("artifacts", "evaluation", "gnn_external_metrics_val.json")],
        module="scripts.08_predict_gnn_external",
        module_args=["--split", "val"],
    )

    s_gnn_external_test = Step(
        key="A9_gnn_external_test",
        title="Inductive GNN predict (external TEST)",
        outputs=[_p("artifacts", "evaluation", "test_pred_gnn_external.parquet"), _p("artifacts", "evaluation", "gnn_external_metrics_test.json")],
        module="scripts.08_predict_gnn_external",
        module_args=["--split", "test"],
    )

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

    # graph viz args
    resolved_pred_path = _resolve_graph_pred_path(graph_pred_path)
    gv_args: List[str] = ["--mode", graph_mode, "--hops", str(graph_hops), "--max-nodes", str(graph_max_nodes), "--max-edges", str(graph_max_edges)]
    if graph_mode == "ego_tx":
        if graph_tx_id is not None:
            gv_args += ["--tx-id", str(graph_tx_id)]
        else:
            gv_args += ["--auto-pick", "--pred-path", resolved_pred_path]
    else:
        if not graph_entity_type or not graph_entity_value:
            raise ValueError("graph_mode=ego_entity requires --graph-entity-type and --graph-entity-value")
        gv_args += ["--entity-type", graph_entity_type, "--entity-value", graph_entity_value]

    s_graph_viz = Step(
        key="A11_graph_viz",
        title="Build interactive graph viz (pyvis HTML)",
        outputs=[_p("reports", "assets", "graph.html")],
        module="scripts.16_build_graph_viz",
        module_args=gv_args,
    )

    s_report = Step(
        key="A11_auto_report",
        title="Auto report (HTML + tables + assets)",
        outputs=[_p("reports", "report.html"), _p("reports", "tables", "model_comparison.csv")],
        module="scripts.11_auto_report",
        module_args=[],
    )

    return [
        s_prepare,
        s_tabular,
        s_build_graph,
        s_make_graph_data,
        s_graph_stats,
        s_graph_metrics_baseline,
        s_train_gnn,
        s_fusion_internal,
        s_gnn_external_val,
        s_gnn_external_test,
        s_gnn_calibrate,
        s_fusion_external,
        s_evaluate,
        s_graph_viz,
        s_report,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="VKR Fraud System — One-click runner (A12)")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--only-report", action="store_true", help="Run only A11_graph_viz + A11_auto_report (expects artifacts exist)")
    parser.add_argument("--only-graph-metrics", action="store_true", help="Run only A6.1 graph-metrics baseline (expects A4 artifacts exist)")

    parser.add_argument("--from-step", type=str, default=None)
    parser.add_argument("--to-step", type=str, default=None)

    parser.add_argument("--graph-mode", choices=["ego_tx", "ego_entity"], default="ego_tx")
    parser.add_argument("--graph-tx-id", type=int, default=None)
    parser.add_argument("--graph-entity-type", type=str, default=None)
    parser.add_argument("--graph-entity-value", type=str, default=None)
    parser.add_argument("--graph-pred-path", type=str, default="artifacts/evaluation/val_pred_fusion_internal.parquet")
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--graph-max-nodes", type=int, default=1500)
    parser.add_argument("--graph-max-edges", type=int, default=3000)

    args = parser.parse_args()

    steps = build_steps(
        graph_mode=args.graph_mode,
        graph_tx_id=args.graph_tx_id,
        graph_entity_type=args.graph_entity_type,
        graph_entity_value=args.graph_entity_value,
        graph_pred_path=args.graph_pred_path,
        graph_hops=args.graph_hops,
        graph_max_nodes=args.graph_max_nodes,
        graph_max_edges=args.graph_max_edges,
    )

    if args.only_graph_metrics:
        steps = [s for s in steps if s.key == "A6_1_graph_metrics_baseline"]

    if args.only_report:
        steps = [s for s in steps if s.key in {"A11_graph_viz", "A11_auto_report"}]

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
    print(f"force={args.force}")
    print(f"graph: mode={args.graph_mode}, tx_id={args.graph_tx_id}, pred_path={args.graph_pred_path}, hops={args.graph_hops}")

    results: Dict[str, str] = {}

    for step in steps:
        need = _should_run(step, force=args.force)
        status = "RUN" if need else "SKIP"
        print(f"\n[{status}] {step.key}: {step.title}")
        if not need:
            results[step.key] = "SKIPPED"
            continue
        _run_module(step.module, step.module_args)
        results[step.key] = "OK"

    print("\n[SUMMARY]")
    for k in [s.key for s in steps]:
        print(f"  {k}: {results.get(k, 'N/A')}")

    report_path = _p("reports", "report.html")
    if report_path.exists():
        print(f"\n[DONE] Report: {report_path}")
    else:
        print("\n[DONE] Pipeline finished (report not generated).")


if __name__ == "__main__":
    main()