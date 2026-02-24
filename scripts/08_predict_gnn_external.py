from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader

from fraud_system.models.gnn import HeteroSAGE, GNNConfig
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, roc_auc, pr_curve_points
from fraud_system.evaluation.plots import plot_pr_curve
from fraud_system.graph.inductive import load_node_map, load_tx_scaler, build_inductive_heterodata


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_state_with_unk_padding(model: torch.nn.Module, state: dict) -> None:
    """
    If current model has +1 node for some entity types (UNK), but checkpoint was trained without UNK,
    pad embedding weights by 1 row so load_state_dict works.
    """
    model_state = model.state_dict()
    patched = dict(state)

    for k, w_new in model_state.items():
        if k not in patched:
            continue

        w_old = patched[k]

        # patch only 2D embedding matrices (num_nodes, embed_dim)
        if not (isinstance(w_old, torch.Tensor) and isinstance(w_new, torch.Tensor)):
            continue
        if w_old.ndim != 2 or w_new.ndim != 2:
            continue

        # If checkpoint has N and model expects N+1: append UNK row
        if w_new.shape[0] == w_old.shape[0] + 1 and w_new.shape[1] == w_old.shape[1]:
            unk_row = w_old.mean(dim=0, keepdim=True)  # stable init
            patched[k] = torch.cat([w_old, unk_row], dim=0)
            continue

        # If shapes match: ok
        if w_new.shape == w_old.shape:
            continue

        # Any other mismatch is a real error
        raise RuntimeError(f"State mismatch for {k}: ckpt {tuple(w_old.shape)} vs model {tuple(w_new.shape)}")

    model.load_state_dict(patched, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    graph_dir = Path("artifacts/graph")
    eval_dir = Path("artifacts/evaluation")
    processed_dir = Path("data/processed")
    _ensure_dir(eval_dir)

    # Load external split
    df_ext = pd.read_parquet(processed_dir / f"{args.split}.parquet")
    if "transaction_id" not in df_ext.columns or "target" not in df_ext.columns:
        raise ValueError("processed split must contain transaction_id and target")

    # Load node map + scaler
    node_map = load_node_map(graph_dir / "node_map.parquet")
    scaler = load_tx_scaler(graph_dir / "tx_scaler.json")
    feat_cols = scaler["feat_cols"]

    # Entity columns must match A4/A5 conventions (values are prefixed by source column)
    entity_cols: Dict[str, list] = {
        "card": ["card1"],
        "email": ["P_emaildomain", "R_emaildomain"],
        "addr": ["addr1", "addr2"],
        "device": ["DeviceInfo", "DeviceType"],
    }

    # Build inductive heterodata (transaction nodes are only external)
    data, tx_index, mapping_stats = build_inductive_heterodata(
        df_ext=df_ext,
        node_map_df=node_map,
        entity_cols=entity_cols,
        tx_feat_cols=feat_cols,
        scaler=scaler,
    )

    # Save mapping stats
    stats_path = eval_dir / f"gnn_external_mapping_{args.split}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(mapping_stats, f, ensure_ascii=False, indent=2)

    # Load trained model config/sampler from artifact (A5)
    gnn_metrics_path = eval_dir / "gnn_metrics.json"
    if not gnn_metrics_path.exists():
        raise FileNotFoundError(f"Missing {gnn_metrics_path}. Run A5 (04_train_gnn) first.")

    gnn_meta = _load_json(gnn_metrics_path)
    cfg_d = gnn_meta.get("config", {}) or {}
    sampler_d = gnn_meta.get("sampler", {}) or {}

    cfg = GNNConfig(
        hidden_dim=int(cfg_d.get("hidden_dim", 64)),
        num_layers=int(cfg_d.get("num_layers", 2)),
        dropout=float(cfg_d.get("dropout", 0.2)),
        lr=float(cfg_d.get("lr", 1e-3)),
        weight_decay=float(cfg_d.get("weight_decay", 1e-4)),
    )
    embed_dim = int(cfg_d.get("embed_dim", 64))
    batch_size_eval = int(sampler_d.get("batch_size_eval", 4096))

    # Model init must match inference graph sizes
    num_nodes_by_type = {nt: int(data[nt].num_nodes) for nt in data.node_types}

    model = HeteroSAGE(
        metadata=data.metadata(),
        num_nodes_by_type=num_nodes_by_type,
        tx_in_dim=data["transaction"].x.size(1),
        cfg=cfg,
        embed_dim=embed_dim,
    ).to(device)

    state = torch.load(graph_dir / "gnn_model.pt", map_location="cpu")
    _load_state_with_unk_padding(model, state)
    model.eval()

    # Loader for inference over ALL external tx nodes
    idx_all = torch.arange(data["transaction"].num_nodes, dtype=torch.long)
    num_neighbors = {etype: [15, 10] for etype in data.edge_types}

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", idx_all),
        batch_size=batch_size_eval,
        shuffle=False,
    )

    ps = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)
            bs = batch["transaction"].batch_size
            logits = logits[:bs]  # seed nodes first
            prob = torch.sigmoid(logits).cpu().numpy()
            ps.append(prob)

    p = np.concatenate(ps).astype("float64").reshape(-1)

    # tx_index ordering is exactly tx_node_id 0..N-1
    tx_index = tx_index.sort_values("tx_node_id").reset_index(drop=True)

    # Hard checks: p length must match number of external tx nodes
    if len(p) != len(tx_index):
        raise RuntimeError(f"Prediction length mismatch: p={len(p)} vs tx_index={len(tx_index)}")

    # Strict 1:1 labels alignment (protect against duplicates/misalignment)
    labels = df_ext[["transaction_id", "target"]].drop_duplicates("transaction_id")
    labels["transaction_id"] = labels["transaction_id"].astype("int64")
    labels["target"] = labels["target"].astype("int8")

    labels_aligned = tx_index.merge(labels, on="transaction_id", how="left", validate="1:1")
    if labels_aligned["target"].isna().any():
        miss = int(labels_aligned["target"].isna().sum())
        raise RuntimeError(f"Missing targets after alignment: {miss} rows. Check processed split integrity.")

    y = labels_aligned["target"].to_numpy(dtype=np.int8)

    # Clip probabilities for stable metrics
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # Diagnostics + baseline
    base_rate = float(y.mean())
    baseline_p = np.full_like(p, base_rate, dtype=np.float64)
    baseline_ll = float(binary_logloss(y, np.clip(baseline_p, 1e-6, 1 - 1e-6)))

    p_stats = {
        "p_min": float(np.min(p)),
        "p_max": float(np.max(p)),
        "p_mean": float(np.mean(p)),
        "p_q01": float(np.quantile(p, 0.01)),
        "p_q99": float(np.quantile(p, 0.99)),
    }

    print(f"[A9] base_rate={base_rate:.6f} baseline_logloss={baseline_ll:.6f}")
    print(
        f"[A9] p_stats: min={p_stats['p_min']:.6f} mean={p_stats['p_mean']:.6f} "
        f"q01={p_stats['p_q01']:.6f} q99={p_stats['p_q99']:.6f} max={p_stats['p_max']:.6f}"
    )

    out = pd.DataFrame(
        {
            "tx_node_id": tx_index["tx_node_id"].astype("int64"),
            "transaction_id": tx_index["transaction_id"].astype("int64"),
            "p_gnn_external": p,
            "target": y,
        }
    )

    out_path = eval_dir / f"{args.split}_pred_gnn_external.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[A9] Saved preds: {out_path}")

    metrics = {
        "logloss": float(binary_logloss(y, p)),
        "pr_auc": float(pr_auc(y, p)),
        "roc_auc": float(roc_auc(y, p)),
        "baseline_logloss": baseline_ll,
        "base_rate": base_rate,
        **p_stats,
        "split": args.split,
        "mapping_stats_path": str(stats_path),
        "model_config_used": cfg_d,
        "sampler_used": sampler_d,
    }

    metrics_path = eval_dir / f"gnn_external_metrics_{args.split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"[A9] {args.split.upper()}: logloss={metrics['logloss']:.6f}, "
        f"pr_auc={metrics['pr_auc']:.6f}, roc_auc={metrics['roc_auc']:.6f}"
    )
    print(f"[A9] Saved metrics: {metrics_path}")

    # PR curve on external VAL only
    if args.split == "val":
        precision, recall, _ = pr_curve_points(y, p)
        pr_path = eval_dir / "pr_curve_gnn_external_val.png"
        plot_pr_curve(precision, recall, str(pr_path), title="PR Curve (GNN, external VAL)")
        print(f"[A9] Saved PR curve: {pr_path}")

    print("[A9] Done.")


if __name__ == "__main__":
    main()