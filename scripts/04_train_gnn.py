from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader

from fraud_system.models.gnn import HeteroSAGE, GNNConfig
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, pr_curve_points, roc_auc
from fraud_system.evaluation.plots import plot_pr_curve


# -----------------------
# Helpers
# -----------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _select_tx_features(df: pd.DataFrame, max_features: int = 64) -> List[str]:
    """Select numeric transaction features (stable and CPU-friendly)."""
    exclude = {"transaction_id", "time", "target"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(num_cols)[:max_features]


def _impute_and_scale(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Median impute NaNs and standardize features."""
    X = X.astype(np.float32, copy=True)

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    scaler_info = {
        "median_nan_to_0": True,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return X, scaler_info


def _build_loaders(
    data,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    num_neighbors: Dict[Tuple[str, str, str], List[int]],
    batch_size_train: int,
    batch_size_eval: int,
):
    """Create NeighborLoaders for train, train_pred, val, test."""
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_train, dtype=torch.long)),
        batch_size=batch_size_train,
        shuffle=True,
    )

    train_pred_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_train, dtype=torch.long)),
        batch_size=batch_size_eval,
        shuffle=False,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_val, dtype=torch.long)),
        batch_size=batch_size_eval,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_test, dtype=torch.long)),
        batch_size=batch_size_eval,
        shuffle=False,
    )

    return train_loader, train_pred_loader, val_loader, test_loader


def _eval(model: torch.nn.Module, loader: NeighborLoader, device: torch.device):
    """Evaluate model on loader, return metrics + arrays."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)

            # Only the first batch_size nodes correspond to the input nodes of this loader batch
            bs = int(batch["transaction"].batch_size)
            logits = logits[:bs]
            yb = batch["transaction"].y[:bs].float()

            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(yb.cpu().numpy())
            ps.append(prob)

    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)

    metrics = {
        "logloss": binary_logloss(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
    }
    return metrics, y_true, y_score


# -----------------------
# Main
# -----------------------

def main() -> None:
    device = torch.device("cpu")

    # Paths
    graph_dir = Path("artifacts/graph")
    eval_dir = Path("artifacts/evaluation")
    _ensure_dir(eval_dir)

    graph_path = graph_dir / "graph_data.pt"
    tx_index_path = graph_dir / "tx_index.parquet"
    train_path = Path("data/processed") / "train.parquet"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph_data.pt: {graph_path} (run scripts.03_make_graph_data)")
    if not tx_index_path.exists():
        raise FileNotFoundError(f"Missing tx_index.parquet: {tx_index_path} (run scripts.03_make_graph_data)")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.parquet: {train_path} (run scripts.00_prepare_data)")

    # Load artifacts
    data = torch.load(graph_path, weights_only=False)
    tx_index = pd.read_parquet(tx_index_path)  # transaction_id -> tx_node_id

    df = pd.read_parquet(train_path)

    # Select features
    max_num_features = 64
    feat_cols = _select_tx_features(df, max_features=max_num_features)

    df = df[["transaction_id", "time", "target"] + feat_cols].copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["time"] = df["time"].astype("int64")
    df["target"] = df["target"].astype("int8")

    # Align to tx_node_id
    merged = tx_index.merge(df, on="transaction_id", how="left")

    # Critical fields must be present
    critical = ["time", "target"]
    bad = merged[critical].isna().any(axis=1)
    bad_rows = int(bad.sum())
    if bad_rows > 0:
        raise ValueError(f"[A5] Found {bad_rows} rows with NaN in critical fields {critical} after merge")

    nan_feat_rate = float(merged.isna().mean().mean())
    print(f"[A5] Overall NaN rate in merged table: {nan_feat_rate:.4f} (ok, will impute)")

    # Prepare X, y, time
    X_raw = merged[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X, scaler_info = _impute_and_scale(X_raw)

    y = merged["target"].to_numpy(dtype=np.int64)
    times = merged["time"].to_numpy(dtype=np.int64)

    # Put features/labels into HeteroData
    data["transaction"].x = torch.tensor(X, dtype=torch.float32)
    data["transaction"].y = torch.tensor(y, dtype=torch.long)

    # For entity nodes, pass node indices as x (for embedding lookup)
    for ntype in data.node_types:
        if ntype == "transaction":
            continue
        data[ntype].x = torch.arange(int(data[ntype].num_nodes), dtype=torch.long)

    # Time-based split inside train-graph: 60/20/20
    order = np.argsort(times)
    n = len(order)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    idx_train = order[:n_train]
    idx_val = order[n_train:n_train + n_val]
    idx_test = order[n_train + n_val:]

    split_info = {
        "n_total": int(n),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "time_min": int(times.min()),
        "time_max": int(times.max()),
        "time_train_end": int(times[idx_train].max()),
        "time_val_end": int(times[idx_val].max()),
        "feat_cols": feat_cols,
        "max_num_features": int(max_num_features),
    }

    with open(graph_dir / "gnn_split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # Sampling parameters
    num_neighbors = {etype: [15, 10] for etype in data.edge_types}  # 2 layers
    batch_size_train = 2048
    batch_size_eval = 4096

    train_loader, train_pred_loader, val_loader, test_loader = _build_loaders(
        data=data,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        num_neighbors=num_neighbors,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
    )

    # Model config
    cfg = GNNConfig(hidden_dim=64, num_layers=2, dropout=0.2, lr=1e-3, weight_decay=1e-4)
    embed_dim = 64

    num_nodes_by_type = {nt: int(data[nt].num_nodes) for nt in data.node_types}
    model = HeteroSAGE(
        metadata=data.metadata(),
        num_nodes_by_type=num_nodes_by_type,
        tx_in_dim=int(data["transaction"].x.size(1)),
        cfg=cfg,
        embed_dim=embed_dim,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop with early stopping
    best_val = 1e9
    best_state = None
    best_epoch = None
    stopped_epoch = None

    patience = 5
    bad = 0
    max_epochs = 15

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()

            logits = model(batch.x_dict, batch.edge_index_dict)
            bs = int(batch["transaction"].batch_size)
            logits = logits[:bs]
            yb = batch["transaction"].y[:bs].float()

            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics, _, _ = _eval(model, val_loader, device=device)

        print(
            f"[A5][{epoch}] train_loss={train_loss:.6f} "
            f"val_logloss={val_metrics['logloss']:.6f} val_pr_auc={val_metrics['pr_auc']:.6f}"
        )

        # early stopping by val logloss
        if val_metrics["logloss"] < best_val - 1e-5:
            best_val = float(val_metrics["logloss"])
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                stopped_epoch = epoch
                print(f"[A5] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval on train/val/test (holdout inside train-graph)
    train_metrics, y_tr, p_tr = _eval(model, train_pred_loader, device=device)
    val_metrics, y_val, p_val = _eval(model, val_loader, device=device)
    test_metrics, y_test, p_test = _eval(model, test_loader, device=device)

    print(f"[A5] TRAIN: logloss={train_metrics['logloss']:.6f}, pr_auc={train_metrics['pr_auc']:.6f}, roc_auc={train_metrics['roc_auc']:.6f}")
    print(f"[A5] VAL:   logloss={val_metrics['logloss']:.6f}, pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}")
    print(f"[A5] TEST:  logloss={test_metrics['logloss']:.6f}, pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}")

    # Save model
    out_model = graph_dir / "gnn_model.pt"
    torch.save(model.state_dict(), out_model)
    print(f"[A5] Saved model: {out_model}")

    # Save predictions aligned by transaction_id
    tx_local = tx_index.set_index("tx_node_id")["transaction_id"]

    def _pred_df(tx_node_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "tx_node_id": tx_node_ids.astype("int64"),
            "transaction_id": tx_local.loc[tx_node_ids].values.astype("int64"),
            "p_gnn": y_score.astype("float64"),
            "target": y_true.astype("int8"),
        })

    train_pred = _pred_df(idx_train, y_tr, p_tr)
    val_pred = _pred_df(idx_val, y_val, p_val)
    test_pred = _pred_df(idx_test, y_test, p_test)

    train_pred_path = eval_dir / "train_pred_gnn.parquet"
    val_pred_path = eval_dir / "val_pred_gnn.parquet"
    test_pred_path = eval_dir / "test_pred_gnn.parquet"

    train_pred.to_parquet(train_pred_path, index=False)
    val_pred.to_parquet(val_pred_path, index=False)
    test_pred.to_parquet(test_pred_path, index=False)

    print(f"[A5] Saved preds: {train_pred_path}")
    print(f"[A5] Saved preds: {val_pred_path}")
    print(f"[A5] Saved preds: {test_pred_path}")

    # Save metrics + config + sampler + training + split
    out = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "split": split_info,
        "config": {
            **asdict(cfg),
            "embed_dim": int(embed_dim),
        },
        "sampler": {
            "num_neighbors": {str(k): v for k, v in num_neighbors.items()},
            "batch_size_train": int(batch_size_train),
            "batch_size_eval": int(batch_size_eval),
        },
        "training": {
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "best_val_logloss": float(best_val),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
        },
        "preprocessing": {
            "impute": "median",
            "scale": "standardize",
            "scaler_info_saved": False,
        },
    }

    metrics_path = eval_dir / "gnn_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    precision, recall, _ = pr_curve_points(y_val, p_val)
    pr_path = eval_dir / "pr_curve_gnn.png"
    plot_pr_curve(precision, recall, str(pr_path), title="PR Curve (GNN, VAL)")

    print(f"[A5] Saved metrics: {metrics_path}")
    print(f"[A5] Saved PR curve: {pr_path}")
    print("[A5] Done.")


if __name__ == "__main__":
    main()