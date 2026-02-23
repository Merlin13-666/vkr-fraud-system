from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader

from fraud_system.models.gnn import HeteroSAGE, GNNConfig
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, pr_curve_points, roc_auc
from fraud_system.evaluation.plots import plot_pr_curve


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _select_tx_features(df: pd.DataFrame, max_features: int = 64) -> List[str]:
    # exclude ids/labels/time and non-numeric
    exclude = {"transaction_id", "time", "target"}
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    # stable ordering
    num_cols = sorted(num_cols)
    return num_cols[:max_features]


def main() -> None:
    device = torch.device("cpu")

    graph_dir = Path("artifacts/graph")
    eval_dir = Path("artifacts/evaluation")
    _ensure_dir(eval_dir)

    data = torch.load(graph_dir / "graph_data.pt", weights_only=False)
    tx_index = pd.read_parquet(graph_dir / "tx_index.parquet")  # transaction_id -> tx_node_id

    # Load train.parquet used for graph building (same period)
    df = pd.read_parquet(Path("data/processed") / "train.parquet")

    # Build mapping transaction_id -> target/time/features
    df = df[["transaction_id", "time", "target"] + _select_tx_features(df)].copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["time"] = df["time"].astype("int64")
    df["target"] = df["target"].astype("int8")

    # Align df rows to tx_node_id (graph order)
    merged = tx_index.merge(df, on="transaction_id", how="left")

    # проверяем только критичные поля (фичи могут иметь NaN — мы их импутим)
    critical = ["time", "target"]
    bad = merged[critical].isna().any(axis=1)
    bad_rows = int(bad.sum())
    if bad_rows > 0:
        raise ValueError(f"[A5] Found {bad_rows} rows with NaN in critical fields {critical} after merge")

    nan_feat_rate = float(merged.isna().mean().mean())
    print(f"[A5] Overall NaN rate in merged table: {nan_feat_rate:.4f} (ok, will impute)")

    feat_cols = [c for c in merged.columns if c not in ("transaction_id", "tx_node_id", "time", "target")]

    # Simple impute (median) + standardize
    X = merged[feat_cols].to_numpy(dtype=np.float32, copy=True)

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    y = merged["target"].to_numpy(dtype=np.int64)
    times = merged["time"].to_numpy(dtype=np.int64)

    # Put transaction features into HeteroData
    data["transaction"].x = torch.tensor(X, dtype=torch.float32)
    data["transaction"].y = torch.tensor(y, dtype=torch.long)

    # For entity nodes, we pass their node indices as x (so model can embed them)
    for ntype in data.node_types:
        if ntype == "transaction":
            continue
        data[ntype].x = torch.arange(data[ntype].num_nodes, dtype=torch.long)

    # Time-based split inside train graph: 60/20/20
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
    }
    with open(graph_dir / "gnn_split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # Neighbor sampling loader (hetero)
    num_neighbors = {etype: [15, 10] for etype in data.edge_types}  # 2 layers
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_train, dtype=torch.long)),
        batch_size=2048,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_val, dtype=torch.long)),
        batch_size=4096,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("transaction", torch.tensor(idx_test, dtype=torch.long)),
        batch_size=4096,
        shuffle=False,
    )

    cfg = GNNConfig(hidden_dim=64, num_layers=2, dropout=0.2, lr=1e-3, weight_decay=1e-4)
    num_nodes_by_type = {nt: int(data[nt].num_nodes) for nt in data.node_types}

    model = HeteroSAGE(
        metadata=data.metadata(),
        num_nodes_by_type=num_nodes_by_type,
        tx_in_dim=data["transaction"].x.size(1),
        cfg=cfg,
        embed_dim=64,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def _eval(loader) -> Dict[str, float]:
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x_dict, batch.edge_index_dict)

                # only predictions for input transaction nodes in the batch
                # NeighborLoader stores them in batch['transaction'].batch_size
                bs = batch["transaction"].batch_size
                logits = logits[:bs]
                yb = batch["transaction"].y[:bs].float()

                prob = torch.sigmoid(logits).cpu().numpy()
                ys.append(yb.cpu().numpy())
                ps.append(prob)

        y_true = np.concatenate(ys)
        y_score = np.concatenate(ps)
        return {
            "logloss": binary_logloss(y_true, y_score),
            "pr_auc": pr_auc(y_true, y_score),
            "roc_auc": roc_auc(y_true, y_score),
        }, y_true, y_score

    best_val = 1e9
    best_state = None
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
            bs = batch["transaction"].batch_size
            logits = logits[:bs]
            yb = batch["transaction"].y[:bs].float()

            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_metrics, _, _ = _eval(val_loader)
        train_loss = float(np.mean(losses))

        print(f"[A5][{epoch}] train_loss={train_loss:.6f} "
              f"val_logloss={val_metrics['logloss']:.6f} val_pr_auc={val_metrics['pr_auc']:.6f}")

        # early stopping by val logloss
        if val_metrics["logloss"] < best_val - 1e-5:
            best_val = val_metrics["logloss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[A5] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    val_metrics, y_val, p_val = _eval(val_loader)
    test_metrics, y_test, p_test = _eval(test_loader)

    print(f"[A5] VAL:  logloss={val_metrics['logloss']:.6f}, pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}")
    print(f"[A5] TEST: logloss={test_metrics['logloss']:.6f}, pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}")

    # Save model
    out_model = graph_dir / "gnn_model.pt"
    torch.save(model.state_dict(), out_model)
    print(f"[A5] Saved model: {out_model}")

    # Save preds (transaction_id aligned)
    # Take the same ordering as idx_val/idx_test refer to tx_node_id indexes
    tx_local = tx_index.set_index("tx_node_id")["transaction_id"]

    val_pred = pd.DataFrame({
        "tx_node_id": idx_val,
        "transaction_id": tx_local.loc[idx_val].values.astype("int64"),
        "p_gnn": p_val.astype("float64"),
        "target": y_val.astype("int8"),
    })
    test_pred = pd.DataFrame({
        "tx_node_id": idx_test,
        "transaction_id": tx_local.loc[idx_test].values.astype("int64"),
        "p_gnn": p_test.astype("float64"),
        "target": y_test.astype("int8"),
    })

    val_pred.to_parquet(eval_dir / "val_pred_gnn.parquet", index=False)
    test_pred.to_parquet(eval_dir / "test_pred_gnn.parquet", index=False)

    with open(eval_dir / "gnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics, "split": split_info}, f, ensure_ascii=False, indent=2)

    precision, recall, _ = pr_curve_points(y_val, p_val)
    plot_pr_curve(precision, recall, str(eval_dir / "pr_curve_gnn.png"), title="PR Curve (GNN, VAL)")

    print(f"[A5] Saved preds: {eval_dir / 'val_pred_gnn.parquet'}")
    print(f"[A5] Saved preds: {eval_dir / 'test_pred_gnn.parquet'}")
    print(f"[A5] Saved metrics: {eval_dir / 'gnn_metrics.json'}")
    print(f"[A5] Saved PR curve: {eval_dir / 'pr_curve_gnn.png'}")
    print("[A5] Done.")


if __name__ == "__main__":
    main()