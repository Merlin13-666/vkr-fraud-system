from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader

from fraud_system.models.gnn import HeteroSAGE, GNNConfig
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, pr_curve_points, roc_auc
from fraud_system.evaluation.plots import plot_pr_curve


# =======================
# Helpers
# =======================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _dir_from_env(env_name: str, default: str) -> Path:
    v = os.environ.get(env_name, "").strip()
    return Path(v) if v else Path(default)

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_tx_features(df: pd.DataFrame, max_features: int = 64) -> List[str]:
    exclude = {"transaction_id", "time", "target"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(num_cols)[:max_features]


def _impute_and_scale(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = X.astype(np.float32, copy=True)

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(med, inds[1])

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std

    scaler_stats = {
        "median": med.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "median_nan_to_0": True,
        "eps": 1e-6,
    }
    return X, scaler_stats


def _build_loaders(
    data,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    num_neighbors: Dict[Tuple[str, str, str], List[int]],
    batch_size_train: int,
    batch_size_eval: int,
):
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
    model.eval()
    ys, ps, ls = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)

            bs = int(batch["transaction"].batch_size)
            logits = logits[:bs]
            yb = batch["transaction"].y[:bs].float()

            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(yb.cpu().numpy())
            ps.append(prob)
            ls.append(logits.cpu().numpy())

    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)
    y_logit = np.concatenate(ls)

    y_score = np.clip(y_score, 1e-6, 1 - 1e-6)

    metrics = {
        "logloss": float(binary_logloss(y_true, y_score)),
        "pr_auc": float(pr_auc(y_true, y_score)),
        "roc_auc": float(roc_auc(y_true, y_score)),
    }
    return metrics, y_true, y_score, y_logit


def _suffix(tag: Optional[str], out_suffix: Optional[str]) -> str:
    # Единая логика имени прогона:
    # 1) если out_suffix задан явно — используем его как “ключ”
    # 2) иначе если tag задан — используем tag
    # 3) иначе пусто (main run)
    if out_suffix:
        return out_suffix
    if tag:
        return tag
    return ""


def _name(base: str, suf: str, ext: str) -> str:
    # base like "gnn_metrics", ext like ".json"
    return f"{base}__{suf}{ext}" if suf else f"{base}{ext}"


# =======================
# Main
# =======================

def main() -> None:
    parser = argparse.ArgumentParser(description="A5 Train GNN (HeteroSAGE) on train graph (internal split)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--neighbors", type=str, default="15,10", help="e.g. '15,10'")
    parser.add_argument("--batch-size-train", type=int, default=2048)
    parser.add_argument("--batch-size-eval", type=int, default=4096)

    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)

    parser.add_argument("--max-num-features", type=int, default=64)

    # Для абляций:
    parser.add_argument("--tag", type=str, default=None, help="Optional tag, e.g. 'ablation'")
    parser.add_argument("--out-suffix", type=str, default=None, help="Optional explicit suffix, e.g. 'L2_H64_E64_N15-10'")

    # ВАЖНО: чтобы run_all не ломался — main run должен писать стандартные имена.
    # Для абляций мы пишем только суффиксные артефакты. Флаг позволяет переопределить.
    parser.add_argument("--write-default-artifacts", action="store_true", help="Also write default gnn_model.pt/gnn_metrics.json/...")

    args = parser.parse_args()

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[A5] CUDA requested but not available -> fallback to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    _set_seed(args.seed)

    graph_dir = _dir_from_env("FRAUD_GRAPH_DIR", "artifacts/graph")
    eval_dir = _dir_from_env("FRAUD_EVAL_DIR", "artifacts/evaluation")
    data_dir = Path("data/processed")

    _ensure_dir(graph_dir)
    _ensure_dir(eval_dir)

    graph_path = graph_dir / "graph_data.pt"
    tx_index_path = graph_dir / "tx_index.parquet"
    train_path = data_dir / "train.parquet"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing: {graph_path} (run scripts.03_make_graph_data)")
    if not tx_index_path.exists():
        raise FileNotFoundError(f"Missing: {tx_index_path} (run scripts.03_make_graph_data)")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path} (run scripts.00_prepare_data)")

    data = torch.load(graph_path, weights_only=False)
    tx_index = pd.read_parquet(tx_index_path)
    df = pd.read_parquet(train_path)

    feat_cols = _select_tx_features(df, max_features=int(args.max_num_features))

    df = df[["transaction_id", "time", "target"] + feat_cols].copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["time"] = df["time"].astype("int64")
    df["target"] = df["target"].astype("int8")

    merged = tx_index.merge(df, on="transaction_id", how="left")

    critical = ["time", "target"]
    bad = merged[critical].isna().any(axis=1)
    bad_rows = int(bad.sum())
    if bad_rows > 0:
        raise ValueError(f"[A5] Found {bad_rows} rows with NaN in critical fields {critical} after merge")

    nan_feat_rate = float(merged.isna().mean().mean())
    print(f"[A5] Overall NaN rate in merged table: {nan_feat_rate:.4f} (ok, will impute)")

    X_raw = merged[feat_cols].to_numpy(dtype=np.float32, copy=True)
    X, scaler_stats = _impute_and_scale(X_raw)

    y = merged["target"].to_numpy(dtype=np.int64)
    times = merged["time"].to_numpy(dtype=np.int64)

    data["transaction"].x = torch.tensor(X, dtype=torch.float32)
    data["transaction"].y = torch.tensor(y, dtype=torch.long)

    # entity features as ids
    for ntype in data.node_types:
        if ntype == "transaction":
            continue
        data[ntype].x = torch.arange(int(data[ntype].num_nodes), dtype=torch.long)

    # scaler saved (needed for inductive external)
    tx_scaler = {
        "feat_cols": feat_cols,
        "median": scaler_stats["median"],
        "mean": scaler_stats["mean"],
        "std": scaler_stats["std"],
        "median_nan_to_0": scaler_stats.get("median_nan_to_0", True),
        "eps": scaler_stats.get("eps", 1e-6),
        "max_num_features": int(args.max_num_features),
        "seed": int(args.seed),
        "source": "A5 train graph transaction features (median impute + standardize)",
    }
    tx_scaler_path = graph_dir / "tx_scaler.json"
    with open(tx_scaler_path, "w", encoding="utf-8") as f:
        json.dump(tx_scaler, f, ensure_ascii=False, indent=2)
    print(f"[A5] Saved tx scaler: {tx_scaler_path}")

    # internal time split (60/20/20)
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
        "max_num_features": int(args.max_num_features),
        "seed": int(args.seed),
    }
    with open(graph_dir / "gnn_split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # neighbors parse
    parts = [p.strip() for p in str(args.neighbors).split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("--neighbors is empty")
    neighbors_list = [int(x) for x in parts]

    num_neighbors = {etype: neighbors_list for etype in data.edge_types}

    # config
    cfg = GNNConfig(
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    embed_dim = int(args.embed_dim)

    # run suffix
    # если не задан out_suffix — делаем информативный автоматически (как у тебя в логах)
    auto_suffix = f"L{cfg.num_layers}_H{cfg.hidden_dim}_E{embed_dim}_N{'-'.join(map(str, neighbors_list))}"
    run_suffix = _suffix(args.tag, args.out_suffix) or auto_suffix

    print(
        f"[A5] run={run_suffix} device={device.type} "
        f"layers={cfg.num_layers} hidden={cfg.hidden_dim} embed={embed_dim} neighbors={neighbors_list}"
    )

    # loaders
    train_loader, train_pred_loader, val_loader, test_loader = _build_loaders(
        data=data,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        num_neighbors=num_neighbors,
        batch_size_train=int(args.batch_size_train),
        batch_size_eval=int(args.batch_size_eval),
    )

    # model
    num_nodes_by_type = {nt: int(data[nt].num_nodes) for nt in data.node_types}
    model = HeteroSAGE(
        metadata=data.metadata(),
        num_nodes_by_type=num_nodes_by_type,
        tx_in_dim=int(data["transaction"].x.size(1)),
        cfg=cfg,
        embed_dim=embed_dim,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # imbalance
    y_train = y[idx_train]
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight_value = float(neg / max(pos, 1.0))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"[A5] pos_weight={pos_weight_value:.6f} (pos={int(pos)}, neg={int(neg)})")

    best_val = 1e18
    best_state = None
    best_epoch = None
    stopped_epoch = None
    bad_epochs = 0

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        losses: List[float] = []

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
        val_metrics, _, _, _ = _eval(model, val_loader, device=device)

        print(
            f"[A5][{run_suffix}][{epoch}] train_loss={train_loss:.6f} "
            f"val_logloss={val_metrics['logloss']:.6f} val_pr_auc={val_metrics['pr_auc']:.6f}"
        )

        if val_metrics["logloss"] < best_val - float(args.early_stop_delta):
            best_val = float(val_metrics["logloss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                stopped_epoch = epoch
                print(f"[A5][{run_suffix}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, y_tr, p_tr, l_tr = _eval(model, train_pred_loader, device=device)
    val_metrics, y_val, p_val, l_val = _eval(model, val_loader, device=device)
    test_metrics, y_te, p_te, l_te = _eval(model, test_loader, device=device)

    print(f"[A5][{run_suffix}] TRAIN: logloss={train_metrics['logloss']:.6f}, pr_auc={train_metrics['pr_auc']:.6f}, roc_auc={train_metrics['roc_auc']:.6f}")
    print(f"[A5][{run_suffix}] VAL:   logloss={val_metrics['logloss']:.6f}, pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}")
    print(f"[A5][{run_suffix}] TEST:  logloss={test_metrics['logloss']:.6f}, pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}")

    # --- outputs (suffix + optional default)
    model_suffix_path = graph_dir / _name("gnn_model", run_suffix, ".pt")
    torch.save(model.state_dict(), model_suffix_path)
    print(f"[A5][{run_suffix}] Saved model: {model_suffix_path}")

    tx_local = tx_index.set_index("tx_node_id")["transaction_id"]

    def _pred_df(tx_node_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, y_logit: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tx_node_id": tx_node_ids.astype("int64"),
                "transaction_id": tx_local.loc[tx_node_ids].values.astype("int64"),
                "p_gnn": y_score.astype("float64"),
                "logit_gnn": y_logit.astype("float64"),
                "target": y_true.astype("int8"),
            }
        )

    train_pred = _pred_df(idx_train, y_tr, p_tr, l_tr)
    val_pred = _pred_df(idx_val, y_val, p_val, l_val)
    test_pred = _pred_df(idx_test, y_te, p_te, l_te)

    train_pred_path = eval_dir / _name("train_pred_gnn", run_suffix, ".parquet")
    val_pred_path = eval_dir / _name("val_pred_gnn", run_suffix, ".parquet")
    test_pred_path = eval_dir / _name("test_pred_gnn", run_suffix, ".parquet")

    train_pred.to_parquet(train_pred_path, index=False)
    val_pred.to_parquet(val_pred_path, index=False)
    test_pred.to_parquet(test_pred_path, index=False)

    print(f"[A5][{run_suffix}] Saved preds: {val_pred_path} (and train/test)")

    out = {
        "run_suffix": run_suffix,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "split": split_info,
        "config": {
            **asdict(cfg),
            "embed_dim": int(embed_dim),
            "seed": int(args.seed),
            "device": device.type,
        },
        "sampler": {
            "neighbors_list": neighbors_list,
            "num_neighbors": {str(k): v for k, v in num_neighbors.items()},
            "batch_size_train": int(args.batch_size_train),
            "batch_size_eval": int(args.batch_size_eval),
        },
        "training": {
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "early_stop_delta": float(args.early_stop_delta),
            "best_val_logloss": float(best_val),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
        },
        "preprocessing": {
            "impute": "median",
            "scale": "standardize",
            "tx_scaler_path": str(tx_scaler_path).replace("\\", "/"),
        },
        "class_balance": {
            "pos": int(pos),
            "neg": int(neg),
            "pos_weight": float(pos_weight_value),
        },
    }

    metrics_path = eval_dir / _name("gnn_metrics", run_suffix, ".json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    precision, recall, _ = pr_curve_points(y_val, p_val)
    pr_path = eval_dir / _name("pr_curve_gnn", run_suffix, ".png")
    plot_pr_curve(precision, recall, str(pr_path), title=f"PR Curve (GNN, VAL) — {run_suffix}")

    print(f"[A5][{run_suffix}] Saved metrics: {metrics_path}")
    print(f"[A5][{run_suffix}] Saved PR curve: {pr_path}")

    # --- default artifacts for pipeline compatibility
    # Пишем стандартные имена, если:
    # 1) явно попросили --write-default-artifacts
    # 2) это не “абляционный” прогон (tag != 'ablation')
    write_default = bool(args.write_default_artifacts) or (str(args.tag or "").lower() != "ablation")

    if write_default:
        torch.save(model.state_dict(), graph_dir / "gnn_model.pt")
        train_pred.to_parquet(eval_dir / "train_pred_gnn.parquet", index=False)
        val_pred.to_parquet(eval_dir / "val_pred_gnn.parquet", index=False)
        test_pred.to_parquet(eval_dir / "test_pred_gnn.parquet", index=False)

        with open(eval_dir / "gnn_metrics.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        plot_pr_curve(precision, recall, str(eval_dir / "pr_curve_gnn.png"), title="PR Curve (GNN, VAL)")

    print("[A5] Done.")


if __name__ == "__main__":
    main()