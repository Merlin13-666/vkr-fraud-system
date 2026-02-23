from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def load_node_map(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"entity_type", "entity_value", "node_id"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"node_map missing columns {miss}")
    df["entity_type"] = df["entity_type"].astype("string")
    df["entity_value"] = df["entity_value"].astype("string")
    df["node_id"] = df["node_id"].astype("int64")
    return df


def load_tx_scaler(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_tx_features(df: pd.DataFrame, feat_cols: List[str]) -> List[str]:
    # ensure stable order and existence
    cols = []
    for c in feat_cols:
        if c in df.columns:
            cols.append(c)
        else:
            cols.append(c)  # will be created as NaN later
    return cols


def _apply_scaler(df: pd.DataFrame, feat_cols: List[str], scaler: Dict) -> np.ndarray:
    # create missing cols as NaN
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feat_cols].to_numpy(dtype=np.float32, copy=True)

    med = np.array(scaler["median"], dtype=np.float32)
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    # impute NaN with median
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(med, inds[1])

    # standardize
    X = (X - mean) / (std + 1e-6)
    return X.astype(np.float32)


def build_inductive_heterodata(
    df_ext: pd.DataFrame,
    node_map_df: pd.DataFrame,
    entity_cols: Dict[str, List[str]],
    tx_feat_cols: List[str],
    scaler: Dict,
) -> Tuple[HeteroData, pd.DataFrame, Dict[str, int]]:
    """
    Build HeteroData for inductive inference:
      - transaction nodes are only external transactions (fresh indexing 0..N-1)
      - entity nodes reuse train node_id indexing from node_map
      - unknown entity values are mapped to per-type UNK node at index = num_nodes_type
    Returns:
      data, tx_index (transaction_id -> tx_node_id), num_nodes_by_type
    """
    if "transaction_id" not in df_ext.columns:
        raise ValueError("df_ext must contain transaction_id")

    # tx indexing for external
    tx_index = df_ext[["transaction_id"]].copy()
    tx_index["transaction_id"] = tx_index["transaction_id"].astype("int64")
    tx_index = tx_index.drop_duplicates("transaction_id").reset_index(drop=True)
    tx_index["tx_node_id"] = np.arange(len(tx_index), dtype=np.int64)

    # mapping (entity_type, entity_value) -> node_id
    key = list(zip(node_map_df["entity_type"].to_list(), node_map_df["entity_value"].to_list()))
    val = node_map_df["node_id"].to_list()
    map_dict = dict(zip(key, val))

    # count nodes per entity type (train)
    counts = node_map_df.groupby("entity_type")["node_id"].max().to_dict()
    num_nodes_by_type = {}
    unk_id_by_type = {}
    for etype, max_id in counts.items():
        n = int(max_id) + 1
        num_nodes_by_type[str(etype)] = n
        unk_id_by_type[str(etype)] = n  # UNK will be appended
    # +1 for UNK
    for etype in list(num_nodes_by_type.keys()):
        num_nodes_by_type[etype] += 1

    # Build edges: (tx_node_id, entity_node_id) for each entity type
    tx_id_to_node = tx_index.set_index("transaction_id")["tx_node_id"]

    # Join df_ext with tx_node_id
    df = df_ext.merge(tx_index, on="transaction_id", how="inner")

    edge_store: Dict[str, Tuple[List[int], List[int]]] = {}
    stats_unknown = {k: 0 for k in entity_cols.keys()}
    stats_total = {k: 0 for k in entity_cols.keys()}

    for etype, cols in entity_cols.items():
        src_list: List[int] = []
        dst_list: List[int] = []

        for col in cols:
            if col not in df.columns:
                continue

            s = df[["tx_node_id", col]].copy()
            s = s.dropna()
            s[col] = s[col].astype("string")

            # prefix by column to avoid collisions (must match A4)
            # example: card1::12345
            s["entity_value"] = col + "::" + s[col]
            s["entity_type"] = etype

            stats_total[etype] += int(len(s))

            # map to node_id (or UNK)
            def _map_one(v: str) -> int:
                nid = map_dict.get((etype, v))
                if nid is None:
                    stats_unknown[etype] += 1
                    return unk_id_by_type[etype]
                return int(nid)

            s["entity_node_id"] = s["entity_value"].map(_map_one).astype("int64")

            src_list.extend(s["tx_node_id"].astype("int64").to_list())
            dst_list.extend(s["entity_node_id"].to_list())

        edge_store[etype] = (src_list, dst_list)

    # Build tx features with stored scaler
    feat_cols = _select_tx_features(df_ext, tx_feat_cols)
    X = _apply_scaler(df_ext.copy(), feat_cols=feat_cols, scaler=scaler)

    data = HeteroData()
    data["transaction"].x = torch.tensor(X, dtype=torch.float32)

    # entity node x: index tensor for embedding
    for etype, n_nodes in num_nodes_by_type.items():
        data[etype].x = torch.arange(n_nodes, dtype=torch.long)

    # edges + reverse edges
    for etype, (src, dst) in edge_store.items():
        if len(src) == 0:
            # empty edge set is ok, but create empty tensors
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        data[("transaction", "to", etype)].edge_index = edge_index
        data[(etype, "rev_to", "transaction")].edge_index = edge_index.flip(0)

    # Save mapping stats (useful for report)
    mapping_stats = {
        "n_tx": int(len(tx_index)),
        "unknown_counts": {k: int(v) for k, v in stats_unknown.items()},
        "total_links": {k: int(v) for k, v in stats_total.items()},
        "unknown_share": {
            k: float(stats_unknown[k] / stats_total[k]) if stats_total[k] > 0 else 0.0
            for k in stats_total.keys()
        },
        "num_nodes_by_type": {k: int(v) for k, v in num_nodes_by_type.items()},
        "entity_cols": entity_cols,
    }

    return data, tx_index, mapping_stats