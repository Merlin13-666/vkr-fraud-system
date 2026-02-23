from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


@dataclass
class GraphArtifacts:
    node_map: pd.DataFrame
    edges: pd.DataFrame


def _normalize_value(x: Any) -> str:
    if pd.isna(x):
        return "__MISSING__"
    x = str(x).strip()
    return x if x else "__MISSING__"


def _build_entity_nodes(
    df: pd.DataFrame,
    cols: List[str],
    entity_type: str,
    min_freq: int,
) -> pd.DataFrame:
    values = []
    for c in cols:
        if c in df.columns:
            # добавляем префикс по имени колонки
            values.append(df[c].map(_normalize_value).map(lambda v: f"{c}::{v}"))
    if not values:
        return pd.DataFrame(columns=["entity_type", "entity_value", "node_id"])

    all_vals = pd.concat(values, ignore_index=True)
    vc = all_vals.value_counts(dropna=False)

    keep = set(vc[vc >= min_freq].index.tolist())
    filtered = all_vals[all_vals.isin(keep)].drop_duplicates()

    entity_df = pd.DataFrame({
        "entity_type": entity_type,
        "entity_value": filtered.values,
    }).drop_duplicates()

    entity_df = entity_df.reset_index(drop=True)
    entity_df["node_id"] = entity_df.index.astype("int64")
    return entity_df


def build_node_map(
    df: pd.DataFrame,
    min_freq: int,
    entity_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Строит node_map для всех сущностей.
    transaction-узлы отдельно не храним здесь (они определяются списком transaction_id).
    """
    parts = []
    for etype, cols in entity_cols.items():
        part = _build_entity_nodes(df, cols, etype, min_freq)
        parts.append(part)

    node_map = pd.concat(parts, ignore_index=True)
    return node_map


def build_edges(
    df: pd.DataFrame,
    node_map: pd.DataFrame,
    entity_cols: Dict[str, List[str]],
    transaction_id_col: str = "transaction_id",
) -> pd.DataFrame:
    if transaction_id_col not in df.columns:
        raise ValueError(f"Missing transaction id column: {transaction_id_col}")

    idx = node_map.set_index(["entity_type", "entity_value"])["node_id"].to_dict()

    edges = []
    tx = df[transaction_id_col].astype("int64")

    for etype, cols in entity_cols.items():
        for c in cols:
            if c not in df.columns:
                continue

            vals = df[c].map(_normalize_value).map(lambda v: f"{c}::{v}")
            dst_ids = vals.map(lambda v: idx.get((etype, v), None))
            mask = dst_ids.notna()

            part = pd.DataFrame({
                "src_type": "transaction",
                "src_id": tx[mask].values,
                "dst_type": etype,
                "dst_id": dst_ids[mask].astype("int64").values,
                "relation": f"tx_to_{etype}",
                "col": c,
            })
            edges.append(part)

    edges_df = pd.concat(edges, ignore_index=True).drop_duplicates()
    return edges_df


def make_graph_artifacts(
    df: pd.DataFrame,
    min_freq: int,
    entity_cols: Dict[str, List[str]],
) -> GraphArtifacts:
    node_map = build_node_map(df, min_freq=min_freq, entity_cols=entity_cols)
    edges = build_edges(df, node_map=node_map, entity_cols=entity_cols)
    return GraphArtifacts(node_map=node_map, edges=edges)