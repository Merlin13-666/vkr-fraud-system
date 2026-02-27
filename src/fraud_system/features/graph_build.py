from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class GraphArtifacts:
    node_map: pd.DataFrame
    edges: pd.DataFrame
    tx_index: pd.DataFrame


def _normalize_value(x: Any) -> str:
    if pd.isna(x):
        return "__MISSING__"
    x = str(x).strip()
    return x if x else "__MISSING__"


def _build_entity_nodes_for_type(
    df: pd.DataFrame,
    cols: List[str],
    entity_type: str,
    min_freq: int,
) -> pd.DataFrame:
    """
    Строит таблицу узлов для одного entity_type.
    node_id локальный: 0..N_type-1
    """
    values = []
    for c in cols:
        if c in df.columns:
            values.append(df[c].map(_normalize_value).map(lambda v: f"{c}::{v}"))

    if not values:
        return pd.DataFrame(columns=["entity_type", "entity_value", "node_id"])

    all_vals = pd.concat(values, ignore_index=True)
    vc = all_vals.value_counts(dropna=False)
    keep = set(vc[vc >= min_freq].index.tolist())

    uniq = all_vals[all_vals.isin(keep)].drop_duplicates().reset_index(drop=True)
    out = pd.DataFrame({"entity_type": entity_type, "entity_value": uniq.values})
    out["node_id"] = out.index.astype("int64")  # локально!
    return out


def build_node_map_and_tx_index(
    df: pd.DataFrame,
    min_freq: int,
    entity_cols: Dict[str, List[str]],
    transaction_id_col: str = "transaction_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    node_map: только сущности (без tx)
    tx_index: отдельно transaction_id -> tx_node_id (локально 0..N_tx-1)
    """
    if transaction_id_col not in df.columns:
        raise ValueError(f"Missing transaction id column: {transaction_id_col}")

    parts = []
    for etype, cols in entity_cols.items():
        part = _build_entity_nodes_for_type(df, cols, etype, min_freq=min_freq)
        if len(part):
            parts.append(part)

    node_map = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["entity_type", "entity_value", "node_id"]
    )

    # tx_index — строго int64
    tx_ids = pd.to_numeric(df[transaction_id_col], errors="coerce").dropna().astype("int64")
    tx_ids = tx_ids.drop_duplicates().sort_values().reset_index(drop=True)
    tx_index = pd.DataFrame({
        "transaction_id": tx_ids.values,
        "tx_node_id": pd.RangeIndex(start=0, stop=len(tx_ids), step=1, dtype="int64"),
    })

    return node_map, tx_index


def build_edges(
    df: pd.DataFrame,
    node_map: pd.DataFrame,
    tx_index: pd.DataFrame,
    entity_cols: Dict[str, List[str]],
    transaction_id_col: str = "transaction_id",
) -> pd.DataFrame:
    """
    Рёбра:
      src_id = transaction_id (int64)
      dst_id = node_id сущности (локальный внутри dst_type)
    """
    if transaction_id_col not in df.columns:
        raise ValueError(f"Missing transaction id column: {transaction_id_col}")

    # entity lookup: (etype, entity_value) -> local node_id
    ent_map = node_map.set_index(["entity_type", "entity_value"])["node_id"].to_dict()

    tx_ids = pd.to_numeric(df[transaction_id_col], errors="coerce").dropna().astype("int64")
    df2 = df.loc[tx_ids.index].copy()
    df2[transaction_id_col] = tx_ids.values

    edges = []
    for etype, cols in entity_cols.items():
        for c in cols:
            if c not in df2.columns:
                continue

            vals = df2[c].map(_normalize_value).map(lambda v: f"{c}::{v}")
            dst_ids = vals.map(lambda v: ent_map.get((etype, v), None))

            mask = dst_ids.notna()
            if not mask.any():
                continue

            part = pd.DataFrame({
                "src_type": "transaction",
                "src_id": df2.loc[mask, transaction_id_col].astype("int64").values,
                "dst_type": etype,
                "dst_id": dst_ids[mask].astype("int64").values,
                "relation": f"tx_to_{etype}",
                "col": c,
            })
            edges.append(part)

    if not edges:
        return pd.DataFrame(columns=["src_type", "src_id", "dst_type", "dst_id", "relation", "col"])

    return pd.concat(edges, ignore_index=True).drop_duplicates()


def make_graph_artifacts(
    df: pd.DataFrame,
    min_freq: int,
    entity_cols: Dict[str, List[str]],
    transaction_id_col: str = "transaction_id",
) -> GraphArtifacts:
    node_map, tx_index = build_node_map_and_tx_index(
        df, min_freq=min_freq, entity_cols=entity_cols, transaction_id_col=transaction_id_col
    )
    edges = build_edges(
        df, node_map=node_map, tx_index=tx_index, entity_cols=entity_cols, transaction_id_col=transaction_id_col
    )
    return GraphArtifacts(node_map=node_map, edges=edges, tx_index=tx_index)