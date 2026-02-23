from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


@dataclass
class PygGraphArtifacts:
    data: HeteroData
    tx_index: pd.DataFrame


def build_heterodata(
    node_map: pd.DataFrame,
    edges: pd.DataFrame,
    transactions: pd.Series,
) -> PygGraphArtifacts:
    """
    node_map: columns [entity_type, entity_value, node_id] where node_id is local per type
    edges: columns [src_type='transaction', src_id=transaction_id, dst_type, dst_id, relation, col]
    transactions: series of transaction_id to define transaction node indices (local 0..N-1)
    """
    data = HeteroData()

    # transaction id -> local index
    tx_ids = transactions.astype("int64").drop_duplicates().sort_values().to_numpy()
    tx_local = pd.DataFrame({
        "transaction_id": tx_ids,
        "tx_node_id": np.arange(len(tx_ids), dtype=np.int64),
    })

    tx_map = dict(zip(tx_local["transaction_id"].values, tx_local["tx_node_id"].values))

    # set num_nodes for each node type
    data["transaction"].num_nodes = int(len(tx_local))

    for etype in node_map["entity_type"].unique():
        n_nodes = int(node_map.loc[node_map["entity_type"] == etype, "node_id"].max()) + 1
        data[etype].num_nodes = n_nodes

    # build edge_index per relation: ('transaction','to','card') and reverse
    # We'll use relation name = f"tx_to_{dst_type}"
    for dst_type in edges["dst_type"].unique():
        sub = edges[edges["dst_type"] == dst_type]

        # map transaction_id -> tx_node_id
        src = sub["src_id"].map(tx_map).astype("int64").to_numpy()
        dst = sub["dst_id"].astype("int64").to_numpy()

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        rel_fwd = ("transaction", "to", dst_type)
        data[rel_fwd].edge_index = edge_index

        # reverse edges
        edge_index_rev = torch.tensor(np.stack([dst, src], axis=0), dtype=torch.long)
        rel_rev = (dst_type, "rev_to", "transaction")
        data[rel_rev].edge_index = edge_index_rev

    return PygGraphArtifacts(data=data, tx_index=tx_local)