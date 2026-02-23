from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv


@dataclass
class GNNConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4


class HeteroSAGE(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        num_nodes_by_type: Dict[str, int],
        tx_in_dim: int,
        cfg: GNNConfig,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.cfg = cfg
        self.node_types, self.edge_types = metadata

        # Embeddings for non-transaction node types
        self.embeddings = nn.ModuleDict()
        for ntype in self.node_types:
            if ntype == "transaction":
                continue
            n = int(num_nodes_by_type[ntype])
            self.embeddings[ntype] = nn.Embedding(n, embed_dim)

        # Transaction feature encoder
        self.tx_mlp = nn.Sequential(
            nn.Linear(tx_in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # Initial dims per type
        self.in_dims = {ntype: cfg.hidden_dim for ntype in self.node_types}
        # non-tx embeddings are projected to hidden_dim
        self.emb_proj = nn.ModuleDict({
            ntype: nn.Linear(embed_dim, cfg.hidden_dim)
            for ntype in self.node_types if ntype != "transaction"
        })

        # Hetero GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            conv = HeteroConv(
                {
                    etype: SAGEConv((cfg.hidden_dim, cfg.hidden_dim), cfg.hidden_dim)
                    for etype in self.edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # x_dict has 'transaction' features, others can be absent -> we build them from embeddings
        h = {}

        # transaction
        h["transaction"] = self.tx_mlp(x_dict["transaction"])

        # entities
        for ntype, emb in self.embeddings.items():
            node_ids = x_dict[ntype]  # expects LongTensor of node indices [0..num_nodes-1]
            h[ntype] = self.emb_proj[ntype](emb(node_ids))

        # message passing
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: F.relu(v) for k, v in h.items()}

        # predict only for transactions
        logits = self.head(h["transaction"]).view(-1)
        return logits