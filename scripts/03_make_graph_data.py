from __future__ import annotations

from pathlib import Path
import torch
import pandas as pd

from fraud_system.graph.pyg_build import build_heterodata


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    graph_dir = Path("artifacts/graph")
    node_map_path = graph_dir / "node_map.parquet"
    edges_path = graph_dir / "edges.parquet"

    if not node_map_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Run scripts.02_build_graph first (node_map/edges missing).")

    node_map = pd.read_parquet(node_map_path)
    edges = pd.read_parquet(edges_path)

    # transactions present in edges (train set)
    tx_ids = edges["src_id"].astype("int64")

    artifacts = build_heterodata(node_map=node_map, edges=edges, transactions=tx_ids)

    _ensure_dir(graph_dir)

    out_pt = graph_dir / "graph_data.pt"
    torch.save(artifacts.data, out_pt)

    tx_index_path = graph_dir / "tx_index.parquet"
    artifacts.tx_index.to_parquet(tx_index_path, index=False)

    print(f"[A4.2] Saved: {out_pt}")
    print(f"[A4.2] Saved: {tx_index_path}")
    print(f"[A4.2] HeteroData node types: {artifacts.data.node_types}")
    print(f"[A4.2] HeteroData edge types: {artifacts.data.edge_types}")
    print("[A4.2] Done.")


if __name__ == "__main__":
    main()