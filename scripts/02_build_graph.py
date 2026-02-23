from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
import json
from datetime import datetime

from fraud_system.features.graph_build import make_graph_artifacts


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = _read_yaml("configs/base.yaml")

    processed_path = Path(cfg["data"]["processed_path"])
    df_train = pd.read_parquet(processed_path / "train.parquet")

    graph_cfg = cfg.get("graph", {})
    min_freq = int(graph_cfg.get("min_freq", 10))
    max_rows = graph_cfg.get("max_rows", None)
    entity_cols = graph_cfg.get("entity_cols", {})

    if max_rows is not None:
        df_train = df_train.head(int(max_rows)).copy()
        print(f"[A4] Using head(max_rows={max_rows}): rows={len(df_train)}")

    print(f"[A4] Building graph from train.parquet rows={len(df_train)}")
    print(f"[A4] min_freq={min_freq}")
    print(f"[A4] entity_cols={entity_cols}")

    artifacts = make_graph_artifacts(df_train, min_freq=min_freq, entity_cols=entity_cols)

    out_dir = Path("artifacts/graph")
    _ensure_dir(out_dir)

    node_map_path = out_dir / "node_map.parquet"
    edges_path = out_dir / "edges.parquet"

    artifacts.node_map.to_parquet(node_map_path, index=False)
    artifacts.edges.to_parquet(edges_path, index=False)

    print(f"[A4] Saved: {node_map_path}")
    print(f"[A4] Saved: {edges_path}")

    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "train.parquet",
        "rows_used": int(len(df_train)),
        "min_freq": int(min_freq),
        "entity_cols": entity_cols,
        "node_counts_by_type": artifacts.node_map["entity_type"].value_counts().to_dict(),
        "total_edges": int(len(artifacts.edges)),
        "edge_counts_by_dst_type": artifacts.edges["dst_type"].value_counts().to_dict(),
        "unique_transactions": int(df_train["transaction_id"].nunique()),
    }

    info_path = out_dir / "graph_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[A4] Saved: {info_path}")

    # Stats
    nm = artifacts.node_map
    ed = artifacts.edges

    if len(nm) > 0:
        stats_nodes = nm.groupby("entity_type")["node_id"].count().sort_values(ascending=False)
        print("[A4] Node counts by type:")
        print(stats_nodes.to_string())
    print(f"[A4] Total edges: {len(ed)}")
    print("[A4] Edge counts by dst_type:")
    print(ed["dst_type"].value_counts().to_string())

    print("[A4] Done.")


if __name__ == "__main__":
    main()