from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    x = x.astype(np.float64)
    return {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


@dataclass
class DSU:
    parent: np.ndarray
    size: np.ndarray

    @classmethod
    def make(cls, n: int) -> "DSU":
        parent = np.arange(n, dtype=np.int64)
        size = np.ones(n, dtype=np.int64)
        return cls(parent=parent, size=size)

    def find(self, a: int) -> int:
        # iterative path compression
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        # union by size
        if self.size[pa] < self.size[pb]:
            pa, pb = pb, pa
        self.parent[pb] = pa
        self.size[pa] += self.size[pb]


def _plot_degree_hist(deg: np.ndarray, out_path: Path, title: str) -> None:
    _ensure_dir(out_path.parent)
    plt.figure()
    if deg.size == 0:
        plt.title(title + " (empty)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    # log-friendly bins: clip to >=1 for log-scale view
    d = deg.astype(np.int64)
    d = d[d >= 0]
    if d.size == 0:
        plt.title(title + " (empty)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    # choose bins based on range
    vmax = int(d.max())
    if vmax <= 50:
        bins = min(50, vmax + 1)
    else:
        bins = 60

    plt.hist(d, bins=bins)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # if heavy-tailed, log-scale x makes it readable
    if vmax >= 100:
        try:
            plt.xscale("log")
        except Exception:
            pass

    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_edges_by_type(counts: pd.Series, out_path: Path, title: str) -> None:
    _ensure_dir(out_path.parent)
    plt.figure()
    if counts.empty:
        plt.title(title + " (empty)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    counts = counts.sort_values(ascending=False)
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Edges")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_top_entities(df_top: pd.DataFrame, out_path: Path, title: str) -> None:
    _ensure_dir(out_path.parent)
    plt.figure()
    if df_top.empty:
        plt.title(title + " (empty)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    # horizontal bar for readability
    labels = df_top["entity_key"].astype(str).values
    vals = df_top["degree"].astype(float).values
    y = np.arange(len(labels))

    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Degree")
    plt.title(title)
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    graph_dir = Path("artifacts/graph")
    eval_dir = Path("artifacts/evaluation")
    _ensure_dir(graph_dir)
    _ensure_dir(eval_dir)

    node_map_path = graph_dir / "node_map.parquet"
    edges_path = graph_dir / "edges.parquet"
    tx_index_path = graph_dir / "tx_index.parquet"
    graph_info_path = graph_dir / "graph_info.json"

    for p in [node_map_path, edges_path, tx_index_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run A4 steps first (02_build_graph + 03_make_graph_data).")

    print(f"[A4.3] Loading: {node_map_path}")
    node_map = pd.read_parquet(node_map_path)

    print(f"[A4.3] Loading: {edges_path}")
    edges = pd.read_parquet(edges_path)

    print(f"[A4.3] Loading: {tx_index_path}")
    tx_index = pd.read_parquet(tx_index_path)

    graph_info: Dict[str, Any] = {}
    if graph_info_path.exists():
        try:
            graph_info = _load_json(graph_info_path)
        except Exception:
            graph_info = {}

    # --- Basic checks (schema based on your build_graph.py)
    need_nm = {"entity_type", "entity_value", "node_id"}
    miss_nm = need_nm - set(node_map.columns)
    if miss_nm:
        raise ValueError(f"node_map missing columns: {miss_nm}")

    need_e = {"src_type", "src_id", "dst_type", "dst_id", "relation", "col"}
    miss_e = need_e - set(edges.columns)
    if miss_e:
        raise ValueError(f"edges missing columns: {miss_e}")

    if "transaction_id" not in tx_index.columns:
        raise ValueError("tx_index missing column 'transaction_id'")

    # --- Node counts
    n_tx = int(tx_index["transaction_id"].nunique())
    node_counts_by_type: Dict[str, int] = {"transaction": n_tx}

    nm_counts = (
        node_map[["entity_type", "node_id"]]
        .drop_duplicates()
        .groupby("entity_type")["node_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    for t, c in nm_counts.items():
        node_counts_by_type[str(t)] = int(c)

    # --- Edge counts
    total_edges = int(len(edges))
    edges_by_dst_type = edges["dst_type"].value_counts(dropna=False).sort_values(ascending=False)
    edges_by_col = edges["col"].value_counts(dropna=False).sort_values(ascending=False)
    edges_by_relation = edges["relation"].value_counts(dropna=False).sort_values(ascending=False)

    # --- Degrees
    tx_deg = edges.groupby("src_id").size().astype(np.int64)
    tx_deg_arr = tx_deg.reindex(tx_index["transaction_id"].astype(np.int64), fill_value=0).to_numpy(dtype=np.int64)

    # entity degree: by (dst_type, dst_id)
    ent_deg = edges.groupby(["dst_type", "dst_id"]).size().astype(np.int64).reset_index(name="degree")
    ent_deg["entity_key"] = ent_deg["dst_type"].astype(str) + ":" + ent_deg["dst_id"].astype(str)

    # isolated shares
    isolated_tx_share = float(np.mean(tx_deg_arr == 0))

    isolated_entity_share_by_type: Dict[str, float] = {}
    for etype in nm_counts.index.astype(str):
        # all nodes of this type:
        all_nodes = node_map.loc[node_map["entity_type"].astype(str) == etype, "node_id"].drop_duplicates()
        if all_nodes.empty:
            isolated_entity_share_by_type[etype] = 0.0
            continue
        # degree map for this type
        deg_nodes = ent_deg.loc[ent_deg["dst_type"].astype(str) == etype, "dst_id"].astype(np.int64)
        deg_nodes_set = set(deg_nodes.tolist())
        iso = int(sum(1 for nid in all_nodes.astype(np.int64).tolist() if nid not in deg_nodes_set))
        isolated_entity_share_by_type[etype] = float(iso / max(1, int(all_nodes.size)))

    # --- Connected components via DSU over bipartite graph (tx + entity nodes)
    # tx nodes: 0..n_tx-1 in the order of tx_index unique
    tx_ids = tx_index["transaction_id"].astype(np.int64).drop_duplicates().to_numpy()
    tx_id_to_idx = {int(t): i for i, t in enumerate(tx_ids)}  # OK for ~500k

    # entity nodes: codes from factorize(entity_key) -> indices n_tx + code
    entity_key = edges["dst_type"].astype(str) + ":" + edges["dst_id"].astype(str)
    ent_codes, ent_uniques = pd.factorize(entity_key, sort=False)
    ent_codes = ent_codes.astype(np.int64)

    # map tx ids for each edge to idx
    tx_edge_idx = edges["src_id"].astype(np.int64).map(tx_id_to_idx)
    if tx_edge_idx.isna().any():
        miss = int(tx_edge_idx.isna().sum())
        raise RuntimeError(f"[A4.3] {miss} edges refer to unknown transaction_id (not in tx_index). Check artifacts integrity.")
    tx_edge_idx = tx_edge_idx.to_numpy(dtype=np.int64)

    n_ent = int(len(ent_uniques))
    dsu = DSU.make(n_tx + n_ent)

    ent_edge_idx = n_tx + ent_codes
    # union for each edge
    for a, b in zip(tx_edge_idx, ent_edge_idx):
        dsu.union(int(a), int(b))

    # component sizes (we focus on tx nodes)
    root_tx = np.fromiter((dsu.find(i) for i in range(n_tx)), dtype=np.int64, count=n_tx)
    comp_sizes_tx = pd.Series(root_tx).value_counts().to_numpy(dtype=np.int64)

    components_count = int(comp_sizes_tx.size)
    largest_component_share_tx = float(comp_sizes_tx.max() / max(1, n_tx)) if comp_sizes_tx.size else 0.0
    comp_sizes_summary = _summary_stats(comp_sizes_tx)

    # --- Plots (match your simple plots.py style)
    p_tx_hist = eval_dir / "graph_degree_tx_hist.png"
    p_edges_type = eval_dir / "graph_edges_by_dst_type.png"
    p_top_ent = eval_dir / "graph_top_entities_degree.png"
    p_comp_hist = eval_dir / "graph_components_hist.png"

    _plot_degree_hist(tx_deg_arr, p_tx_hist, title="Graph degree distribution (transactions)")
    _plot_edges_by_type(edges_by_dst_type, p_edges_type, title="Edges by destination node type")

    top20 = ent_deg.sort_values("degree", ascending=False).head(20)[["entity_key", "degree"]].copy()
    _plot_top_entities(top20, p_top_ent, title="Top-20 entity nodes by degree")

    _plot_degree_hist(comp_sizes_tx, p_comp_hist, title="Connected component sizes (TX nodes)")

    # --- Assemble JSON
    out = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_files": {
                "node_map": str(node_map_path).replace("\\", "/"),
                "edges": str(edges_path).replace("\\", "/"),
                "tx_index": str(tx_index_path).replace("\\", "/"),
                "graph_info": str(graph_info_path).replace("\\", "/") if graph_info_path.exists() else None,
            },
            "graph_info": graph_info,
        },
        "nodes": {
            "counts_by_type": node_counts_by_type,
        },
        "edges": {
            "total": total_edges,
            "by_dst_type": {str(k): int(v) for k, v in edges_by_dst_type.items()},
            "by_col": {str(k): int(v) for k, v in edges_by_col.items()},
            "by_relation": {str(k): int(v) for k, v in edges_by_relation.items()},
        },
        "degrees": {
            "tx_degree_summary": _summary_stats(tx_deg_arr),
            "isolated_tx_share": isolated_tx_share,
            "isolated_entity_share_by_type": isolated_entity_share_by_type,
        },
        "components": {
            "method": "dsu_bipartite(tx + entity)",
            "components_count_tx": components_count,
            "largest_component_share_tx": largest_component_share_tx,
            "component_sizes_tx_summary": comp_sizes_summary,
        },
        "figures": {
            "degree_tx_hist": str(p_tx_hist).replace("\\", "/"),
            "edges_by_dst_type": str(p_edges_type).replace("\\", "/"),
            "top_entities_degree": str(p_top_ent).replace("\\", "/"),
            "components_hist": str(p_comp_hist).replace("\\", "/"),
        },
    }

    out_path = graph_dir / "graph_stats.json"
    _save_json(out_path, out)

    print(f"[A4.3] Nodes: {node_counts_by_type}")
    print(f"[A4.3] Edges: total={total_edges}, by_dst_type={dict(edges_by_dst_type.head(10))}")
    print(f"[A4.3] TX degree: {out['degrees']['tx_degree_summary']}, isolated_share={isolated_tx_share:.4f}")
    print(f"[A4.3] Components (TX): count={components_count}, largest_share={largest_component_share_tx:.4f}")
    print(f"[A4.3] Saved: {out_path}")
    print(f"[A4.3] Saved figs: {p_tx_hist}, {p_edges_type}, {p_top_ent}, {p_comp_hist}")
    print("[A4.3] Done.")


if __name__ == "__main__":
    main()