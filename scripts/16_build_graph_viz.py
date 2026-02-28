from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"node_id", "entity_type", "entity_value"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"node_map missing columns: {sorted(miss)}; have={list(df.columns)}")
    df = df.copy()
    df["entity_type"] = df["entity_type"].astype(str)
    df["entity_value"] = df["entity_value"].astype(str)
    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("int64")
    return df


def _read_edges(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"src_type", "src_id", "dst_type", "dst_id"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"edges missing columns: {sorted(miss)}; have={list(df.columns)}")

    df = df.copy()
    df["src_type"] = df["src_type"].astype(str)
    df["dst_type"] = df["dst_type"].astype(str)
    df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce")
    df["dst_id"] = pd.to_numeric(df["dst_id"], errors="coerce")
    df = df.dropna(subset=["src_id", "dst_id"]).copy()
    df["src_id"] = df["src_id"].astype("int64")
    df["dst_id"] = df["dst_id"].astype("int64")

    keep = ["src_type", "src_id", "dst_type", "dst_id"]
    for c in ["relation", "col"]:
        if c in df.columns:
            keep.append(c)
    return df[keep].copy()


def _read_tx_index(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"transaction_id", "tx_node_id"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"tx_index missing columns: {sorted(miss)}; have={list(df.columns)}")

    df = df.copy()
    df["transaction_id"] = pd.to_numeric(df["transaction_id"], errors="coerce")
    df["tx_node_id"] = pd.to_numeric(df["tx_node_id"], errors="coerce")
    df = df.dropna(subset=["transaction_id", "tx_node_id"]).copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["tx_node_id"] = df["tx_node_id"].astype("int64")
    return df


def _format_entity_label(entity_type: str, entity_value: str, max_len: int = 26) -> str:
    v = str(entity_value)
    if "::" in v:
        v = v.split("::", 1)[1]
    if len(v) > max_len:
        v = v[: max_len - 1] + "…"
    return v


def _type_style(entity_type: str) -> Dict[str, str]:
    t = str(entity_type).lower()
    if t in ("tx", "transaction"):
        return {"shape": "box", "color": "#4C8BF5"}     # tx
    if "card" in t:
        return {"shape": "ellipse", "color": "#00A889"} # card
    if "email" in t:
        return {"shape": "ellipse", "color": "#F4B400"} # email
    if "device" in t:
        return {"shape": "ellipse", "color": "#DB4437"} # device
    if "addr" in t:
        return {"shape": "ellipse", "color": "#546E7A"} # addr
    return {"shape": "dot", "color": "#9E9E9E"}


def _bfs_nodes(adj: Dict[int, List[int]], start: int, hops: int, max_nodes: int) -> Set[int]:
    visited: Set[int] = {start}
    q = deque([(start, 0)])
    while q and len(visited) < max_nodes:
        u, d = q.popleft()
        if d >= hops:
            continue
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                if len(visited) >= max_nodes:
                    break
                q.append((v, d + 1))
    return visited


def _build_global_offsets(nodes: pd.DataFrame, tx_index: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    counts: Dict[str, int] = {}
    counts["transaction"] = int(tx_index["tx_node_id"].max()) + 1 if len(tx_index) else 0

    for et in sorted(nodes["entity_type"].unique().tolist()):
        mx = nodes.loc[nodes["entity_type"] == et, "node_id"].max()
        counts[et] = int(mx) + 1 if pd.notna(mx) else 0

    offsets: Dict[str, int] = {}
    cur = 0
    for t in ["transaction"] + [t for t in sorted(counts.keys()) if t != "transaction"]:
        offsets[t] = cur
        cur += counts[t]

    return offsets, counts


def _apply_vis_options(net, options_js: str) -> None:
    try:
        net.set_options(options_js)
        return
    except Exception:
        pass

    s = options_js.strip()
    if s.startswith("var options"):
        s = s.split("=", 1)[1].strip()
        if s.endswith(";"):
            s = s[:-1].strip()

    try:
        net.options = json.loads(s)
    except Exception as e:
        raise RuntimeError(
            "Cannot apply vis options (pyvis incompatibility). "
            "Try upgrading pyvis or check options string format."
        ) from e


def _pick_score_col(cols: List[str]) -> Optional[str]:
    cols_set = set(cols)
    candidates = [
        "p_fusion_external",
        "p_fusion_internal",
        "p_fusion",
        "p_tabular",
        "p_gnn",
        "score",
        "pred",
        "proba",
        "risk",
        "risk_score",
    ]
    for c in candidates:
        if c in cols_set:
            return c
    # fallback: any p_*
    p_cols = [c for c in cols if isinstance(c, str) and c.startswith("p_")]
    if p_cols:
        p_cols_sorted = sorted(p_cols, key=lambda x: (len(x), x), reverse=True)
        return p_cols_sorted[0]
    return None


def _pick_tx_from_preds(pred_path: Path, allowed_tx_ids: Set[int]) -> int:
    """
    ВАЖНО: выбираем tx только из тех, которые реально есть в tx_index (т.е. в графе).
    Если пересечение пустое — fallback на первую tx из tx_index, чтобы пайплайн не падал.
    """
    if not pred_path.exists():
        raise RuntimeError(f"pred parquet not found: {pred_path}")

    df = pd.read_parquet(pred_path)
    cols = list(df.columns)

    # tx col
    tx_candidates = ["transaction_id", "TransactionID", "tx_id"]
    tx_col = next((c for c in tx_candidates if c in df.columns), None)
    if tx_col is None:
        raise RuntimeError(f"Cannot auto-pick tx: tx column not found. Have columns: {cols[:50]}")

    score_col = _pick_score_col(cols)
    if score_col is None:
        raise RuntimeError(f"Cannot auto-pick tx: score column not found. Have columns: {cols[:50]}")

    df2 = df[[tx_col, score_col]].copy()
    df2[tx_col] = pd.to_numeric(df2[tx_col], errors="coerce")
    df2[score_col] = pd.to_numeric(df2[score_col], errors="coerce")
    df2 = df2.dropna(subset=[tx_col, score_col]).copy()
    if df2.empty:
        # fallback: берем любую tx из графа
        return int(next(iter(allowed_tx_ids)))

    df2[tx_col] = df2[tx_col].astype("int64")

    # INTERSECTION with graph
    df2 = df2[df2[tx_col].isin(list(allowed_tx_ids))]
    if df2.empty:
        # нет пересечения preds и graph -> fallback, но НЕ падаем
        return int(next(iter(allowed_tx_ids)))

    df2 = df2.sort_values(score_col, ascending=False)
    return int(df2.iloc[0][tx_col])


def main() -> None:
    parser = argparse.ArgumentParser(description="A11+: Interactive graph visualization (pyvis)")

    parser.add_argument("--node-map", default="artifacts/graph/node_map.parquet")
    parser.add_argument("--edges", default="artifacts/graph/edges.parquet")
    parser.add_argument("--tx-index", default="artifacts/graph/tx_index.parquet")
    parser.add_argument("--out", default="reports/assets/graph.html")

    parser.add_argument("--mode", choices=["ego_tx", "ego_entity"], default="ego_tx")
    parser.add_argument("--tx-id", type=int, default=None)
    parser.add_argument("--entity-type", type=str, default=None)
    parser.add_argument("--entity-value", type=str, default=None)

    parser.add_argument("--auto-pick", action="store_true")
    parser.add_argument("--pred-path", default="artifacts/evaluation/val_pred_fusion_external.parquet")

    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--max-edges", type=int, default=3000)

    parser.add_argument("--show-physics", action="store_true")
    args = parser.parse_args()

    node_path = Path(args.node_map)
    edge_path = Path(args.edges)
    tx_path = Path(args.tx_index)
    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    nodes = _read_nodes(node_path)
    edges = _read_edges(edge_path)
    tx_index = _read_tx_index(tx_path)

    txid_to_txnode = dict(zip(tx_index["transaction_id"].values, tx_index["tx_node_id"].values))
    txnode_to_txid = dict(zip(tx_index["tx_node_id"].values, tx_index["transaction_id"].values))
    allowed_tx_ids: Set[int] = set(int(x) for x in tx_index["transaction_id"].tolist())

    offsets, counts = _build_global_offsets(nodes, tx_index)

    ent_local_lookup = nodes.set_index(["entity_type", "node_id"])["entity_value"].to_dict()
    ent_value_lookup = nodes.set_index(["entity_type", "entity_value"])["node_id"].to_dict()

    def g_tx(tx_node_id: int) -> int:
        return offsets["transaction"] + int(tx_node_id)

    def g_ent(entity_type: str, local_id: int) -> int:
        return offsets[str(entity_type)] + int(local_id)

    def global_type_and_local(gid: int) -> Tuple[str, int]:
        items = sorted(offsets.items(), key=lambda x: x[1])
        for i, (t, off) in enumerate(items):
            nxt_off = items[i + 1][1] if i + 1 < len(items) else 10**18
            if off <= gid < nxt_off:
                return t, gid - off
        return "unknown", gid

    # ---- Build global undirected edge list: transaction(tx_node_id) <-> entity(dst_type, dst_id)
    global_edges: List[Tuple[int, int, str]] = []
    for r in edges.itertuples(index=False):
        if r.src_type != "transaction":
            continue
        tx_id = int(r.src_id)
        tx_node = txid_to_txnode.get(tx_id)
        if tx_node is None:
            continue

        u = g_tx(int(tx_node))
        v = g_ent(str(r.dst_type), int(r.dst_id))

        col = getattr(r, "col", None)
        rel = getattr(r, "relation", None)
        if col is not None and str(col):
            el = f"{r.dst_type} via {col}"
        elif rel is not None and str(rel):
            el = str(rel)
        else:
            el = f"tx_to_{r.dst_type}"

        global_edges.append((u, v, el))

    adj: Dict[int, List[int]] = {}
    for u, v, _ in global_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # ---- choose start node
    start_gid: Optional[int] = None
    start_title = ""

    if args.mode == "ego_tx":
        tx_id = args.tx_id
        if tx_id is None and args.auto_pick:
            tx_id = _pick_tx_from_preds(Path(args.pred_path), allowed_tx_ids)
        if tx_id is None:
            raise RuntimeError("--mode ego_tx requires --tx-id, or use --auto-pick")

        tx_id = int(tx_id)
        tx_node = txid_to_txnode.get(tx_id)
        if tx_node is None:
            # теперь это почти не должно случаться, но оставим safety
            examples = list(txid_to_txnode.keys())[:10]
            raise RuntimeError(f"transaction_id={tx_id} not found in tx_index. Example tx_id: {examples}")

        start_gid = g_tx(int(tx_node))
        start_title = f"TX ego-graph: transaction_id={tx_id} (tx_node_id={int(tx_node)})"

    elif args.mode == "ego_entity":
        if not args.entity_type or not args.entity_value:
            raise RuntimeError("--mode ego_entity requires --entity-type and --entity-value")
        et = str(args.entity_type)
        ev = str(args.entity_value)
        local_id = ent_value_lookup.get((et, ev))
        if local_id is None:
            sample = nodes[nodes["entity_type"] == et].head(8)["entity_value"].tolist()
            raise RuntimeError(f"entity not found: {et}/{ev}. Examples: {sample}")
        start_gid = g_ent(et, int(local_id))
        start_title = f"Entity ego-graph: {et} / {ev} (node_id={int(local_id)})"

    if start_gid is None:
        raise RuntimeError("Cannot determine start node (check args).")

    keep = _bfs_nodes(adj, start=int(start_gid), hops=int(args.hops), max_nodes=int(args.max_nodes))
    e_keep: List[Tuple[int, int, str]] = [(u, v, lab) for (u, v, lab) in global_edges if u in keep and v in keep]
    if len(e_keep) > int(args.max_edges):
        e_keep = e_keep[: int(args.max_edges)]

    deg: Dict[int, int] = {}
    for u, v, _ in global_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    try:
        from pyvis.network import Network
    except Exception as e:
        raise RuntimeError(
            "pyvis is not installed. Install it:\n"
            "  pip install pyvis\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    net = Network(height="850px", width="100%", bgcolor="#ffffff", font_color="#111111", directed=False, notebook=False)

    _apply_vis_options(net, """var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 250, "updateInterval": 25},
        "barnesHut": {
          "gravitationalConstant": -2500,
          "springLength": 160,
          "springConstant": 0.02,
          "damping": 0.35,
          "avoidOverlap": 0.85
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true,
        "keyboard": true,
        "hideEdgesOnDrag": true
      },
      "nodes": {"font": {"size": 14}},
      "edges": {"smooth": false}
    };""")

    for gid in sorted(keep):
        t, local = global_type_and_local(int(gid))
        is_center = int(gid) == int(start_gid)

        if t == "transaction":
            tx_id = txnode_to_txid.get(int(local))
            label = f"tx:{tx_id}" if tx_id is not None else f"tx_node:{local}"
            title = (
                f"{start_title if is_center else ''}\n\n"
                f"type=transaction\n"
                f"transaction_id={tx_id}\n"
                f"tx_node_id={local}\n"
                f"deg={deg.get(int(gid), 0)}"
            )
        else:
            ev = ent_local_lookup.get((t, int(local)), f"{t}::{local}")
            label = _format_entity_label(t, ev, max_len=26)
            title = (
                f"{start_title if is_center else ''}\n\n"
                f"type={t}\n"
                f"value={ev}\n"
                f"node_id={local}\n"
                f"deg={deg.get(int(gid), 0)}"
            )

        st = _type_style(t)
        size = 10 + min(28, deg.get(int(gid), 0) // 2)
        if is_center:
            size = 34

        net.add_node(
            str(gid),
            label=label,
            title=title,
            color=("#111111" if is_center else st["color"]),
            shape=("diamond" if is_center else st["shape"]),
            value=int(size),
            group=t,
        )

    for u, v, lab in e_keep:
        w = 2 if (int(u) == int(start_gid) or int(v) == int(start_gid)) else 1
        net.add_edge(str(u), str(v), width=w, title=lab)

    if args.show_physics:
        net.show_buttons(filter_=["physics"])

    net.write_html(str(out_path))

    # freeze physics after stabilization
    html = out_path.read_text(encoding="utf-8")
    if "network = new vis.Network(container, data, options);" in html:
        html = html.replace(
            "network = new vis.Network(container, data, options);",
            "network = new vis.Network(container, data, options);\n"
            'network.once("stabilizationIterationsDone", function () {'
            "  network.setOptions({physics: {enabled: false}});"
            "});\n",
            1,
        )
    elif "new vis.Network(container, data, options);" in html:
        html = html.replace(
            "new vis.Network(container, data, options);",
            "var network = new vis.Network(container, data, options);\n"
            'network.once("stabilizationIterationsDone", function () {'
            "  network.setOptions({physics: {enabled: false}});"
            "});\n",
            1,
        )

    legend = """
    <div style="position:fixed; right:18px; top:18px; z-index:9999;
                background:#fff; border:1px solid #eee; border-radius:12px;
                padding:12px; font-family:Arial; font-size:12px; color:#111;">
      <div style="font-weight:bold; margin-bottom:6px;">Legend</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#4C8BF5;margin-right:6px;"></span>transaction</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#00A889;margin-right:6px;"></span>card*</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#F4B400;margin-right:6px;"></span>*email*</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#DB4437;margin-right:6px;"></span>*device*</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#546E7A;margin-right:6px;"></span>addr*</div>
      <div style="margin-top:8px;color:#555;">Tip: hover nodes/edges for details.</div>
    </div>
    """
    html = html.replace("</body>", f"{legend}\n</body>", 1)
    out_path.write_text(html, encoding="utf-8")

    print(f"[A11+] Saved: {out_path.as_posix()}")
    print(f"[A11+] {start_title}")
    print(f"[A11+] types_offsets={offsets}, counts={counts}")
    print(f"[A11+] nodes={len(keep)}, edges={len(e_keep)}")


if __name__ == "__main__":
    main()