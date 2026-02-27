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
    df["node_id"] = df["node_id"].astype("int64")
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
    # src_id = transaction_id (int)
    df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce")
    # dst_id = local node_id inside dst_type (int)
    df["dst_id"] = pd.to_numeric(df["dst_id"], errors="coerce")
    df = df.dropna(subset=["src_id", "dst_id"]).copy()
    df["src_id"] = df["src_id"].astype("int64")
    df["dst_id"] = df["dst_id"].astype("int64")

    # keep useful columns if exist
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
    df = df.dropna(subset=["transaction_id"]).copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["tx_node_id"] = pd.to_numeric(df["tx_node_id"], errors="coerce").astype("int64")
    return df


def _pick_entity_local_id(nodes: pd.DataFrame, entity_type: str, entity_value: str) -> int:
    m = (nodes["entity_type"] == str(entity_type)) & (nodes["entity_value"] == str(entity_value))
    sub = nodes.loc[m, ["node_id"]]
    if sub.empty:
        sample = nodes[nodes["entity_type"] == str(entity_type)].head(8)["entity_value"].tolist()
        raise RuntimeError(
            f"entity not found: type={entity_type}, value={entity_value}. "
            f"Examples for this type: {sample}"
        )
    return int(sub.iloc[0]["node_id"])


def _format_entity_label(entity_type: str, entity_value: str, max_len: int = 26) -> str:
    """
    Лейбл делаем более "человеческим":
    entity_value у тебя вида "card1::13926" / "P_emaildomain::gmail.com" и т.п.
    Покажем правую часть после '::', а тип оставим цветом/формой.
    """
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


def _build_global_offsets(
    nodes: pd.DataFrame,
    tx_index: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Глобальные id = offset[type] + local_id
    - transaction local_id = tx_node_id (0..N_tx-1)
    - entity local_id = node_id внутри entity_type (0..N_type-1)
    """
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
    """
    Совместимость между версиями pyvis:
    - новые: net.set_options(str) работает
    - старые/поломанные: net.options это dict и set_options падает -> кладём dict напрямую
    """
    try:
        net.set_options(options_js)
        return
    except Exception:
        pass

    # fallback: пытаемся распарсить "var options = {...};" или просто "{...}"
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

def main() -> None:
    parser = argparse.ArgumentParser(description="A11+: Interactive graph visualization (pyvis)")
    parser.add_argument("--node-map", default="artifacts/graph/node_map.parquet")
    parser.add_argument("--edges", default="artifacts/graph/edges.parquet")
    parser.add_argument("--tx-index", default="artifacts/graph/tx_index.parquet")

    parser.add_argument("--out", default="reports/assets/graph.html")

    parser.add_argument("--mode", choices=["ego_tx", "ego_entity", "ego_global"], default="ego_tx")
    parser.add_argument("--tx-id", type=int, default=None, help="transaction_id as integer (e.g. 2987000)")
    parser.add_argument("--entity-type", type=str, default=None, help="card/email/device/addr")
    parser.add_argument("--entity-value", type=str, default=None, help='e.g. "card1::13926"')

    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--max-edges", type=int, default=10000)

    parser.add_argument("--show-physics", action="store_true", help="Show physics settings panel (optional)")
    args = parser.parse_args()

    node_path = Path(args.node_map)
    edge_path = Path(args.edges)
    tx_path = Path(args.tx_index)
    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    nodes = _read_nodes(node_path)
    edges = _read_edges(edge_path)
    tx_index = _read_tx_index(tx_path)

    # maps for tx
    txid_to_txnode = dict(zip(tx_index["transaction_id"].values, tx_index["tx_node_id"].values))
    txnode_to_txid = dict(zip(tx_index["tx_node_id"].values, tx_index["transaction_id"].values))

    # global id offsets
    offsets, counts = _build_global_offsets(nodes, tx_index)

    # entity local maps
    ent_local_lookup = nodes.set_index(["entity_type", "node_id"])["entity_value"].to_dict()
    ent_value_lookup = nodes.set_index(["entity_type", "entity_value"])["node_id"].to_dict()

    def g_tx(tx_node_id: int) -> int:
        return offsets["transaction"] + int(tx_node_id)

    def g_ent(entity_type: str, local_id: int) -> int:
        return offsets[str(entity_type)] + int(local_id)

    def global_type_and_local(gid: int) -> Tuple[str, int]:
        # find type by offsets ranges
        # offsets are increasing; do a simple scan (small number of types)
        items = sorted(offsets.items(), key=lambda x: x[1])
        for i, (t, off) in enumerate(items):
            nxt_off = items[i + 1][1] if i + 1 < len(items) else 10**18
            if off <= gid < nxt_off:
                return t, gid - off
        return "unknown", gid

    # ---- Build global undirected edge list: transaction(tx_node_id) <-> entity(dst_type, dst_id)
    global_edges: List[Tuple[int, int, str]] = []  # (u,v,edge_label)
    for r in edges.itertuples(index=False):
        if r.src_type != "transaction":
            continue  # по твоим артефактам src_type всегда transaction, но оставим безопасно
        tx_id = int(r.src_id)
        tx_node = txid_to_txnode.get(tx_id)
        if tx_node is None:
            continue

        u = g_tx(int(tx_node))
        v = g_ent(str(r.dst_type), int(r.dst_id))

        # label ребра в тултипе: какая колонка связала
        col = getattr(r, "col", None)
        rel = getattr(r, "relation", None)
        el = ""
        if col is not None and str(col):
            el = f"{r.dst_type} via {col}"
        elif rel is not None and str(rel):
            el = str(rel)
        else:
            el = f"tx_to_{r.dst_type}"

        global_edges.append((u, v, el))

    # adjacency for BFS
    adj: Dict[int, List[int]] = {}
    for u, v, _ in global_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # ---- choose start node (global id)
    start_gid: Optional[int] = None
    start_title = ""

    if args.mode == "ego_tx":
        if args.tx_id is None:
            raise RuntimeError("--mode ego_tx requires --tx-id (int), e.g. --tx-id 2987000")
        tx_id = int(args.tx_id)
        tx_node = txid_to_txnode.get(tx_id)
        if tx_node is None:
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

    elif args.mode == "ego_global":
        raise RuntimeError("--mode ego_global is reserved (not exposed here). Use ego_tx or ego_entity.")

    if start_gid is None:
        raise RuntimeError("Cannot determine start node (check args).")

    # ---- BFS subgraph on global ids
    keep = _bfs_nodes(adj, start=int(start_gid), hops=int(args.hops), max_nodes=int(args.max_nodes))

    # filter edges inside keep
    e_keep: List[Tuple[int, int, str]] = [(u, v, lab) for (u, v, lab) in global_edges if u in keep and v in keep]
    if len(e_keep) > int(args.max_edges):
        e_keep = e_keep[: int(args.max_edges)]

    # degrees for size (within full global graph)
    deg: Dict[int, int] = {}
    for u, v, _ in global_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    # ---- pyvis render
    try:
        from pyvis.network import Network
    except Exception as e:
        raise RuntimeError(
            "pyvis is not installed. Install it:\n"
            "  pip install pyvis\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    net = Network(height="850px", width="100%", bgcolor="#ffffff", font_color="#111111", directed=False, notebook=False)

    # options: стабильная физика + hover
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

    # add nodes with meaningful labels
    for gid in sorted(keep):
        t, local = global_type_and_local(int(gid))
        is_center = int(gid) == int(start_gid)

        if t == "transaction":
            tx_id = txnode_to_txid.get(int(local))
            label = f"tx:{tx_id}" if tx_id is not None else f"tx_node:{local}"
            title = f"{start_title if is_center else ''}\n\n" \
                    f"type=transaction\n" \
                    f"transaction_id={tx_id}\n" \
                    f"tx_node_id={local}\n" \
                    f"deg={deg.get(int(gid), 0)}"
        else:
            ev = ent_local_lookup.get((t, int(local)), f"{t}::{local}")
            label = _format_entity_label(t, ev, max_len=26)
            title = f"{start_title if is_center else ''}\n\n" \
                    f"type={t}\n" \
                    f"value={ev}\n" \
                    f"node_id={local}\n" \
                    f"deg={deg.get(int(gid), 0)}"

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

    # add edges with labels (tooltip)
    for u, v, lab in e_keep:
        if int(u) == int(start_gid) or int(v) == int(start_gid):
            net.add_edge(str(u), str(v), width=2, title=lab)
        else:
            net.add_edge(str(u), str(v), width=1, title=lab)

    if args.show_physics:
        net.show_buttons(filter_=["physics"])

    net.write_html(str(out_path))

    # --- postprocess: freeze physics after stabilization
    html = out_path.read_text(encoding="utf-8")
    needles = [
        "network = new vis.Network(container, data, options);",
        "new vis.Network(container, data, options);",
    ]
    for needle in needles:
        if needle in html:
            html = html.replace(
                needle,
                needle + """
                network.once("stabilizationIterationsDone", function () {
                  network.setOptions({physics: {enabled: false}});
                });
                """,
                1
            )
            out_path.write_text(html, encoding="utf-8")
            break

    print(f"[A11+] Saved: {out_path.as_posix()}")
    print(f"[A11+] {start_title}")
    print(f"[A11+] types_offsets={offsets}, counts={counts}")
    print(f"[A11+] nodes={len(keep)}, edges={len(e_keep)}")


if __name__ == "__main__":
    main()