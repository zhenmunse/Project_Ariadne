"""
graph_builder.py  --  Build a knowledge-prerequisite DAG and persist it.

Output artifact: graph.pkl  (dict with keys listed below)
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Toy defaults (used when no external files are supplied)
# ------------------------------------------------------------------
TOY_ITEM2NODE: Dict[int, int] = {
    100: 0, 101: 0,   # two items map to concept 0
    200: 1, 201: 1,   # concept 1
    300: 2, 301: 2,   # concept 2
    400: 3,            # concept 3
}

TOY_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # concept 0 -> 1
    (0, 2),  # concept 0 -> 2
    (1, 3),  # concept 1 -> 3
    (2, 3),  # concept 2 -> 3
]


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def load_item2node(path: Optional[str] = None) -> Dict[int, int]:
    """Return item_id -> node_id mapping.

    If *path* is given it must be a CSV with columns (item_id, node_id).
    Otherwise the built-in toy mapping is returned.
    """
    if path is not None and os.path.isfile(path):
        df = pd.read_csv(path)
        return dict(zip(df["item_id"].astype(int), df["node_id"].astype(int)))
    return dict(TOY_ITEM2NODE)


def load_edges(path: Optional[str] = None) -> List[Tuple[int, int]]:
    """Return list of (src, dst) prerequisite edges.

    If *path* is given it must be a CSV with columns (src, dst).
    Otherwise the built-in toy edges are returned.
    """
    if path is not None and os.path.isfile(path):
        df = pd.read_csv(path)
        return list(zip(df["src"].astype(int), df["dst"].astype(int)))
    return list(TOY_EDGES)


# ------------------------------------------------------------------
# Core builder
# ------------------------------------------------------------------

def build_graph(
    item2node: Dict[int, int],
    edges: List[Tuple[int, int]],
) -> dict:
    """Build the knowledge DAG and return a dict with all required fields.

    Raises ValueError if the graph contains a cycle.
    """
    # --- Collect all node ids (from mapping + edges) -----------------
    node_set = set(item2node.values())
    for s, d in edges:
        node_set.add(s)
        node_set.add(d)
    node_ids: List[int] = sorted(node_set)

    node_id_to_idx: Dict[int, int] = {nid: i for i, nid in enumerate(node_ids)}
    idx_to_node_id: List[int] = list(node_ids)
    n_nodes = len(node_ids)

    # --- NetworkX DAG ------------------------------------------------
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(
            f"Graph is NOT a DAG!  Found cycle(s): {cycles[:5]}"
        )

    # --- edge_index  (idx-based, shape [2, E]) ----------------------
    src_idx = [node_id_to_idx[s] for s, _ in edges]
    dst_idx = [node_id_to_idx[d] for _, d in edges]
    edge_index = np.array([src_idx, dst_idx], dtype=np.int64)  # [2, E]

    # --- adjacency matrix (idx-based) --------------------------------
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for si, di in zip(src_idx, dst_idx):
        adj[si, di] = 1.0

    graph_data = {
        "node_ids": node_ids,
        "node_id_to_idx": node_id_to_idx,
        "idx_to_node_id": idx_to_node_id,
        "edge_index": edge_index,
        "adjacency": adj,
        "nx_dag": G,
        "item2node": item2node,
    }
    return graph_data


# ------------------------------------------------------------------
# Build + save convenience wrapper
# ------------------------------------------------------------------

def build_and_save_graph(
    output_dir: str,
    item2node_path: Optional[str] = None,
    dag_edges_path: Optional[str] = None,
) -> dict:
    """End-to-end: load sources -> build -> save graph.pkl -> return."""
    item2node = load_item2node(item2node_path)
    edges = load_edges(dag_edges_path)

    graph_data = build_graph(item2node, edges)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "graph.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"[graph_builder] Saved graph.pkl -> {out_path}")
    print(f"  nodes : {graph_data['node_ids']}")
    print(f"  edges : {graph_data['edge_index'].shape[1]}")
    print(f"  items : {len(graph_data['item2node'])} item->node mappings")
    return graph_data
