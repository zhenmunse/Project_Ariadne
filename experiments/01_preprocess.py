"""
experiments/01_preprocess.py
============================
Run the full data preprocessing pipeline:
  1. (Optional) Generate toy raw data if no real CSV exists
  2. Build knowledge DAG  ->  data/processed/graph.pkl
  3. Preprocess raw logs  ->  data/processed/sessions.pkl
                              data/processed/train_sessions.pkl
  4. Print summary & first samples for verification

Usage:
    python experiments/01_preprocess.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import yaml

# ---------- make project root importable -------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_engine.graph_builder import (
    build_and_save_graph,
    TOY_ITEM2NODE,
    TOY_EDGES,
)
from src.data_engine.preprocessor import run_preprocess


# ------------------------------------------------------------------
# Toy data generator
# ------------------------------------------------------------------

def generate_toy_data(out_csv: str, item2node: dict, seed: int = 42):
    """Create a small but non-trivial toy interaction log.

    Design choices for verification:
      - 3 users, ~8-15 interactions each
      - Includes concurrent events (same timestamp, different items)
      - Includes items NOT in the mapping (item_id=999) to test filtering
      - Includes repeated same-concept interactions to form sessions
    """
    rng = np.random.RandomState(seed)
    valid_items = sorted(item2node.keys())

    rows = []

    # ---- User 1 : clean sequential learner ----
    # ts=1: items 100,101 (both concept 0) -- concurrent, should serialize by item_id
    rows.append((1, 101, 1, 1))   # deliberately out of order
    rows.append((1, 100, 1, 1))   # same ts -> should come first after flattening
    # ts=2: item 200 (concept 1)
    rows.append((1, 200, 1, 2))
    rows.append((1, 201, 0, 2))   # concurrent with above
    # ts=3: item 300 (concept 2)
    rows.append((1, 300, 1, 3))
    rows.append((1, 301, 1, 3))
    # ts=4: item 400 (concept 3)
    rows.append((1, 400, 0, 4))
    # ts=5: back to concept 0 (new session)
    rows.append((1, 100, 1, 5))

    # ---- User 2 : has invalid items ----
    rows.append((2, 999, 1, 1))   # not in graph -> should be filtered
    rows.append((2, 100, 0, 2))
    rows.append((2, 100, 1, 3))   # concept 0 again but different ts -> new session
    rows.append((2, 200, 1, 4))
    rows.append((2, 300, 0, 5))
    rows.append((2, 300, 1, 5))   # concurrent with above, same concept

    # ---- User 3 : longer sequence ----
    rows.append((3, 100, 1, 1))
    rows.append((3, 101, 1, 1))   # concurrent
    rows.append((3, 200, 1, 2))
    rows.append((3, 201, 1, 2))
    rows.append((3, 300, 0, 3))
    rows.append((3, 301, 0, 3))
    rows.append((3, 400, 1, 4))
    rows.append((3, 400, 0, 5))   # concept 3 again, new ts -> new session

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "is_correct", "timestamp"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[toy] Generated toy raw data -> {out_csv}  ({len(df)} rows)")
    return out_csv


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # --- Load config -------------------------------------------------
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_log_path = os.path.join(ROOT, cfg["data"]["raw_log"])
    item2node_path = os.path.join(ROOT, cfg["data"]["item2node"])
    dag_edges_path = os.path.join(ROOT, cfg["data"]["dag_edges"])
    processed_dir  = os.path.join(ROOT, cfg["data"]["processed_dir"])
    seed = cfg["seed"]

    # --- If no real data, generate toy data --------------------------
    use_toy = not os.path.isfile(raw_log_path)
    if use_toy:
        print("=" * 60)
        print("No real data found. Generating toy dataset.")
        print("=" * 60)
        toy_csv = os.path.join(ROOT, "data", "raw", "toy_logs.csv")
        generate_toy_data(toy_csv, TOY_ITEM2NODE, seed=seed)
        raw_log_path = toy_csv
        # toy mode -> use built-in mappings (ignore file paths)
        item2node_path = None
        dag_edges_path = None

    # --- Step A: Build graph -----------------------------------------
    print("\n" + "=" * 60)
    print("Building knowledge DAG ...")
    print("=" * 60)
    graph_data = build_and_save_graph(
        output_dir=processed_dir,
        item2node_path=item2node_path,
        dag_edges_path=dag_edges_path,
    )

    # --- Step B: Preprocess logs -------------------------------------
    print("\n" + "=" * 60)
    print("Preprocessing raw logs ...")
    print("=" * 60)
    sess_df, train_samples = run_preprocess(
        raw_path=raw_log_path,
        item2node=graph_data["item2node"],
        output_dir=processed_dir,
    )

    # --- Verification prints -----------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # 1) graph.pkl sanity
    graph_pkl = os.path.join(processed_dir, "graph.pkl")
    with open(graph_pkl, "rb") as f:
        g = pickle.load(f)
    print(f"\ngraph.pkl keys: {sorted(g.keys())}")
    print(f"  node_ids        : {g['node_ids']}")
    print(f"  edge_index shape: {g['edge_index'].shape}")
    print(f"  adjacency shape : {g['adjacency'].shape}")

    # 2) sessions overview
    print(f"\nSessions DataFrame ({len(sess_df)} rows):")
    print(sess_df.to_string(index=False))

    # 3) train_sessions.pkl first 3 samples
    print(f"\ntrain_sessions.pkl: {len(train_samples)} samples")
    print("First 3 samples:")
    for i, (hist, tgt, lbl) in enumerate(train_samples[:3]):
        print(f"  [{i}] history={hist}, target_node={tgt}, label={lbl:.4f}")

    # 4) Assertions
    print("\n--- Running assertions ---")
    # labels in [0, 1]
    for hist, tgt, lbl in train_samples:
        assert 0.0 <= lbl <= 1.0, f"Label out of range: {lbl}"
    print("  [OK] All labels in [0, 1]")

    # session_id monotonic per user
    for uid, udf in sess_df.groupby("user_id"):
        sids = udf["session_id"].tolist()
        assert sids == sorted(sids), f"session_id not monotonic for user {uid}"
        assert sids == list(range(len(sids))), f"session_id not contiguous for user {uid}"
    print("  [OK] session_id monotonic & contiguous per user")

    # graph is DAG
    import networkx as nx
    assert nx.is_directed_acyclic_graph(g["nx_dag"]), "Graph is not a DAG!"
    print("  [OK] Graph is a valid DAG")

    # no filtered items leak through
    valid_nodes = set(g["item2node"].values())
    for _, row in sess_df.iterrows():
        assert int(row["node_id"]) in valid_nodes
    print("  [OK] All session node_ids are in the graph")

    print("\n=== Step 1 COMPLETE ===")


if __name__ == "__main__":
    main()
