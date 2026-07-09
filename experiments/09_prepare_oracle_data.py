"""Prepare real ECS32A data for MonotonicOracle training."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

from src.data_engine.graph_builder import build_graph
from src.data_engine.preprocessor import (
    _aggregate_sessions,
    _build_train_samples,
    _flatten,
    _load_and_filter,
    _map_to_nodes,
)


def save_pickle(value: object, name: str) -> None:
    with (OUTPUT / name).open("wb") as file:
        pickle.dump(value, file)


def main() -> None:
    interactions = ROOT / "data" / "processed" / "cleaned_interactions.csv"
    mapping_path = ROOT / "data" / "question_concept_mapping_final.csv"
    dag_path = ROOT / "data" / "ecs32a_dag_required_full_v1.json"

    mapping = pd.read_csv(mapping_path, usecols=["item_id", "concept_id"])
    if mapping["item_id"].duplicated().any():
        raise ValueError("item_id must map to exactly one concept_id")
    item_to_node = dict(zip(mapping["item_id"].astype(int), mapping["concept_id"].astype(int)))

    with dag_path.open(encoding="utf-8") as file:
        dag = json.load(file)
    edges = [(int(edge["src"]), int(edge["dst"])) for edge in dag["edges"]]
    graph = build_graph(
        item_to_node,
        edges,
    )
    if len(graph["node_ids"]) != dag["node_count"] or len(edges) != dag["edge_count"]:
        raise ValueError("DAG metadata does not match its nodes or edges")

    rows = _load_and_filter(str(interactions), item_to_node)
    sessions = _aggregate_sessions(_map_to_nodes(_flatten(rows), item_to_node))

    users = sessions["user_id"].drop_duplicates().to_numpy()
    np.random.default_rng(42).shuffle(users)
    split_size = round(len(users) * 0.1)
    split_users = {
        "test": set(users[:split_size]),
        "valid": set(users[split_size : 2 * split_size]),
        "train": set(users[2 * split_size :]),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_pickle(graph, "graph.pkl")
    save_pickle(sessions, "sessions.pkl")

    all_split_users = set().union(*split_users.values())
    assert len(all_split_users) == len(users)

    for split, user_ids in split_users.items():
        split_sessions = sessions[sessions["user_id"].isin(user_ids)].copy()
        samples = _build_train_samples(split_sessions)
        assert all(
            0.0 <= label <= 1.0
            and target in graph["node_id_to_idx"]
            and all(node in graph["node_id_to_idx"] for node, _ in history)
            for history, target, label in samples
        )
        save_pickle(samples, f"{split}_sessions.pkl")
        print(
            f"{split}: users={len(user_ids)} sessions={len(split_sessions)} "
            f"samples={len(samples)}"
        )

    print(
        f"graph: nodes={len(graph['node_ids'])} edges={graph['edge_index'].shape[1]} "
        f"mapped_items={len(item_to_node)}"
    )
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
