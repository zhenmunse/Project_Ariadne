"""Build frozen train-only aggregate statistics for LLM-Full."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from experiments.llm.artifacts import load_json, sha256_file, value_hash


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments" / "llm" / "protocol.json"
SESSIONS_PATH = ROOT / "data" / "kt_set" / "concept_sessions.parquet"
SPLIT_PATH = ROOT / "data" / "kt_set" / "student_split.json"
DAG_PATH = ROOT / "data" / "ecs32a_dag_required_full_v1.json"


def compute_statistics_values(
    sessions: pd.DataFrame,
    nodes: list[int],
    threshold: float,
) -> list[dict[str, Any]]:
    """Compute values after selecting train rows; non-train rows are irrelevant."""
    train = sessions.loc[sessions["split"] == "train", ["target_node", "session_score"]].copy()
    train["correct"] = (train["session_score"] >= threshold).astype(int)
    grouped = train.groupby("target_node", sort=True)["correct"].agg(["count", "sum"])
    values = []
    for node in nodes:
        if node not in grouped.index:
            count = 0
            rate = None
        else:
            count = int(grouped.loc[node, "count"])
            rate = float(grouped.loc[node, "sum"] / count)
        values.append({
            "real_node_id": node,
            "attempt_count": count,
            "success_rate": rate,
        })
    return values


def build_full_statistics() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    threshold = float(protocol["statistics"]["success_rate"]["correctness_threshold"])
    if threshold != 0.8:
        raise ValueError("Task 15 success threshold must be 0.8")
    sessions = pd.read_parquet(SESSIONS_PATH)
    split = load_json(SPLIT_PATH)
    expected_columns = {"student_id", "target_node", "session_score", "correct", "split"}
    if not expected_columns.issubset(sessions.columns):
        raise ValueError("Canonical concept sessions are missing required columns")
    expected_correct = (sessions["session_score"] >= threshold).astype(int)
    if not expected_correct.equals(sessions["correct"].astype(int)):
        raise ValueError("Canonical correct column disagrees with frozen threshold")
    train = sessions.loc[sessions["split"] == "train", ["target_node", "correct"]].copy()
    if set(sessions["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Unexpected canonical split labels")
    expected_train_students = set(split["train"])
    actual_train_students = set(sessions.loc[sessions["split"] == "train", "student_id"])
    if actual_train_students != expected_train_students:
        raise ValueError("Canonical train sessions disagree with frozen student split")
    if expected_train_students & (set(split["validation"]) | set(split["test"])):
        raise ValueError("Frozen student split is not disjoint")
    dag = load_json(DAG_PATH)
    nodes = sorted(int(node["node_id"]) for node in dag["nodes"])
    values = compute_statistics_values(sessions, nodes, threshold)
    return {
        "schema_version": 1,
        "protocol_hash": sha256_file(PROTOCOL_PATH),
        "concept_sessions_hash": sha256_file(SESSIONS_PATH),
        "student_split_hash": sha256_file(SPLIT_PATH),
        "dag_hash": sha256_file(DAG_PATH),
        "generation_source_hash": sha256_file(Path(__file__)),
        "split": "train",
        "unit": "canonical_concept_session",
        "correctness_threshold": threshold,
        "render_decimal_places": int(protocol["statistics"]["success_rate"]["render_decimal_places"]),
        "train_session_count": int(len(train)),
        "statistics_values_hash": value_hash(values),
        "nodes": values,
    }
