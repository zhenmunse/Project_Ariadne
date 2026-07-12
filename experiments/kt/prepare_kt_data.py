"""Build canonical concept sessions and the student split for KT distillation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_dag
from experiments.kt.artifacts import protocol_path, sha256_file, write_json


DEFAULT_INTERACTIONS = ROOT / "data" / "processed" / "cleaned_interactions.csv"
DEFAULT_MAPPING = ROOT / "data" / "question_concept_mapping_final.csv"
DEFAULT_DAG = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
DEFAULT_SPEC = ROOT / "documents" / "kt_set_adapter_spec.md"
DEFAULT_OUTPUT = ROOT / "data" / "kt_set"
SEED = 42
CORRECT_THRESHOLD = 0.8
SESSION_COLUMNS = [
    "student_id",
    "session_index",
    "source_order",
    "timestamp",
    "target_node",
    "session_score",
    "correct",
    "split",
]


def _load_mapping(path: Path) -> dict[int, int]:
    mapping = pd.read_csv(path, usecols=["item_id", "concept_id"])
    if mapping["item_id"].duplicated().any():
        duplicates = sorted(mapping.loc[mapping["item_id"].duplicated(), "item_id"].unique())
        raise ValueError(f"item_id must map exactly once; duplicates: {duplicates[:10]}")
    return dict(
        zip(mapping["item_id"].astype(int), mapping["concept_id"].astype(int))
    )


def _canonical_interactions(path: Path, item_to_node: dict[int, int]) -> pd.DataFrame:
    interactions = pd.read_csv(path)
    required = {"user_id", "item_id", "is_correct", "timestamp"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"cleaned interactions missing columns: {missing}")
    interactions = interactions.copy()
    interactions["source_order"] = np.arange(len(interactions), dtype=np.int64)
    interactions = interactions[interactions["item_id"].isin(item_to_node)].copy()
    interactions["target_node"] = interactions["item_id"].map(item_to_node).astype(int)
    if not interactions["is_correct"].isin([0, 1]).all():
        raise ValueError("interaction correctness must be binary")
    interactions["student_id"] = interactions["user_id"].astype(str)
    return interactions.sort_values(
        ["student_id", "timestamp", "source_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def aggregate_concept_sessions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consecutive same-concept interactions for each student."""
    rows: list[dict[str, Any]] = []
    for student_id, student_rows in interactions.groupby("student_id", sort=False):
        session_index = 0
        current_node: int | None = None
        scores: list[int] = []
        first_timestamp: str | None = None
        first_source_order: int | None = None

        def flush() -> None:
            nonlocal session_index, scores
            if current_node is None:
                return
            score = float(np.mean(scores))
            rows.append(
                {
                    "student_id": str(student_id),
                    "session_index": session_index,
                    "source_order": first_source_order,
                    "timestamp": first_timestamp,
                    "target_node": current_node,
                    "session_score": score,
                    "correct": int(score >= CORRECT_THRESHOLD),
                }
            )
            session_index += 1
            scores = []

        for event in student_rows.itertuples(index=False):
            node = int(event.target_node)
            if current_node is None or node != current_node:
                flush()
                current_node = node
                first_timestamp = str(event.timestamp)
                first_source_order = int(event.source_order)
            scores.append(int(event.is_correct))
        flush()

    sessions = pd.DataFrame(rows)
    return sessions[
        [
            "student_id",
            "session_index",
            "source_order",
            "timestamp",
            "target_node",
            "session_score",
            "correct",
        ]
    ]


def canonical_student_split(student_ids: list[str], seed: int = SEED) -> dict[str, Any]:
    """Apply the frozen first-occurrence 80/10/10 student split."""
    users = np.asarray(student_ids, dtype=object).copy()
    np.random.default_rng(seed).shuffle(users)
    split_size = int(len(users) * 0.1)
    return {
        "seed": seed,
        "train": [str(user) for user in users[2 * split_size :]],
        "validation": [str(user) for user in users[split_size : 2 * split_size]],
        "test": [str(user) for user in users[:split_size]],
    }


def validate_sessions(
    sessions: pd.DataFrame,
    split: dict[str, Any],
    dag_nodes: set[int],
) -> None:
    """Enforce all frozen split, chronology, and session invariants."""
    split_sets = {name: set(split[name]) for name in ("train", "validation", "test")}
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("student split sets must be pairwise disjoint")
    all_students = set(sessions["student_id"])
    if set().union(*split_sets.values()) != all_students:
        raise ValueError("student split union must equal all session students")
    if sessions.groupby("student_id")["split"].nunique().max() != 1:
        raise ValueError("a student occurs in multiple splits")
    expected_indices = sessions.groupby("student_id", sort=False).cumcount()
    if not np.array_equal(sessions["session_index"].to_numpy(), expected_indices.to_numpy()):
        raise ValueError("session_index must be consecutive in chronological order")
    parsed_timestamps = pd.to_datetime(sessions["timestamp"], utc=True, errors="raise")
    for positions in sessions.groupby("student_id", sort=False).indices.values():
        if not parsed_timestamps.iloc[positions].is_monotonic_increasing:
            raise ValueError("concept sessions are not in chronological timestamp order")
    if not set(sessions["target_node"].astype(int)).issubset(dag_nodes):
        raise ValueError("concept sessions contain unknown DAG nodes")
    if not sessions["session_score"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("session_score must be in [0, 1]")
    if not sessions["correct"].isin([0, 1]).all():
        raise ValueError("correct must be binary")
    expected_correct = (sessions["session_score"] >= CORRECT_THRESHOLD).astype(int)
    if not np.array_equal(sessions["correct"].to_numpy(), expected_correct.to_numpy()):
        raise ValueError("correct does not match the frozen threshold")


def prepare_kt_data(
    *,
    interactions_path: Path = DEFAULT_INTERACTIONS,
    mapping_path: Path = DEFAULT_MAPPING,
    dag_path: Path = DEFAULT_DAG,
    spec_path: Path = DEFAULT_SPEC,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Generate all Commit-1 KT preprocessing artifacts."""
    item_to_node = _load_mapping(mapping_path)
    dag_nodes, _ = load_dag(dag_path)
    unknown_mapping_nodes = sorted(set(item_to_node.values()) - set(dag_nodes))
    if unknown_mapping_nodes:
        raise ValueError(f"mapping references unknown DAG nodes: {unknown_mapping_nodes}")

    interactions = _canonical_interactions(interactions_path, item_to_node)
    sessions = aggregate_concept_sessions(interactions)
    first_occurrence_students = sessions["student_id"].drop_duplicates().tolist()
    split = canonical_student_split(first_occurrence_students)
    student_to_split = {
        student: name
        for name in ("train", "validation", "test")
        for student in split[name]
    }
    sessions["split"] = sessions["student_id"].map(student_to_split)
    sessions = sessions[SESSION_COLUMNS]
    validate_sessions(sessions, split, set(dag_nodes))

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "student_split.json"
    sessions_path = output_dir / "concept_sessions.parquet"
    manifest_path = output_dir / "preprocessing_manifest.json"
    write_json(split_path, split)
    sessions.to_parquet(sessions_path, engine="pyarrow", index=False)

    split_counts = {
        name: {
            "students": len(split[name]),
            "sessions": int((sessions["split"] == name).sum()),
        }
        for name in ("train", "validation", "test")
    }
    manifest = {
        "schema_version": 1,
        "generation_command": "python experiments/kt/prepare_kt_data.py",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "split_algorithm": {
            "student_order": "first occurrence after canonical session aggregation",
            "rng": "numpy.random.default_rng",
            "seed": SEED,
            "test_fraction": 0.1,
            "validation_fraction": 0.1,
            "rounding": "floor",
        },
        "session_algorithm": {
            "unit": "consecutive same-node interactions per student",
            "score": "arithmetic mean of binary interaction correctness",
            "correct_rule": "1 iff session_score >= 0.8",
            "tie_break": ["timestamp", "source_order"],
        },
        "mastery_rule": {
            "threshold": 0.8,
            "consecutive": 3,
            "irreversible": True,
            "projection": "largest prerequisite-closed subset without adding ancestors",
        },
        "sources": {
            "cleaned_interactions": {
                "path": protocol_path(interactions_path),
                "sha256": sha256_file(interactions_path),
            },
            "question_concept_mapping": {
                "path": protocol_path(mapping_path),
                "sha256": sha256_file(mapping_path),
            },
            "dag": {"path": protocol_path(dag_path), "sha256": sha256_file(dag_path)},
            "adapter_spec": {
                "path": protocol_path(spec_path),
                "sha256": sha256_file(spec_path),
            },
        },
        "artifacts": {
            "student_split": {
                "path": protocol_path(split_path),
                "sha256": sha256_file(split_path),
            },
            "concept_sessions": {
                "path": protocol_path(sessions_path),
                "sha256": sha256_file(sessions_path),
                "format": "parquet",
                "rows": len(sessions),
                "columns": SESSION_COLUMNS,
                "dtypes": {column: str(dtype) for column, dtype in sessions.dtypes.items()},
            },
        },
        "counts": {
            "mapped_interactions": len(interactions),
            "students": len(first_occurrence_students),
            "sessions": len(sessions),
            "splits": split_counts,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare_kt_data(output_dir=args.output_dir)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
