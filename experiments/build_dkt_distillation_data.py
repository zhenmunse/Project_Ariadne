"""Build DKT-teacher prefix examples and grouped set-distillation tuples."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PATH = ROOT / "data" / "kt_set" / "concept_sessions.parquet"
SPLIT_PATH = ROOT / "data" / "kt_set" / "student_split.json"
PREPROCESSING_PATH = ROOT / "data" / "kt_set" / "preprocessing_manifest.json"
DAG_PATH = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
BKT_SURROGATE_CONFIG_PATH = ROOT / "artifacts" / "bkt_set" / "surrogate_config.json"
DKT_CONFIG_PATH = ROOT / "artifacts" / "dkt_set" / "dkt_config.json"
DKT_CHECKPOINT_PATH = ROOT / "artifacts" / "dkt_set" / "dkt_checkpoint.pt"
DKT_METRICS_PATH = ROOT / "artifacts" / "dkt_set" / "dkt_training_metrics.json"
DKT_INPUT_METADATA_PATH = ROOT / "artifacts" / "dkt_set" / "dkt_input_metadata.json"
DKT_TEACHER_METADATA_PATH = ROOT / "artifacts" / "dkt_set" / "dkt_teacher_metadata.json"
OUTPUT = ROOT / "artifacts" / "dkt_set"
METADATA_PATH = OUTPUT / "distillation_metadata.json"
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_dag, load_manifest, manifest_hash
from experiments.kt.artifacts import (
    canonical_json_bytes,
    protocol_path,
    sha256_file,
    write_json,
)
from experiments.kt.mastery import (
    ConceptSession,
    ancestor_map,
    canonical_mastery_tuple,
    mastery_completion_before_prefix,
    zero_observation_prerequisite_completion,
)
from experiments.train_bkt_teacher import required_nodes_from_manifest
from src.oracle_core.dkt_teacher import FrozenDKTTeacher


MASTERY_THRESHOLD = 0.8
MASTERY_CONSECUTIVE = 3


def _state_string(state: frozenset[int] | tuple[int, ...]) -> str:
    return json.dumps(list(sorted(state)), separators=(",", ":"))


def _state_mask(state: frozenset[int], node_order: list[int]) -> str:
    return "".join("1" if node in state else "0" for node in node_order)


def build_prefix_examples(
    sessions: pd.DataFrame,
    *,
    split_name: str,
    teacher: FrozenDKTTeacher,
    required_nodes: list[int],
    node_order: list[int],
    ancestors: dict[int, frozenset[int]],
    training_observed_nodes: frozenset[int],
) -> pd.DataFrame:
    selected = sessions[sessions["split"] == split_name]
    required_indices = [teacher.node_to_index[node] for node in required_nodes]
    columns: dict[str, list[Any]] = {
        "student_id": [],
        "prefix_index": [],
        "split": [],
        "raw_mastery_state": [],
        "mastery_state": [],
        "completed_ancestors": [],
        "mastery_mask": [],
        "target_node": [],
        "teacher_probability": [],
    }
    for student_id, rows in selected.groupby("student_id", sort=True):
        rows = rows.sort_values("session_index", kind="mergesort")
        concept_sessions = [
            ConceptSession(int(row.target_node), float(row.session_score))
            for row in rows.itertuples(index=False)
        ]
        tokens = [
            teacher.token(int(row.target_node), int(row.correct))
            for row in rows.itertuples(index=False)
        ]
        probability_table = teacher.probability_table(tokens)
        if probability_table.shape != (len(rows) + 1, len(node_order)):
            raise AssertionError("DKT probability table has the wrong shape")
        for prefix_index in range(len(rows)):
            completion = mastery_completion_before_prefix(
                concept_sessions,
                prefix_index,
                ancestors=ancestors,
                training_observed_nodes=training_observed_nodes,
                threshold=MASTERY_THRESHOLD,
                consecutive=MASTERY_CONSECUTIVE,
            )
            probabilities = [
                float(probability_table[prefix_index, index])
                for index in required_indices
            ]
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
                raise ValueError("DKT teacher returned an invalid probability")
            count = len(required_nodes)
            columns["student_id"].extend([str(student_id)] * count)
            columns["prefix_index"].extend([prefix_index] * count)
            columns["split"].extend([split_name] * count)
            columns["raw_mastery_state"].extend(
                [_state_string(completion.raw_mastery)] * count
            )
            columns["mastery_state"].extend([_state_string(completion.state)] * count)
            columns["completed_ancestors"].extend(
                [_state_string(completion.completed_ancestors)] * count
            )
            columns["mastery_mask"].extend(
                [_state_mask(completion.state, node_order)] * count
            )
            columns["target_node"].extend(required_nodes)
            columns["teacher_probability"].extend(probabilities)
    return pd.DataFrame(columns).sort_values(
        ["student_id", "prefix_index", "target_node"], kind="mergesort"
    ).reset_index(drop=True)


def group_prefix_examples(examples: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        examples.groupby(
            ["split", "mastery_mask", "mastery_state", "target_node"],
            sort=False,
            as_index=False,
        )["teacher_probability"]
        .agg(teacher_probability_mean="mean", count="size")
        .sort_values(["mastery_mask", "target_node"], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["count"] = grouped["count"].astype(np.int64)
    return grouped


def validate_examples(
    examples: pd.DataFrame,
    grouped: pd.DataFrame,
    *,
    split_name: str,
    allowed_students: set[str],
    required_nodes: list[int],
    ancestors: dict[int, frozenset[int]],
    training_observed_nodes: frozenset[int],
) -> None:
    if set(examples["split"]) != {split_name}:
        raise ValueError("Distillation examples contain another split")
    if set(examples["student_id"]) != allowed_students:
        raise ValueError("Distillation student set mismatch")
    if set(examples["target_node"].astype(int)) != set(required_nodes):
        raise ValueError("Distillation target coverage mismatch")
    prefix_rows = examples.drop_duplicates(["student_id", "prefix_index"])
    if len(examples) != len(prefix_rows) * len(required_nodes):
        raise ValueError("Every prefix must query every required target")
    for row in prefix_rows.itertuples(index=False):
        raw = frozenset(json.loads(row.raw_mastery_state))
        state = frozenset(json.loads(row.mastery_state))
        expected = zero_observation_prerequisite_completion(
            raw, ancestors, training_observed_nodes
        )
        if state != expected.state:
            raise ValueError("DKT state does not use the frozen mastery rule")
        if any(not ancestors[node].issubset(state) for node in state):
            raise ValueError("DKT mastery state is not prerequisite-closed")
    if int(grouped["count"].sum()) != len(examples):
        raise ValueError("Grouped counts do not reconcile with raw tuples")
    pd.testing.assert_frame_equal(
        grouped, group_prefix_examples(examples), check_exact=True
    )


def _statistics(
    sessions: pd.DataFrame,
    examples: pd.DataFrame,
    grouped: pd.DataFrame,
    split_name: str,
    required_nodes: list[int],
) -> dict:
    prefix_rows = examples.drop_duplicates(["student_id", "prefix_index"])
    return {
        "students": int(sessions[sessions["split"] == split_name]["student_id"].nunique()),
        "prefixes": len(prefix_rows),
        "raw_tuple_count": len(examples),
        "grouped_tuple_count": len(grouped),
        "raw_mastery_state_count": int(prefix_rows["raw_mastery_state"].nunique()),
        "completed_state_count": int(prefix_rows["mastery_state"].nunique()),
        "non_empty_completed_state_count": int(
            sum(value != "[]" for value in prefix_rows["mastery_state"].unique())
        ),
        "states_changed_by_completion": int(
            (prefix_rows["raw_mastery_state"].to_numpy() != prefix_rows["mastery_state"].to_numpy()).sum()
        ),
        "required_target_coverage": sorted(
            int(node) for node in examples["target_node"].unique()
        ),
        "targets_per_prefix": len(required_nodes),
        "teacher_probability_min": float(examples["teacher_probability"].min()),
        "teacher_probability_max": float(examples["teacher_probability"].max()),
    }


def build_dkt_distillation_data() -> dict[str, str]:
    sessions = pd.read_parquet(SESSIONS_PATH)
    with SPLIT_PATH.open("r", encoding="utf-8") as file:
        split = json.load(file)
    with PREPROCESSING_PATH.open("r", encoding="utf-8") as file:
        preprocessing = json.load(file)
    with BKT_SURROGATE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        bkt_surrogate_config = json.load(file)
    teacher = FrozenDKTTeacher.from_artifacts(
        config_path=DKT_CONFIG_PATH,
        checkpoint_path=DKT_CHECKPOINT_PATH,
    )
    manifest = load_manifest()
    required_nodes = required_nodes_from_manifest(manifest)
    node_order, edges = load_dag(DAG_PATH)
    ancestors = ancestor_map(node_order, edges)
    training_observed_nodes = frozenset(
        sessions.loc[sessions["split"] == "train", "target_node"].astype(int)
    )
    if set(split["test"]) & set(sessions.loc[sessions["split"] != "test", "student_id"]):
        raise ValueError("Test student leaked into DKT distillation")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    statistics = {}
    for split_name in ("train", "validation"):
        examples = build_prefix_examples(
            sessions,
            split_name=split_name,
            teacher=teacher,
            required_nodes=required_nodes,
            node_order=node_order,
            ancestors=ancestors,
            training_observed_nodes=training_observed_nodes,
        )
        grouped = group_prefix_examples(examples)
        validate_examples(
            examples,
            grouped,
            split_name=split_name,
            allowed_students=set(split[split_name]),
            required_nodes=required_nodes,
            ancestors=ancestors,
            training_observed_nodes=training_observed_nodes,
        )
        raw_path = OUTPUT / f"{split_name}_prefix_examples.parquet"
        grouped_path = OUTPUT / f"{split_name}_grouped_tuples.parquet"
        examples.to_parquet(raw_path, engine="pyarrow", index=False)
        grouped.to_parquet(grouped_path, engine="pyarrow", index=False)
        paths[f"{split_name}_prefix_examples"] = raw_path
        paths[f"{split_name}_grouped_tuples"] = grouped_path
        statistics[split_name] = _statistics(
            sessions, examples, grouped, split_name, required_nodes
        )

    source_paths = {
        "concept_sessions": SESSIONS_PATH,
        "student_split": SPLIT_PATH,
        "preprocessing_manifest": PREPROCESSING_PATH,
        "dkt_config": DKT_CONFIG_PATH,
        "dkt_checkpoint": DKT_CHECKPOINT_PATH,
        "dkt_training_metrics": DKT_METRICS_PATH,
        "dkt_input_metadata": DKT_INPUT_METADATA_PATH,
        "dkt_teacher_metadata": DKT_TEACHER_METADATA_PATH,
        "adapter_spec": ROOT / "documents" / "kt_set_adapter_spec.md",
    }
    metadata = {
        "schema_version": 1,
        "teacher": "DKT",
        "condition_name": "DKT-derived Set Oracle",
        "manifest_hash": manifest_hash(manifest),
        "required_nodes": required_nodes,
        "test_students_used": 0,
        "prefix_protocol": {
            "current_session_excluded": True,
            "hidden_state_advanced_after_query": True,
            "empty_prefix_included": True,
            "targets_per_prefix": len(required_nodes),
        },
        "mastery": {
            "threshold": MASTERY_THRESHOLD,
            "consecutive": MASTERY_CONSECUTIVE,
            "irreversible": True,
            "projection": "zero-observation prerequisite completion",
            "compression_config_hash": bkt_surrogate_config[
                "compression_config_hash"
            ],
            "zero_observation_nodes_hash": preprocessing["mastery_rule"][
                "zero_observation_nodes_hash"
            ],
            "training_observed_nodes_hash": preprocessing["mastery_rule"][
                "training_observed_nodes_hash"
            ],
        },
        "statistics": statistics,
        "sources": {
            name: {"path": protocol_path(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "artifacts": {
            name: {
                "path": protocol_path(path),
                "sha256": sha256_file(path),
                "rows": int(len(pd.read_parquet(path))),
            }
            for name, path in paths.items()
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "generation_command": "python experiments/build_dkt_distillation_data.py",
    }
    metadata["tuple_collection_hash"] = hashlib.sha256(
        canonical_json_bytes(
            {name: metadata["artifacts"][name]["sha256"] for name in sorted(paths)}
        )
    ).hexdigest()
    write_json(METADATA_PATH, metadata)
    return {
        **{name: sha256_file(path) for name, path in paths.items()},
        "distillation_metadata": sha256_file(METADATA_PATH),
    }


def main() -> None:
    print(json.dumps(build_dkt_distillation_data(), indent=2, sort_keys=True))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
