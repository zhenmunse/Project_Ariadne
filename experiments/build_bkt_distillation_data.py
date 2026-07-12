"""Build deterministic BKT teacher prefix examples and grouped tuples."""

from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
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
    mastery_state_before_prefix,
)
from experiments.train_bkt_teacher import required_nodes_from_manifest
from src.oracle_core.bkt_teacher import BKTTeacher


DEFAULT_SESSIONS = ROOT / "data" / "kt_set" / "concept_sessions.parquet"
DEFAULT_SPLIT = ROOT / "data" / "kt_set" / "student_split.json"
DEFAULT_PREPROCESSING_MANIFEST = ROOT / "data" / "kt_set" / "preprocessing_manifest.json"
DEFAULT_PARAMETERS = ROOT / "artifacts" / "bkt_set" / "bkt_parameters.json"
DEFAULT_POOLED_PARAMETERS = ROOT / "artifacts" / "bkt_set" / "pooled_bkt_parameters.json"
DEFAULT_COVERAGE = ROOT / "artifacts" / "bkt_set" / "bkt_coverage.json"
DEFAULT_TEACHER_METADATA = ROOT / "artifacts" / "bkt_set" / "bkt_teacher_metadata.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "bkt_set"
MASTERY_THRESHOLD = 0.8
MASTERY_CONSECUTIVE = 3


def _mastery_string(state: tuple[int, ...]) -> str:
    return json.dumps(list(state), separators=(",", ":"))


def _mastery_mask(state: tuple[int, ...], nodes: list[int]) -> str:
    state_set = set(state)
    return "".join("1" if node in state_set else "0" for node in nodes)


def build_prefix_examples(
    sessions: pd.DataFrame,
    *,
    split_name: str,
    teacher: BKTTeacher,
    required_nodes: list[int],
    dag_nodes: list[int],
    ancestors: dict[int, frozenset[int]],
) -> pd.DataFrame:
    """Query every required target before each real session observation."""
    selected = sessions[sessions["split"] == split_name]
    columns: dict[str, list[Any]] = {
        "student_id": [],
        "prefix_index": [],
        "split": [],
        "mastery_state": [],
        "mastery_mask": [],
        "target_node": [],
        "teacher_probability": [],
    }
    for student_id, student_rows in selected.groupby("student_id", sort=True):
        student_rows = student_rows.sort_values("session_index", kind="mergesort")
        concept_sessions = [
            ConceptSession(int(row.target_node), float(row.session_score))
            for row in student_rows.itertuples(index=False)
        ]
        student_teacher = teacher.new_student_state()
        for prefix_index, current in enumerate(student_rows.itertuples(index=False)):
            state = mastery_state_before_prefix(
                concept_sessions,
                prefix_index,
                ancestors=ancestors,
                threshold=MASTERY_THRESHOLD,
                consecutive=MASTERY_CONSECUTIVE,
            )
            canonical_state = canonical_mastery_tuple(state)
            state_string = _mastery_string(canonical_state)
            state_mask = _mastery_mask(canonical_state, dag_nodes)
            probabilities = [student_teacher.query(node) for node in required_nodes]
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
                raise ValueError("BKT teacher returned an invalid probability")
            count = len(required_nodes)
            columns["student_id"].extend([str(student_id)] * count)
            columns["prefix_index"].extend([prefix_index] * count)
            columns["split"].extend([split_name] * count)
            columns["mastery_state"].extend([state_string] * count)
            columns["mastery_mask"].extend([state_mask] * count)
            columns["target_node"].extend(required_nodes)
            columns["teacher_probability"].extend(probabilities)

            observed_node = int(current.target_node)
            if observed_node in teacher.parameters:
                student_teacher.observe(observed_node, int(current.correct))
    result = pd.DataFrame(columns)
    return result.sort_values(
        ["student_id", "prefix_index", "target_node"], kind="mergesort"
    ).reset_index(drop=True)


def group_prefix_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate identical state-target inputs and sort by mastery bit vector."""
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


def _validate_examples(
    examples: pd.DataFrame,
    grouped: pd.DataFrame,
    *,
    split_name: str,
    allowed_students: set[str],
    required_nodes: list[int],
    ancestors: dict[int, frozenset[int]],
) -> None:
    if set(examples["split"]) != {split_name}:
        raise ValueError(f"{split_name} examples contain another split")
    if not set(examples["student_id"]).issubset(allowed_students):
        raise ValueError(f"{split_name} examples contain disallowed students")
    if set(examples["target_node"].astype(int)) != set(required_nodes):
        raise ValueError(f"{split_name} examples do not cover every required target")
    if not np.isfinite(examples["teacher_probability"]).all():
        raise ValueError("teacher probabilities must be finite")
    if not examples["teacher_probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("teacher probabilities must be in [0, 1]")
    for encoded in examples["mastery_state"].drop_duplicates():
        state = frozenset(json.loads(encoded))
        if any(node not in ancestors for node in state):
            raise ValueError("mastery state contains an unknown node")
        if any(not ancestors[node].issubset(state) for node in state):
            raise ValueError("mastery state is not prerequisite-closed")
    if int(grouped["count"].sum()) != len(examples):
        raise ValueError("grouped tuple counts do not sum to raw tuple count")
    recomputed = group_prefix_examples(examples)
    pd.testing.assert_frame_equal(grouped, recomputed, check_exact=True)


def _split_statistics(
    sessions: pd.DataFrame,
    examples: pd.DataFrame,
    grouped: pd.DataFrame,
    split_name: str,
    required_nodes: list[int],
) -> dict[str, Any]:
    selected_sessions = sessions[sessions["split"] == split_name]
    raw_per_target = examples.groupby("target_node").size()
    grouped_per_target = grouped.groupby("target_node").size()
    return {
        "students": int(selected_sessions["student_id"].nunique()),
        "prefixes": int(len(selected_sessions)),
        "included_empty_prefixes": int(selected_sessions["student_id"].nunique()),
        "excluded_empty_prefixes": 0,
        "unique_mastery_states": int(examples["mastery_state"].nunique()),
        "raw_tuple_count": len(examples),
        "grouped_tuple_count": len(grouped),
        "required_target_coverage": [
            int(node) for node in sorted(examples["target_node"].astype(int).unique())
        ],
        "per_target_raw_count": {
            str(node): int(raw_per_target.get(node, 0)) for node in required_nodes
        },
        "per_target_grouped_count": {
            str(node): int(grouped_per_target.get(node, 0)) for node in required_nodes
        },
        "teacher_probability_min": float(examples["teacher_probability"].min()),
        "teacher_probability_max": float(examples["teacher_probability"].max()),
    }


def build_bkt_distillation_data(
    *,
    sessions_path: Path = DEFAULT_SESSIONS,
    split_path: Path = DEFAULT_SPLIT,
    preprocessing_manifest_path: Path = DEFAULT_PREPROCESSING_MANIFEST,
    parameters_path: Path = DEFAULT_PARAMETERS,
    pooled_parameters_path: Path = DEFAULT_POOLED_PARAMETERS,
    coverage_path: Path = DEFAULT_COVERAGE,
    teacher_metadata_path: Path = DEFAULT_TEACHER_METADATA,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, str]:
    sessions = pd.read_parquet(sessions_path)
    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    if set(split["test"]) & set(sessions.loc[sessions["split"] != "test", "student_id"]):
        raise ValueError("test students leaked into a non-test split")

    manifest = load_manifest()
    required_nodes = required_nodes_from_manifest(manifest)
    teacher = BKTTeacher.from_artifact(parameters_path)
    if set(teacher.parameters) != set(required_nodes):
        raise ValueError("BKT teacher parameter coverage does not match required targets")
    dag_nodes, dag_edges = load_dag(
        ROOT / "data" / "ecs32a_dag_required_full_v1.json"
    )
    ancestors = ancestor_map(dag_nodes, dag_edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    statistics: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        examples = build_prefix_examples(
            sessions,
            split_name=split_name,
            teacher=teacher,
            required_nodes=required_nodes,
            dag_nodes=dag_nodes,
            ancestors=ancestors,
        )
        grouped = group_prefix_examples(examples)
        _validate_examples(
            examples,
            grouped,
            split_name=split_name,
            allowed_students=set(split[split_name]),
            required_nodes=required_nodes,
            ancestors=ancestors,
        )
        raw_path = output_dir / f"{split_name}_prefix_examples.parquet"
        grouped_path = output_dir / f"{split_name}_grouped_tuples.parquet"
        examples.to_parquet(raw_path, engine="pyarrow", index=False)
        grouped.to_parquet(grouped_path, engine="pyarrow", index=False)
        artifact_paths[f"{split_name}_prefix_examples"] = raw_path
        artifact_paths[f"{split_name}_grouped_tuples"] = grouped_path
        statistics[split_name] = _split_statistics(
            sessions, examples, grouped, split_name, required_nodes
        )

    metadata_path = output_dir / "distillation_metadata.json"
    source_paths = {
        "concept_sessions": sessions_path,
        "student_split": split_path,
        "preprocessing_manifest": preprocessing_manifest_path,
        "bkt_parameters": parameters_path,
        "pooled_bkt_parameters": pooled_parameters_path,
        "bkt_coverage": coverage_path,
        "bkt_teacher_metadata": teacher_metadata_path,
        "adapter_spec": ROOT / "documents" / "kt_set_adapter_spec.md",
    }
    with coverage_path.open("r", encoding="utf-8") as file:
        coverage = json.load(file)
    metadata = {
        "teacher": "BKT",
        "teacher_parameterization": "concept_specific_with_pooled_zero_observation_backoff",
        "manifest_hash": manifest_hash(manifest),
        "required_nodes": required_nodes,
        "test_students_used": 0,
        "mastery": {
            "threshold": MASTERY_THRESHOLD,
            "consecutive": MASTERY_CONSECUTIVE,
            "irreversible": True,
            "projection": "largest prerequisite-closed subset without adding ancestors",
            "state_encoding": "compact sorted-node JSON plus 61-bit manifest-order mask",
        },
        "prefix_protocol": {
            "current_session_excluded": True,
            "teacher_query_read_only": True,
            "empty_prefix": "included once per student",
            "targets_per_prefix": len(required_nodes),
        },
        "coverage": coverage,
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
            for name, path in artifact_paths.items()
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "generation_command": "python experiments/build_bkt_distillation_data.py",
    }
    metadata["tuple_collection_hash"] = hashlib.sha256(
        canonical_json_bytes(
            {
                name: metadata["artifacts"][name]["sha256"]
                for name in sorted(metadata["artifacts"])
            }
        )
    ).hexdigest()
    write_json(metadata_path, metadata)
    return {
        **{name: sha256_file(path) for name, path in artifact_paths.items()},
        "distillation_metadata": sha256_file(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    hashes = build_bkt_distillation_data(output_dir=args.output_dir)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
