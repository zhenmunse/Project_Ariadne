"""Fit canonical concept-specific and pooled-backoff BKT teacher parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_manifest, manifest_hash
from experiments.kt.artifacts import (
    canonical_json_bytes,
    protocol_path,
    sha256_file,
    write_json,
)
from src.oracle_core.bkt_teacher import (
    BKTParameters,
    BOUNDS,
    OBJECTIVE_TIE_TOLERANCE,
    OPTIMIZER_OPTIONS,
    PARAMETER_NAMES,
    STARTING_POINTS,
    NUMERICAL_EPSILON,
    fit_bkt_parameters,
)


DEFAULT_SESSIONS = ROOT / "data" / "kt_set" / "concept_sessions.parquet"
DEFAULT_SPLIT = ROOT / "data" / "kt_set" / "student_split.json"
DEFAULT_PREPROCESSING_MANIFEST = ROOT / "data" / "kt_set" / "preprocessing_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "bkt_set"
FROZEN_ZERO_OBSERVATION_NODES = [0, 1, 2, 5, 11, 32, 37, 51]
PARAMETERIZATION = "concept_specific_with_pooled_zero_observation_backoff"


def required_nodes_from_manifest(manifest: dict[str, Any]) -> list[int]:
    return sorted(
        {
            node
            for closure in manifest["closures"]
            for node in closure["sequence_nodes"]
        }
    )


def _sequences_for_node(sessions: pd.DataFrame, node: int) -> dict[str, list[int]]:
    selected = sessions[sessions["target_node"] == node]
    return {
        str(student): group["correct"].astype(int).tolist()
        for student, group in selected.groupby("student_id", sort=True)
    }


def _pooled_sequences(sessions: pd.DataFrame) -> dict[str, list[int]]:
    return {
        f"{student}::{int(node)}": group["correct"].astype(int).tolist()
        for (student, node), group in sessions.groupby(
            ["student_id", "target_node"], sort=True
        )
    }


def _fit_payload(result: Any) -> dict[str, Any]:
    return {
        **result.parameters.to_dict(),
        "objective": result.objective,
        "selected_restart": result.selected_restart,
        "restarts": list(result.restarts),
    }


def train_bkt_teacher(
    *,
    sessions_path: Path = DEFAULT_SESSIONS,
    split_path: Path = DEFAULT_SPLIT,
    preprocessing_manifest_path: Path = DEFAULT_PREPROCESSING_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, str]:
    sessions = pd.read_parquet(sessions_path)
    if set(sessions["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("concept sessions must contain canonical train/validation/test splits")
    train = sessions[sessions["split"] == "train"].copy()
    if train.empty:
        raise ValueError("canonical training sessions are empty")
    if not train["correct"].isin([0, 1]).all():
        raise ValueError("BKT observations must be binary")

    manifest = load_manifest()
    required_nodes = required_nodes_from_manifest(manifest)
    observed_required_nodes = sorted(
        set(required_nodes) & set(train["target_node"].astype(int))
    )
    zero_observation_nodes = sorted(set(required_nodes) - set(observed_required_nodes))
    if zero_observation_nodes != FROZEN_ZERO_OBSERVATION_NODES:
        raise ValueError(
            "Frozen zero-observation node list mismatch: "
            f"expected {FROZEN_ZERO_OBSERVATION_NODES}, found {zero_observation_nodes}"
        )

    pooled_sequences = _pooled_sequences(train)
    pooled_result = fit_bkt_parameters(pooled_sequences)
    concept_results = {
        node: fit_bkt_parameters(_sequences_for_node(train, node))
        for node in observed_required_nodes
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = output_dir / "bkt_parameters.json"
    pooled_path = output_dir / "pooled_bkt_parameters.json"
    coverage_path = output_dir / "bkt_coverage.json"
    metadata_path = output_dir / "bkt_teacher_metadata.json"

    input_hashes = {
        "concept_sessions": sha256_file(sessions_path),
        "student_split": sha256_file(split_path),
        "preprocessing_manifest": sha256_file(preprocessing_manifest_path),
    }
    optimizer = {
        "algorithm": "scipy.optimize.minimize:L-BFGS-B",
        "parameter_order": list(PARAMETER_NAMES),
        "bounds": [list(bounds) for bounds in BOUNDS],
        "starting_points": [list(point) for point in STARTING_POINTS],
        "options": OPTIMIZER_OPTIONS,
        "objective_tie_tolerance": OBJECTIVE_TIE_TOLERANCE,
        "numerical_probability_epsilon": float(NUMERICAL_EPSILON),
        "arithmetic": "cpu_float64",
        "scipy_version": scipy.__version__,
    }
    ordered_pooled_ids = sorted(pooled_sequences)
    pooled_provenance = {
        "parameters": pooled_result.parameters.to_dict(),
        "optimizer": optimizer,
        "restarts": list(pooled_result.restarts),
        "ordered_sequence_ids": ordered_pooled_ids,
        "observation_counts": [len(pooled_sequences[key]) for key in ordered_pooled_ids],
        "input_hashes": input_hashes,
    }
    pooled_parameter_vector_hash = hashlib.sha256(
        canonical_json_bytes(pooled_result.parameters.to_dict())
    ).hexdigest()
    pooled_artifact = {
        **pooled_provenance,
        "objective": pooled_result.objective,
        "selected_restart": pooled_result.selected_restart,
        "sequence_count": len(pooled_sequences),
        "observation_count": int(sum(map(len, pooled_sequences.values()))),
        "pooled_parameter_vector_hash": pooled_parameter_vector_hash,
        "sequence_definition": "independent (student_id, target_node) trajectories",
        "training_split": "train",
    }
    write_json(pooled_path, pooled_artifact)
    pooled_parameter_artifact_hash = sha256_file(pooled_path)

    parameter_rows = []
    pooled_parameters = pooled_result.parameters
    for node in required_nodes:
        if node in concept_results:
            result = concept_results[node]
            parameters = result.parameters
            source = "concept_specific"
            node_sessions = train[train["target_node"] == node]
            fit = _fit_payload(result)
        else:
            parameters = pooled_parameters
            source = "pooled_zero_observation_bkt"
            node_sessions = train.iloc[0:0]
            fit = None
        parameter_values = parameters.to_dict()
        parameter_rows.append(
            {
                "node_id": node,
                **parameter_values,
                "parameter_values_hash": hashlib.sha256(
                    canonical_json_bytes(parameter_values)
                ).hexdigest(),
                "parameter_source": source,
                "train_observations": int(len(node_sessions)),
                "train_students": int(node_sessions["student_id"].nunique()),
                "fit": fit,
            }
        )
    parameter_values_payload = [
        {
            "node_id": entry["node_id"],
            "parameter_source": entry["parameter_source"],
            "prior": entry["prior"],
            "learn": entry["learn"],
            "guess": entry["guess"],
            "slip": entry["slip"],
        }
        for entry in parameter_rows
    ]
    parameter_values_hash = hashlib.sha256(
        canonical_json_bytes(parameter_values_payload)
    ).hexdigest()
    parameter_artifact = {
        "teacher_parameterization": PARAMETERIZATION,
        "required_nodes": required_nodes,
        "parameters": parameter_rows,
        "parameter_values_hash": parameter_values_hash,
        "optimizer": optimizer,
        "pooled_parameter_vector_hash": pooled_parameter_vector_hash,
        "input_hashes": input_hashes,
    }
    write_json(parameters_path, parameter_artifact)
    bkt_parameter_artifact_hash = sha256_file(parameters_path)

    backoff_nodes_hash = hashlib.sha256(
        canonical_json_bytes(zero_observation_nodes)
    ).hexdigest()
    coverage = {
        "required_node_count": len(required_nodes),
        "required_nodes": required_nodes,
        "concept_specific_nodes": observed_required_nodes,
        "pooled_backoff_nodes": zero_observation_nodes,
        "pooled_backoff_nodes_hash": backoff_nodes_hash,
        "missing_nodes": [],
        "coverage_fraction": 1.0,
        "backoff_rule": "pooled_zero_observation_bkt",
        "parameter_values_hash": parameter_values_hash,
        "bkt_parameter_artifact_hash": bkt_parameter_artifact_hash,
        "pooled_parameter_vector_hash": pooled_parameter_vector_hash,
        "pooled_parameter_artifact_hash": pooled_parameter_artifact_hash,
    }
    write_json(coverage_path, coverage)

    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    metadata = {
        "teacher_parameterization": PARAMETERIZATION,
        "training_split": "train",
        "train_students": split["train"],
        "train_student_count": len(split["train"]),
        "validation_students_used": 0,
        "test_students_used": 0,
        "train_observation_count": len(train),
        "manifest_hash": manifest_hash(manifest),
        "required_nodes": required_nodes,
        "concept_specific_nodes": observed_required_nodes,
        "pooled_backoff_nodes": zero_observation_nodes,
        "parameter_values_hash": parameter_values_hash,
        "pooled_parameter_vector_hash": pooled_parameter_vector_hash,
        "optimizer": optimizer,
        "input_hashes": input_hashes,
        "artifacts": {
            "bkt_parameters": {
                "path": protocol_path(parameters_path),
                "sha256": bkt_parameter_artifact_hash,
                "artifact_hash": bkt_parameter_artifact_hash,
                "parameter_values_hash": parameter_values_hash,
            },
            "pooled_bkt_parameters": {
                "path": protocol_path(pooled_path),
                "sha256": pooled_parameter_artifact_hash,
                "artifact_hash": pooled_parameter_artifact_hash,
                "parameter_vector_hash": pooled_parameter_vector_hash,
            },
            "bkt_coverage": {
                "path": protocol_path(coverage_path),
                "sha256": sha256_file(coverage_path),
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "generation_command": "python experiments/train_bkt_teacher.py",
    }
    write_json(metadata_path, metadata)
    return {
        "bkt_parameters": sha256_file(parameters_path),
        "pooled_bkt_parameters": sha256_file(pooled_path),
        "bkt_coverage": sha256_file(coverage_path),
        "bkt_teacher_metadata": sha256_file(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    hashes = train_bkt_teacher(output_dir=args.output_dir)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
