"""Train the canonical CPU deterministic DKT teacher for Task 13."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow
import torch


ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PATH = ROOT / "data" / "kt_set" / "concept_sessions.parquet"
SPLIT_PATH = ROOT / "data" / "kt_set" / "student_split.json"
PREPROCESSING_PATH = ROOT / "data" / "kt_set" / "preprocessing_manifest.json"
DAG_PATH = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
OUTPUT = ROOT / "artifacts" / "dkt_set"
CONFIG_PATH = OUTPUT / "dkt_config.json"
CHECKPOINT_PATH = OUTPUT / "dkt_checkpoint.pt"
METRICS_PATH = OUTPUT / "dkt_training_metrics.json"
INPUT_METADATA_PATH = OUTPUT / "dkt_input_metadata.json"
TEACHER_METADATA_PATH = OUTPUT / "dkt_teacher_metadata.json"
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_dag, load_manifest, manifest_hash
from experiments.kt.artifacts import (
    canonical_json_bytes,
    protocol_path,
    sha256_file,
    write_json,
)
from experiments.train_bkt_teacher import required_nodes_from_manifest
from src.oracle_core.dkt_teacher import (
    DKTSequence,
    DKTTeacherModel,
    binary_auc,
    masked_next_target_bce,
    pad_sequences,
)
from src.oracle_core.set_oracle_surrogate import save_deterministic_checkpoint


SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 200
PATIENCE = 20
MINIMUM_DELTA = 1e-6
EMBEDDING_DIM = 64
HIDDEN_DIM = 128


def configure_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def build_sequences(
    sessions: pd.DataFrame,
    *,
    split_name: str,
    node_to_index: dict[int, int],
) -> list[DKTSequence]:
    selected = sessions[sessions["split"] == split_name]
    sequences = []
    for student_id, rows in selected.groupby("student_id", sort=True):
        rows = rows.sort_values("session_index", kind="mergesort")
        indices = tuple(int(node_to_index[int(node)]) for node in rows["target_node"])
        outcomes = tuple(int(value) for value in rows["correct"])
        tokens = tuple(2 * index + outcome for index, outcome in zip(indices, outcomes))
        sequences.append(
            DKTSequence(str(student_id), tokens, indices, outcomes)
        )
    return sequences


def deterministic_batches(
    sequences: list[DKTSequence], batch_size: int = BATCH_SIZE
) -> Iterable[list[DKTSequence]]:
    for start in range(0, len(sequences), batch_size):
        yield sequences[start : start + batch_size]


def evaluate(
    model: DKTTeacherModel,
    sequences: list[DKTSequence],
) -> dict[str, float | int]:
    total_loss = 0.0
    total_count = 0
    labels: list[int] = []
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in deterministic_batches(sequences):
            tokens, targets, outcomes, mask = pad_sequences(batch)
            table = model.prefix_probabilities(tokens)
            observed = table.gather(2, targets.unsqueeze(2)).squeeze(2)
            losses = torch.nn.functional.binary_cross_entropy(
                observed, outcomes, reduction="none"
            )
            total_loss += float(losses[mask].sum())
            total_count += int(mask.sum())
            labels.extend(int(value) for value in outcomes[mask].tolist())
            probabilities.extend(float(value) for value in observed[mask].tolist())
    predictions = [value >= 0.5 for value in probabilities]
    return {
        "events": total_count,
        "bce": total_loss / total_count,
        "auc": binary_auc(labels, probabilities),
        "accuracy": sum(prediction == bool(label) for prediction, label in zip(predictions, labels))
        / total_count,
        "probability_min": min(probabilities),
        "probability_max": max(probabilities),
    }


def _input_metadata(
    sessions: pd.DataFrame,
    split: dict,
    node_order: list[int],
) -> dict:
    statistics_by_split = {}
    for name in ("train", "validation", "test"):
        rows = sessions[sessions["split"] == name]
        lengths = [len(group) for _, group in rows.groupby("student_id", sort=True)]
        statistics_by_split[name] = {
            "students": int(rows["student_id"].nunique()),
            "sessions": int(len(rows)),
            "sequences": len(lengths),
            "sequence_length_min": min(lengths),
            "sequence_length_median": float(statistics.median(lengths)),
            "sequence_length_max": max(lengths),
        }
        if set(rows["student_id"].astype(str)) != set(split[name]):
            raise ValueError(f"Canonical sessions/{name} student split mismatch")
    return {
        "schema_version": 1,
        "event_unit": "canonical concept session",
        "event_fields": ["student_id", "session_index", "target_node", "correct"],
        "correct_rule": "1 iff session_score >= 0.8",
        "interaction_token": "2 * model_node_index + correct",
        "node_order": node_order,
        "node_vocabulary_size": len(node_order),
        "interaction_vocabulary_size": 2 * len(node_order),
        "sequence_order": "student_id lexical; session_index ascending",
        "current_label_excluded": True,
        "empty_prefix_prediction": True,
        "training_uses_test_students": False,
        "statistics": statistics_by_split,
        "sources": {
            "concept_sessions": {
                "path": protocol_path(SESSIONS_PATH),
                "sha256": sha256_file(SESSIONS_PATH),
            },
            "student_split": {
                "path": protocol_path(SPLIT_PATH),
                "sha256": sha256_file(SPLIT_PATH),
            },
            "preprocessing_manifest": {
                "path": protocol_path(PREPROCESSING_PATH),
                "sha256": sha256_file(PREPROCESSING_PATH),
            },
            "dag": {"path": protocol_path(DAG_PATH), "sha256": sha256_file(DAG_PATH)},
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": torch.__version__,
        },
        "generation_command": "python experiments/train_dkt_teacher.py",
    }


def train_dkt_teacher() -> dict[str, str]:
    configure_determinism()
    sessions = pd.read_parquet(SESSIONS_PATH)
    with SPLIT_PATH.open("r", encoding="utf-8") as file:
        split = json.load(file)
    node_order, _ = load_dag(DAG_PATH)
    node_to_index = {node: index for index, node in enumerate(node_order)}
    manifest = load_manifest()
    required_nodes = required_nodes_from_manifest(manifest)
    if set(sessions["target_node"].astype(int)) - set(node_order):
        raise ValueError("DKT input contains an unknown DAG node")
    if set(split["test"]) & set(split["train"] + split["validation"]):
        raise ValueError("Test students overlap train/validation")

    train_sequences = build_sequences(
        sessions, split_name="train", node_to_index=node_to_index
    )
    validation_sequences = build_sequences(
        sessions, split_name="validation", node_to_index=node_to_index
    )
    input_metadata = _input_metadata(sessions, split, node_order)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(INPUT_METADATA_PATH, input_metadata)

    config = {
        "schema_version": 1,
        "condition_name": "DKT teacher for DKT-derived Set Oracle",
        "node_order": node_order,
        "required_nodes": required_nodes,
        "architecture": {
            "interaction_vocabulary_size": 2 * len(node_order),
            "num_nodes": len(node_order),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "lstm_layers": 1,
            "dropout": 0.0,
            "output_dim": len(node_order),
            "initial_prediction": "learned 61-logit vector",
        },
        "training": {
            "seed": SEED,
            "device": "cpu",
            "deterministic_algorithms": True,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "batch_order": "student_id lexical; no shuffle",
            "padding": "right padded; masked from loss",
            "loss": "masked observed-next-target binary cross entropy",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "minimum_delta": MINIMUM_DELTA,
            "selection": "earliest epoch improving validation BCE by minimum_delta",
        },
        "split_hash": sha256_file(SPLIT_PATH),
        "preprocessing_manifest_hash": sha256_file(PREPROCESSING_PATH),
        "concept_sessions_hash": sha256_file(SESSIONS_PATH),
        "input_metadata_hash": sha256_file(INPUT_METADATA_PATH),
        "manifest_hash": manifest_hash(manifest),
        "test_students_used": 0,
    }
    write_json(CONFIG_PATH, config)
    config_hash = sha256_file(CONFIG_PATH)

    model = DKTTeacherModel(
        num_nodes=len(node_order),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    selected_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in deterministic_batches(train_sequences):
            tokens, targets, outcomes, mask = pad_sequences(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_next_target_bce(model, tokens, targets, outcomes, mask)
            loss.backward()
            optimizer.step()
        train_result = evaluate(model, train_sequences)
        validation_result = evaluate(model, validation_sequences)
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_result["bce"],
                "validation_bce": validation_result["bce"],
            }
        )
        if float(validation_result["bce"]) < best_validation - MINIMUM_DELTA:
            best_validation = float(validation_result["bce"])
            selected_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_sequences)
    validation_metrics = evaluate(model, validation_sequences)
    tensor_hash = hashlib.sha256(
        b"".join(
            model.state_dict()[name].detach().cpu().contiguous().numpy().tobytes()
            for name in sorted(model.state_dict())
        )
    ).hexdigest()
    save_deterministic_checkpoint(
        CHECKPOINT_PATH,
        state_dict=model.state_dict(),
        metadata={
            "condition_name": "DKT teacher",
            "config_hash": config_hash,
            "node_order": node_order,
            "selected_epoch": selected_epoch,
            "tensor_hash": tensor_hash,
        },
    )
    checkpoint_hash = sha256_file(CHECKPOINT_PATH)
    metrics = {
        "schema_version": 1,
        "selected_epoch": selected_epoch,
        "epochs_run": len(history),
        "best_validation_bce": best_validation,
        "train": train_metrics,
        "validation": validation_metrics,
        "training_history": history,
        "teacher_checkpoint_tensor_hash": tensor_hash,
        "teacher_checkpoint_artifact_hash": checkpoint_hash,
        "config_hash": config_hash,
        "test_students_used": 0,
    }
    write_json(METRICS_PATH, metrics)
    teacher_metadata = {
        "schema_version": 1,
        "teacher": "DKT",
        "condition_name": "DKT-derived Set Oracle teacher",
        "input_is_canonical_concept_session": True,
        "current_outcome_excluded": True,
        "hidden_state_scope": "independent per student sequence",
        "query_targets": "all 27 planning-required nodes at every prefix",
        "train_students": len(train_sequences),
        "validation_students": len(validation_sequences),
        "test_students_used": 0,
        "required_nodes": required_nodes,
        "config_hash": config_hash,
        "checkpoint_tensor_hash": tensor_hash,
        "checkpoint_artifact_hash": checkpoint_hash,
        "training_metrics_hash": sha256_file(METRICS_PATH),
        "input_metadata_hash": sha256_file(INPUT_METADATA_PATH),
        "generation_command": "python experiments/train_dkt_teacher.py",
    }
    write_json(TEACHER_METADATA_PATH, teacher_metadata)
    return {
        "dkt_input_metadata": sha256_file(INPUT_METADATA_PATH),
        "dkt_config": config_hash,
        "dkt_checkpoint": checkpoint_hash,
        "dkt_training_metrics": sha256_file(METRICS_PATH),
        "dkt_teacher_metadata": sha256_file(TEACHER_METADATA_PATH),
    }


def main() -> None:
    print(json.dumps(train_dkt_teacher(), indent=2, sort_keys=True))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
