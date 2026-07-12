"""Train the deterministic DKT-derived set-oracle surrogate."""

from __future__ import annotations

import copy
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dkt_set"
RESULTS = ROOT / "results" / "dkt_set"
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_dag, load_manifest, manifest_hash
from experiments.kt.artifacts import protocol_path, sha256_file, write_json
from experiments.train_bkt_set_oracle import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MINIMUM_DELTA,
    PATIENCE,
    SEED,
    _split_metrics,
    _state_dependence,
    _table_tensors,
)
from src.oracle_core.set_oracle_surrogate import (
    SetOracleSurrogate,
    save_deterministic_checkpoint,
)


def train_dkt_set_oracle() -> dict[str, str]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifest = load_manifest()
    node_order, _ = load_dag(ROOT / "data" / "ecs32a_dag_required_full_v1.json")

    paths = {
        "train_grouped_tuples": ARTIFACTS / "train_grouped_tuples.parquet",
        "validation_grouped_tuples": ARTIFACTS / "validation_grouped_tuples.parquet",
        "train_prefix_examples": ARTIFACTS / "train_prefix_examples.parquet",
        "validation_prefix_examples": ARTIFACTS / "validation_prefix_examples.parquet",
        "distillation_metadata": ARTIFACTS / "distillation_metadata.json",
        "dkt_config": ARTIFACTS / "dkt_config.json",
        "dkt_checkpoint": ARTIFACTS / "dkt_checkpoint.pt",
        "dkt_training_metrics": ARTIFACTS / "dkt_training_metrics.json",
        "dkt_input_metadata": ARTIFACTS / "dkt_input_metadata.json",
        "dkt_teacher_metadata": ARTIFACTS / "dkt_teacher_metadata.json",
        "adapter_spec": ROOT / "documents" / "kt_set_adapter_spec.md",
        "dag": ROOT / "data" / "ecs32a_dag_required_full_v1.json",
        "evaluator": ROOT / "experiments" / "common" / "evaluator.py",
        "student_split": ROOT / "data" / "kt_set" / "student_split.json",
        "preprocessing_manifest": ROOT / "data" / "kt_set" / "preprocessing_manifest.json",
    }
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    with paths["distillation_metadata"].open(encoding="utf-8") as file:
        distillation = json.load(file)
    with paths["dkt_training_metrics"].open(encoding="utf-8") as file:
        teacher_metrics = json.load(file)
    with paths["dkt_teacher_metadata"].open(encoding="utf-8") as file:
        teacher_metadata = json.load(file)
    supported_targets = distillation["required_nodes"]
    if set(supported_targets) != set(manifest["targets"]) | {
        node for closure in manifest["closures"] for node in closure["sequence_nodes"]
    }:
        raise ValueError("DKT distillation required-target coverage mismatch")
    if teacher_metadata["checkpoint_artifact_hash"] != source_hashes["dkt_checkpoint"]:
        raise ValueError("DKT teacher checkpoint provenance mismatch")

    config = {
        "schema_version": 1,
        "condition_name": "DKT-derived Set Oracle",
        "architecture": {
            "num_nodes": len(node_order),
            "input_dim": 2 * len(node_order),
            "hidden_dims": [128, 64],
            "output": "Linear(64,1)+Sigmoid",
            "dropout": None,
        },
        "node_order": node_order,
        "supported_targets": supported_targets,
        "base_cost": manifest["base_cost"],
        "device": "cpu",
        "seed": SEED,
        "deterministic_algorithms": True,
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "shuffle": False,
            "batch_size": BATCH_SIZE,
            "loss": "count-weighted grouped MSE",
            "max_epochs": MAX_EPOCHS,
            "minimum_delta": MINIMUM_DELTA,
            "patience": PATIENCE,
            "selection": "earliest epoch improving validation grouped MSE by minimum_delta",
        },
        "manifest_hash": manifest_hash(manifest),
        "teacher_tensor_hash": teacher_metrics["teacher_checkpoint_tensor_hash"],
        "teacher_checkpoint_hash": source_hashes["dkt_checkpoint"],
        "teacher_config_hash": source_hashes["dkt_config"],
        "teacher_metrics_hash": source_hashes["dkt_training_metrics"],
        "tuple_collection_hash": distillation["tuple_collection_hash"],
        "split_hash": source_hashes["student_split"],
        "compression_config_hash": distillation["mastery"]["compression_config_hash"],
        "zero_observation_nodes_hash": distillation["mastery"]["zero_observation_nodes_hash"],
        "training_observed_nodes_hash": distillation["mastery"]["training_observed_nodes_hash"],
        "evaluator_hash": source_hashes["evaluator"],
        "source_hashes": source_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "generation_command": "python experiments/train_dkt_set_oracle.py",
    }
    config_path = ARTIFACTS / "surrogate_config.json"
    write_json(config_path, config)
    config_hash = sha256_file(config_path)

    train_grouped = pd.read_parquet(paths["train_grouped_tuples"])
    validation_grouped = pd.read_parquet(paths["validation_grouped_tuples"])
    train_masks, train_targets, train_labels = _table_tensors(
        train_grouped, node_order, probability_column="teacher_probability_mean"
    )
    validation_masks, validation_targets, validation_labels = _table_tensors(
        validation_grouped, node_order, probability_column="teacher_probability_mean"
    )
    train_weights = torch.tensor(train_grouped["count"].to_numpy(), dtype=torch.float32)
    validation_weights = torch.tensor(
        validation_grouped["count"].to_numpy(), dtype=torch.float32
    )
    model = SetOracleSurrogate(num_nodes=len(node_order)).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_masks, train_targets)
        train_loss = torch.sum(train_weights * (predictions - train_labels) ** 2) / torch.sum(train_weights)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            valid_predictions = model(validation_masks, validation_targets)
            validation_loss = torch.sum(
                validation_weights * (valid_predictions - validation_labels) ** 2
            ) / torch.sum(validation_weights)
        train_value = float(train_loss.detach())
        validation_value = float(validation_loss)
        history.append({
            "epoch": epoch,
            "train_grouped_weighted_mse": train_value,
            "validation_grouped_weighted_mse": validation_value,
        })
        if validation_value < best_loss - MINIMUM_DELTA:
            best_loss = validation_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("DKT set surrogate did not select a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval().requires_grad_(False)
    checkpoint_path = ARTIFACTS / "surrogate_checkpoint.pt"
    save_deterministic_checkpoint(
        checkpoint_path,
        state_dict=best_state,
        metadata={
            "config_hash": config_hash,
            "node_order": node_order,
            "supported_targets": supported_targets,
            "selected_epoch": best_epoch,
            "teacher_tensor_hash": config["teacher_tensor_hash"],
            "teacher_checkpoint_hash": config["teacher_checkpoint_hash"],
            "tuple_collection_hash": config["tuple_collection_hash"],
        },
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    train_raw = pd.read_parquet(paths["train_prefix_examples"])
    validation_raw = pd.read_parquet(paths["validation_prefix_examples"])
    metrics = {
        "condition_name": "DKT-derived Set Oracle",
        "go": False,
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_grouped_mse": best_loss,
        "train": _split_metrics(model, train_grouped, train_raw, node_order, distillation["statistics"]["train"]),
        "validation": _split_metrics(model, validation_grouped, validation_raw, node_order, distillation["statistics"]["validation"]),
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "teacher_tensor_hash": config["teacher_tensor_hash"],
        "teacher_checkpoint_hash": config["teacher_checkpoint_hash"],
        "tuple_collection_hash": config["tuple_collection_hash"],
        "split_hash": config["split_hash"],
        "compression_config_hash": config["compression_config_hash"],
        "evaluator_hash": config["evaluator_hash"],
        "source_hashes": source_hashes,
        "training_history": history,
    }
    state_summary, state_effects = _state_dependence(
        model, validation_grouped, node_order
    )
    metrics["state_dependence"] = state_summary
    metrics["go"] = state_summary["go"]
    if not metrics["go"]:
        raise RuntimeError("DKT-derived Set Oracle failed state-dependence gate")
    metrics_path = ARTIFACTS / "surrogate_metrics.json"
    write_json(metrics_path, metrics)
    RESULTS.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "teacher_tensor_hash": config["teacher_tensor_hash"],
        "teacher_checkpoint_hash": config["teacher_checkpoint_hash"],
        "tuple_collection_hash": config["tuple_collection_hash"],
        "split_hash": config["split_hash"],
        "compression_config_hash": config["compression_config_hash"],
        "evaluator_hash": config["evaluator_hash"],
    }
    for name, value in provenance.items():
        state_effects[name] = value
    state_path = RESULTS / "state_dependence.csv"
    state_effects.to_csv(state_path, index=False, lineterminator="\n")
    oracle_path = RESULTS / "oracle_metrics.csv"
    pd.DataFrame([
        {"split": name, **metrics[name], **provenance}
        for name in ("train", "validation")
    ]).to_csv(oracle_path, index=False, lineterminator="\n")
    return {
        "surrogate_config": config_hash,
        "surrogate_checkpoint": checkpoint_hash,
        "surrogate_metrics": sha256_file(metrics_path),
        "state_dependence": sha256_file(state_path),
        "oracle_metrics": sha256_file(oracle_path),
    }


def main() -> None:
    print(json.dumps(train_dkt_set_oracle(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
