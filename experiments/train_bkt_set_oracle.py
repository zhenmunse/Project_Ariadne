"""Train and validate the deterministic BKT-derived set-oracle surrogate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import pearsonr, spearmanr


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
from src.oracle_core.set_oracle_surrogate import (
    SetOracleSurrogate,
    save_deterministic_checkpoint,
)


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "bkt_set"
DEFAULT_RESULTS_DIR = ROOT / "results" / "bkt_set"
SEED = 42
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
MINIMUM_DELTA = 1e-8
PATIENCE = 30
BATCH_SIZE = "full"
PREDICTION_BATCH_SIZE = 8192


def _mask_matrix(values: pd.Series, num_nodes: int) -> torch.Tensor:
    rows = []
    for value in values:
        if not isinstance(value, str) or len(value) != num_nodes or set(value) - {"0", "1"}:
            raise ValueError("mastery_mask must be a fixed-width binary string")
        rows.append([float(bit) for bit in value])
    return torch.tensor(rows, dtype=torch.float32)


def _target_indices(
    values: pd.Series, node_id_to_index: dict[int, int]
) -> torch.Tensor:
    try:
        indices = [node_id_to_index[int(value)] for value in values]
    except KeyError as error:
        raise ValueError(f"Tuple table contains unknown target: {error.args[0]}") from error
    return torch.tensor(indices, dtype=torch.long)


def _table_tensors(
    table: pd.DataFrame,
    node_order: list[int],
    *,
    target_column: str = "target_node",
    probability_column: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_id_to_index = {node: index for index, node in enumerate(node_order)}
    return (
        _mask_matrix(table["mastery_mask"], len(node_order)),
        _target_indices(table[target_column], node_id_to_index),
        torch.tensor(table[probability_column].to_numpy(), dtype=torch.float32),
    )


def _predict(
    model: SetOracleSurrogate,
    masks: torch.Tensor,
    targets: torch.Tensor,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(masks), PREDICTION_BATCH_SIZE):
            batches.append(
                model(
                    masks[start : start + PREDICTION_BATCH_SIZE],
                    targets[start : start + PREDICTION_BATCH_SIZE],
                ).cpu()
            )
    return torch.cat(batches).numpy().astype(np.float64)


def _weighted_mse(
    predictions: np.ndarray, labels: np.ndarray, counts: np.ndarray
) -> float:
    return float(np.sum(counts * np.square(predictions - labels)) / np.sum(counts))


def _correlation(function: Any, labels: np.ndarray, predictions: np.ndarray) -> float:
    if np.std(labels) == 0.0 or np.std(predictions) == 0.0:
        return 0.0
    result = function(labels, predictions)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def _split_metrics(
    model: SetOracleSurrogate,
    grouped: pd.DataFrame,
    raw: pd.DataFrame,
    node_order: list[int],
    split_stats: dict[str, Any],
) -> dict[str, Any]:
    grouped_masks, grouped_targets, grouped_labels = _table_tensors(
        grouped,
        node_order,
        probability_column="teacher_probability_mean",
    )
    grouped_predictions = _predict(model, grouped_masks, grouped_targets)
    grouped_truth = grouped_labels.numpy().astype(np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)

    raw_masks, raw_targets, raw_labels = _table_tensors(
        raw,
        node_order,
        probability_column="teacher_probability",
    )
    raw_predictions = _predict(model, raw_masks, raw_targets)
    raw_truth = raw_labels.numpy().astype(np.float64)
    errors = raw_predictions - raw_truth
    return {
        "students": split_stats["students"],
        "prefixes": split_stats["prefixes"],
        "unique_mastery_states": int(grouped["mastery_state"].nunique()),
        "unique_state_target_pairs": len(grouped),
        "grouped_weighted_mse": _weighted_mse(
            grouped_predictions, grouped_truth, counts
        ),
        "prefix_level_mse": float(np.mean(np.square(errors))),
        "prefix_level_mae": float(np.mean(np.abs(errors))),
        "maximum_absolute_error": float(np.max(np.abs(errors))),
        "prediction_min": float(np.min(raw_predictions)),
        "prediction_max": float(np.max(raw_predictions)),
        "teacher_min": float(np.min(raw_truth)),
        "teacher_max": float(np.max(raw_truth)),
        "pearson_correlation": _correlation(pearsonr, raw_truth, raw_predictions),
        "spearman_correlation": _correlation(spearmanr, raw_truth, raw_predictions),
        "coverage": sorted(int(node) for node in grouped["target_node"].unique()),
        "per_target_raw_count": {
            str(int(node)): int(count)
            for node, count in raw.groupby("target_node").size().items()
        },
        "per_target_grouped_count": {
            str(int(node)): int(count)
            for node, count in grouped.groupby("target_node").size().items()
        },
    }


def _state_dependence(
    model: SetOracleSurrogate,
    validation_grouped: pd.DataFrame,
    node_order: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    masks, targets, _ = _table_tensors(
        validation_grouped,
        node_order,
        probability_column="teacher_probability_mean",
    )
    predictions = _predict(model, masks, targets)
    table = validation_grouped.copy()
    table["surrogate_probability"] = predictions
    table["target_already_mastered"] = [
        int(target) in set(json.loads(state))
        for target, state in zip(table["target_node"], table["mastery_state"])
    ]
    rows = []
    for target, group in table.groupby("target_node", sort=True):
        valid_group = group[~group["target_already_mastered"]]
        values = valid_group["surrogate_probability"].to_numpy(dtype=np.float64)
        if len(values) == 0:
            max_effect = 0.0
            packed_different = False
        else:
            max_effect = float(np.max(values) - np.min(values))
            packed_different = len(
                {struct.pack("!d", float(value)) for value in values}
            ) > 1
        rows.append(
            {
                "target_node": int(target),
                "validation_state_count": len(valid_group),
                "excluded_mastered_state_count": int(
                    group["target_already_mastered"].sum()
                ),
                "max_state_effect": max_effect,
                "packed_outputs_differ": packed_different,
                "effect_at_least_0_01": bool(max_effect >= 0.01),
                "effect_at_least_0_05": bool(max_effect >= 0.05),
            }
        )
    effects = pd.DataFrame(rows).sort_values("target_node").reset_index(drop=True)
    multiple = effects[effects["validation_state_count"] >= 2]
    single = effects[effects["validation_state_count"] < 2]
    maximum = float(multiple["max_state_effect"].max()) if not multiple.empty else 0.0
    packed_different = bool(multiple["packed_outputs_differ"].any())
    summary = {
        "packed_outputs_differ": packed_different,
        "max_state_effect": maximum,
        "median_per_target_max_effect": float(multiple["max_state_effect"].median())
        if not multiple.empty
        else 0.0,
        "p95_per_target_max_effect": float(
            np.percentile(multiple["max_state_effect"], 95)
        )
        if not multiple.empty
        else 0.0,
        "targets_with_multiple_states": int(len(multiple)),
        "targets_with_single_state": int(len(single)),
        "targets_with_effect_at_least_0_01": int(
            multiple["effect_at_least_0_01"].sum()
        ),
        "targets_with_effect_at_least_0_05": int(
            multiple["effect_at_least_0_05"].sum()
        ),
        "minimum_required_effect": 1e-6,
        "go": bool(packed_different and maximum >= 1e-6),
    }
    return summary, effects


def train_bkt_set_oracle(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[str, str]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    manifest = load_manifest()
    node_order, _ = load_dag(ROOT / "data" / "ecs32a_dag_required_full_v1.json")
    train_grouped_path = artifact_dir / "train_grouped_tuples.parquet"
    validation_grouped_path = artifact_dir / "validation_grouped_tuples.parquet"
    train_raw_path = artifact_dir / "train_prefix_examples.parquet"
    validation_raw_path = artifact_dir / "validation_prefix_examples.parquet"
    distillation_metadata_path = artifact_dir / "distillation_metadata.json"
    teacher_metadata_path = artifact_dir / "bkt_teacher_metadata.json"
    parameter_path = artifact_dir / "bkt_parameters.json"
    pooled_path = artifact_dir / "pooled_bkt_parameters.json"
    coverage_path = artifact_dir / "bkt_coverage.json"

    with distillation_metadata_path.open(encoding="utf-8") as file:
        distillation_metadata = json.load(file)
    with teacher_metadata_path.open(encoding="utf-8") as file:
        teacher_metadata = json.load(file)
    with coverage_path.open(encoding="utf-8") as file:
        coverage = json.load(file)
    supported_targets = coverage["required_nodes"]
    if coverage["coverage_fraction"] != 1.0 or coverage["missing_nodes"]:
        raise ValueError("BKT teacher coverage gate failed")

    source_paths = {
        "train_grouped_tuples": train_grouped_path,
        "validation_grouped_tuples": validation_grouped_path,
        "train_prefix_examples": train_raw_path,
        "validation_prefix_examples": validation_raw_path,
        "distillation_metadata": distillation_metadata_path,
        "bkt_teacher_metadata": teacher_metadata_path,
        "bkt_parameters": parameter_path,
        "pooled_bkt_parameters": pooled_path,
        "bkt_coverage": coverage_path,
        "adapter_spec": ROOT / "documents" / "kt_set_adapter_spec.md",
        "dag": ROOT / "data" / "ecs32a_dag_required_full_v1.json",
        "evaluator": ROOT / "experiments" / "common" / "evaluator.py",
        "student_split": ROOT / "data" / "kt_set" / "student_split.json",
        "preprocessing_manifest": ROOT
        / "data"
        / "kt_set"
        / "preprocessing_manifest.json",
        "cleaned_interactions": ROOT
        / "data"
        / "processed"
        / "cleaned_interactions.csv",
        "question_concept_mapping": ROOT
        / "data"
        / "question_concept_mapping_final.csv",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    config = {
        "schema_version": 1,
        "condition_name": "BKT-derived Set Oracle",
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
        "parameter_values_hash": teacher_metadata["parameter_values_hash"],
        "pooled_parameter_vector_hash": teacher_metadata[
            "pooled_parameter_vector_hash"
        ],
        "bkt_parameter_artifact_hash": source_hashes["bkt_parameters"],
        "pooled_parameter_artifact_hash": source_hashes["pooled_bkt_parameters"],
        "pooled_backoff_nodes_hash": coverage["pooled_backoff_nodes_hash"],
        "tuple_collection_hash": distillation_metadata["tuple_collection_hash"],
        "split_hash": source_hashes["student_split"],
        "compression_config_hash": hashlib.sha256(
            canonical_json_bytes(distillation_metadata["mastery"])
        ).hexdigest(),
        "evaluator_hash": source_hashes["evaluator"],
        "source_hashes": source_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "generation_command": "python experiments/train_bkt_set_oracle.py",
    }
    config_path = artifact_dir / "surrogate_config.json"
    write_json(config_path, config)
    config_hash = sha256_file(config_path)

    train_grouped = pd.read_parquet(train_grouped_path)
    validation_grouped = pd.read_parquet(validation_grouped_path)
    train_masks, train_targets, train_labels = _table_tensors(
        train_grouped,
        node_order,
        probability_column="teacher_probability_mean",
    )
    validation_masks, validation_targets, validation_labels = _table_tensors(
        validation_grouped,
        node_order,
        probability_column="teacher_probability_mean",
    )
    train_weights = torch.tensor(train_grouped["count"].to_numpy(), dtype=torch.float32)
    validation_weights = torch.tensor(
        validation_grouped["count"].to_numpy(), dtype=torch.float32
    )

    model = SetOracleSurrogate(num_nodes=len(node_order)).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_masks, train_targets)
        train_loss = torch.sum(train_weights * torch.square(predictions - train_labels)) / torch.sum(
            train_weights
        )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_predictions = model(validation_masks, validation_targets)
            validation_loss = torch.sum(
                validation_weights
                * torch.square(validation_predictions - validation_labels)
            ) / torch.sum(validation_weights)
        train_value = float(train_loss.item())
        validation_value = float(validation_loss.item())
        training_history.append(
            {
                "epoch": epoch,
                "train_grouped_weighted_mse": train_value,
                "validation_grouped_weighted_mse": validation_value,
            }
        )
        if validation_value < best_loss - MINIMUM_DELTA:
            best_loss = validation_value
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Surrogate training did not select a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval().requires_grad_(False)

    checkpoint_metadata = {
        "config_hash": config_hash,
        "node_order": node_order,
        "supported_targets": supported_targets,
        "selected_epoch": best_epoch,
        "parameter_values_hash": config["parameter_values_hash"],
        "pooled_parameter_vector_hash": config["pooled_parameter_vector_hash"],
    }
    checkpoint_path = artifact_dir / "surrogate_checkpoint.pt"
    save_deterministic_checkpoint(
        checkpoint_path,
        state_dict=best_state,
        metadata=checkpoint_metadata,
    )
    checkpoint_hash = sha256_file(checkpoint_path)

    train_raw = pd.read_parquet(train_raw_path)
    validation_raw = pd.read_parquet(validation_raw_path)
    metrics = {
        "condition_name": "BKT-derived Set Oracle",
        "go": False,
        "selected_epoch": best_epoch,
        "epochs_run": len(training_history),
        "best_validation_grouped_mse": best_loss,
        "train": _split_metrics(
            model,
            train_grouped,
            train_raw,
            node_order,
            distillation_metadata["statistics"]["train"],
        ),
        "validation": _split_metrics(
            model,
            validation_grouped,
            validation_raw,
            node_order,
            distillation_metadata["statistics"]["validation"],
        ),
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "parameter_values_hash": config["parameter_values_hash"],
        "pooled_parameter_vector_hash": config["pooled_parameter_vector_hash"],
        "bkt_parameter_artifact_hash": config["bkt_parameter_artifact_hash"],
        "pooled_parameter_artifact_hash": config["pooled_parameter_artifact_hash"],
        "pooled_backoff_nodes_hash": config["pooled_backoff_nodes_hash"],
        "tuple_collection_hash": config["tuple_collection_hash"],
        "split_hash": config["split_hash"],
        "compression_config_hash": config["compression_config_hash"],
        "evaluator_hash": config["evaluator_hash"],
        "source_hashes": source_hashes,
        "training_history": training_history,
    }
    state_dependence, state_effects = _state_dependence(
        model, validation_grouped, node_order
    )
    metrics["state_dependence"] = state_dependence
    metrics["go"] = state_dependence["go"]
    if not metrics["go"]:
        raise RuntimeError(
            "BKT-derived Set Oracle failed the state-dependence go/no-go gate"
        )

    metrics_path = artifact_dir / "surrogate_metrics.json"
    write_json(metrics_path, metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_provenance = {
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "parameter_values_hash": config["parameter_values_hash"],
        "pooled_parameter_vector_hash": config["pooled_parameter_vector_hash"],
        "bkt_parameter_artifact_hash": config["bkt_parameter_artifact_hash"],
        "pooled_parameter_artifact_hash": config[
            "pooled_parameter_artifact_hash"
        ],
        "tuple_collection_hash": config["tuple_collection_hash"],
        "split_hash": config["split_hash"],
        "compression_config_hash": config["compression_config_hash"],
        "evaluator_hash": config["evaluator_hash"],
    }
    for name, value in result_provenance.items():
        state_effects[name] = value
    state_path = results_dir / "state_dependence.csv"
    state_effects.to_csv(state_path, index=False, lineterminator="\n")
    oracle_metrics_path = results_dir / "oracle_metrics.csv"
    pd.DataFrame(
        [
            {
                "split": split_name,
                **metrics[split_name],
                **result_provenance,
            }
            for split_name in ("train", "validation")
        ]
    ).to_csv(oracle_metrics_path, index=False, lineterminator="\n")
    return {
        "surrogate_config": config_hash,
        "surrogate_checkpoint": checkpoint_hash,
        "surrogate_metrics": sha256_file(metrics_path),
        "state_dependence": sha256_file(state_path),
        "oracle_metrics": sha256_file(oracle_metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    hashes = train_bkt_set_oracle(
        artifact_dir=args.artifact_dir, results_dir=args.results_dir
    )
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
