"""Planner-facing deterministic DKT-derived Set Oracle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch

from experiments.common.manifest import load_dag, load_manifest
from experiments.kt.artifacts import sha256_file
from experiments.kt.mastery import ancestor_map
from src.oracle_core.set_oracle_surrogate import (
    SetOracleSurrogate,
    load_deterministic_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "artifacts" / "dkt_set" / "surrogate_config.json"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "dkt_set" / "surrogate_checkpoint.pt"
DEFAULT_DAG = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
MIN_PROBABILITY = 1e-12


class DKTSetOracle:
    """Expose the frozen DKT surrogate as a pure function of ``(v, s)``."""

    def __init__(
        self,
        model: SetOracleSurrogate,
        *,
        node_order: list[int],
        supported_targets: Iterable[int],
        ancestors: dict[int, frozenset[int]],
        base_cost: float,
        config_hash: str,
        checkpoint_hash: str,
        teacher_tensor_hash: str,
        teacher_checkpoint_hash: str,
        tuple_collection_hash: str,
    ) -> None:
        self.model = model.cpu().eval().requires_grad_(False)
        self.node_order = list(node_order)
        self.node_id_to_index = {node: index for index, node in enumerate(node_order)}
        if node_order != sorted(node_order) or len(self.node_id_to_index) != len(node_order):
            raise ValueError("node_order must contain sorted unique node IDs")
        if set(node_order) != set(ancestors):
            raise ValueError("node_order and DAG nodes differ")
        self.supported_targets = frozenset(supported_targets)
        if not self.supported_targets or not self.supported_targets.issubset(node_order):
            raise ValueError("supported_targets must be non-empty DAG nodes")
        if isinstance(base_cost, bool) or not isinstance(base_cost, (int, float)):
            raise TypeError("base_cost must be numeric")
        if not math.isfinite(float(base_cost)) or float(base_cost) <= 0:
            raise ValueError("base_cost must be positive and finite")
        self.ancestors = dict(ancestors)
        self.base_cost_value = float(base_cost)
        self.config_hash = config_hash
        self.checkpoint_hash = checkpoint_hash
        self.teacher_tensor_hash = teacher_tensor_hash
        self.teacher_checkpoint_hash = teacher_checkpoint_hash
        self.tuple_collection_hash = tuple_collection_hash
        self._cache: dict[tuple[int, frozenset[int]], float] = {}

    @classmethod
    def from_artifacts(
        cls,
        config_path: str | Path = DEFAULT_CONFIG,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        dag_path: str | Path = DEFAULT_DAG,
        *,
        device: str | torch.device = "cpu",
    ) -> "DKTSetOracle":
        if torch.device(device).type != "cpu":
            raise ValueError("DKTSetOracle canonical inference device must be CPU")
        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        config_hash = sha256_file(config_path)
        checkpoint = load_deterministic_checkpoint(checkpoint_path)
        required = {
            "state_dict", "config_hash", "node_order", "supported_targets",
            "teacher_tensor_hash", "teacher_checkpoint_hash", "tuple_collection_hash",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"DKT surrogate checkpoint missing fields: {missing}")
        comparisons = {
            "config_hash": config_hash,
            "node_order": config["node_order"],
            "supported_targets": config["supported_targets"],
            "teacher_tensor_hash": config["teacher_tensor_hash"],
            "teacher_checkpoint_hash": config["teacher_checkpoint_hash"],
            "tuple_collection_hash": config["tuple_collection_hash"],
        }
        for key, expected in comparisons.items():
            if checkpoint[key] != expected:
                raise ValueError(f"DKT surrogate {key} mismatch")
        dag_nodes, dag_edges = load_dag(dag_path)
        if dag_nodes != config["node_order"]:
            raise ValueError("DAG nodes do not match DKT surrogate config")
        model = SetOracleSurrogate(num_nodes=len(dag_nodes))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        manifest = load_manifest()
        if float(manifest["base_cost"]) != float(config["base_cost"]):
            raise ValueError("Manifest and DKT surrogate base costs differ")
        return cls(
            model,
            node_order=dag_nodes,
            supported_targets=config["supported_targets"],
            ancestors=ancestor_map(dag_nodes, dag_edges),
            base_cost=config["base_cost"],
            config_hash=config_hash,
            checkpoint_hash=sha256_file(checkpoint_path),
            teacher_tensor_hash=config["teacher_tensor_hash"],
            teacher_checkpoint_hash=config["teacher_checkpoint_hash"],
            tuple_collection_hash=config["tuple_collection_hash"],
        )

    def _normalize_query(
        self, target_node: int, state: Iterable[int]
    ) -> tuple[int, frozenset[int]]:
        if not isinstance(target_node, int) or isinstance(target_node, bool):
            raise TypeError("target_node must be an integer node ID")
        if target_node not in self.node_id_to_index:
            raise ValueError(f"Unknown target node: {target_node}")
        if target_node not in self.supported_targets:
            raise ValueError(f"Target node is not covered by the surrogate: {target_node}")
        try:
            frozen = frozenset(state)
        except TypeError as error:
            raise TypeError("state must be an iterable of integer node IDs") from error
        if any(not isinstance(node, int) or isinstance(node, bool) for node in frozen):
            raise TypeError("state must contain only integer node IDs")
        unknown = sorted(frozen - self.node_id_to_index.keys())
        if unknown:
            raise ValueError(f"State contains unknown nodes: {unknown}")
        if target_node in frozen:
            raise ValueError("target_node must not already be mastered")
        invalid = sorted(node for node in frozen if not self.ancestors[node].issubset(frozen))
        if invalid:
            raise ValueError(f"state must be prerequisite-closed; invalid nodes: {invalid}")
        return target_node, frozen

    def success_prob(self, v: int, state: Iterable[int]) -> float:
        key = self._normalize_query(v, state)
        if key in self._cache:
            return self._cache[key]
        mask = torch.zeros(len(self.node_order), dtype=torch.float32)
        for node in key[1]:
            mask[self.node_id_to_index[node]] = 1.0
        target = torch.tensor(self.node_id_to_index[v], dtype=torch.long)
        with torch.inference_mode():
            value = float(self.model(mask, target).item())
        if not math.isfinite(value):
            raise ValueError("DKT set surrogate returned a non-finite probability")
        value = min(1.0, max(MIN_PROBABILITY, value))
        self._cache[key] = value
        return value

    def base_cost(self, v: int) -> float:
        self._normalize_query(v, ())
        return self.base_cost_value

    def best_case_success_prob(self, v: int) -> float:
        self._normalize_query(v, ())
        return 1.0
