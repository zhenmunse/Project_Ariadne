"""Deterministic MonotonicOracle adapter for planning and evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch

from src.oracle_core.model import MonotonicOracle
from experiments.common.manifest import _load_dag


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = ROOT / "data" / "processed" / "oracle_ckpt.pt"
DEFAULT_DAG_PATH = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
MIN_PROBABILITY = 1e-12


class FrozenMonotonicOracle:
    """Expose a checkpoint as a fixed function of ``(node, mastery set)``.

    Inference always uses ``model.eval()`` and a zero feature tensor.  The
    CUDA is explicit and mandatory by default so experiment runs cannot
    silently fall back to CPU.
    """

    def __init__(
        self,
        model: MonotonicOracle,
        edge_index: torch.Tensor,
        node_id_to_idx: dict[int, int],
        *,
        base_cost: float = 60.0,
        device: str | torch.device = "cuda",
    ) -> None:
        if isinstance(base_cost, bool) or not isinstance(base_cost, (int, float)):
            raise TypeError("base_cost must be a positive number")
        if not math.isfinite(base_cost) or base_cost <= 0:
            raise ValueError("base_cost must be a positive finite number")
        if not node_id_to_idx:
            raise ValueError("node_id_to_idx must not be empty")
        if any(
            not isinstance(node_id, int)
            or isinstance(node_id, bool)
            or not isinstance(index, int)
            or isinstance(index, bool)
            for node_id, index in node_id_to_idx.items()
        ):
            raise TypeError("node_id_to_idx must map integer node IDs to integer indices")

        indices = sorted(node_id_to_idx.values())
        if indices != list(range(len(indices))):
            raise ValueError("node indices must be contiguous from zero")
        if model.num_nodes != len(node_id_to_idx):
            raise ValueError("model and node mapping have different node counts")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.model = model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.edge_index = edge_index.to(device=self.device, dtype=torch.long)
        self.node_id_to_idx = dict(node_id_to_idx)
        self.base_cost_value = float(base_cost)
        self.x_zero = torch.zeros(model.num_nodes, 2, device=self.device)
        self._cache: dict[tuple[int, frozenset[int]], float] = {}

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        dag_path: str | Path = DEFAULT_DAG_PATH,
        *,
        base_cost: float = 60.0,
        device: str | torch.device = "cuda",
    ) -> "FrozenMonotonicOracle":
        """Build the adapter from the frozen checkpoint and DAG artifacts."""
        device = torch.device(device)
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=False
        )
        required = {"state_dict", "config", "node_id_to_idx", "num_nodes"}
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Oracle checkpoint is missing fields: {missing}")

        raw_mapping = checkpoint["node_id_to_idx"]
        if not isinstance(raw_mapping, dict):
            raise TypeError("checkpoint node_id_to_idx must be a dictionary")
        node_id_to_idx = dict(raw_mapping)

        dag_nodes, dag_edges = _load_dag(Path(dag_path))
        if set(dag_nodes) != set(node_id_to_idx):
            raise ValueError("DAG and checkpoint use different node IDs")
        indexed_edges: list[tuple[int, int]] = []
        for src, dst in dag_edges:
            indexed_edges.append((node_id_to_idx[src], node_id_to_idx[dst]))
        if indexed_edges:
            edge_index = torch.tensor(
                indexed_edges, dtype=torch.long, device=device
            ).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        config = checkpoint["config"]
        model = MonotonicOracle(
            num_nodes=checkpoint["num_nodes"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return cls(
            model,
            edge_index,
            node_id_to_idx,
            base_cost=base_cost,
            device=device,
        )

    def _normalize_query(
        self, node_id: int, state: Iterable[int]
    ) -> tuple[int, frozenset[int]]:
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise TypeError("node_id must be an integer")
        if node_id not in self.node_id_to_idx:
            raise ValueError(f"Unknown target node ID: {node_id}")
        try:
            frozen_state = frozenset(state)
        except TypeError as error:
            raise TypeError("state must be an iterable of integer node IDs") from error
        if any(not isinstance(node, int) or isinstance(node, bool) for node in frozen_state):
            raise TypeError("state must contain only integer node IDs")
        unknown = sorted(frozen_state - self.node_id_to_idx.keys())
        if unknown:
            raise ValueError(f"State contains unknown node IDs: {unknown}")
        return node_id, frozen_state

    def success_prob(self, node_id: int, state: Iterable[int]) -> float:
        """Return deterministic ``p(node_id, state)`` with content-based caching."""
        key = self._normalize_query(node_id, state)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        target_idx = self.node_id_to_idx[node_id]
        mask = torch.zeros(self.model.num_nodes, device=self.device)
        for mastered_id in key[1]:
            mask[self.node_id_to_idx[mastered_id]] = 1.0
        target_tensor = torch.tensor(target_idx, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            probability, _ = self.model.forward(
                self.x_zero,
                self.edge_index,
                target_tensor,
                mask,
            )
        value = float(probability.detach().cpu().item())
        if not math.isfinite(value):
            raise ValueError("Oracle returned a non-finite probability")
        value = min(1.0, max(MIN_PROBABILITY, value))
        self._cache[key] = value
        return value

    def base_cost(self, node_id: int) -> float:
        """Return the protocol's uniform per-attempt cost."""
        self._normalize_query(node_id, ())
        return self.base_cost_value

    def best_case_success_prob(self, node_id: int) -> float:
        """Return the only currently certified heuristic upper bound."""
        self._normalize_query(node_id, ())
        return 1.0
