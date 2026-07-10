"""
baselines.py  --  Ablation baselines for Project Ariadne.

GreedyPlanner:  one-step-lookahead (myopic) planner.
    Proves that DAGPlanner's long-horizon DP outperforms local greedy.

FrequencyOracle: naive frequency-based success-probability predictor.
    Proves that MonotonicOracle's GNN structure sharing matters,
    especially under few-shot / cold-start conditions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

import networkx as nx

from src.planner_engine.zpd_utils import get_valid_actions


# ==================================================================
# GreedyPlanner
# ==================================================================

class GreedyPlanner:
    """One-step-lookahead planner (myopic baseline).

    At each step picks the valid action with the lowest immediate cost.
    Never considers future consequences.

    Determinism tie-break: when costs are equal, chooses smallest node_id.

    Interface mirrors DAGPlanner for drop-in replacement.
    """

    def __init__(
        self,
        oracle,
        nx_graph: nx.DiGraph,
        config: dict,
        edge_index,
        num_nodes: int,
    ):
        assert nx.is_directed_acyclic_graph(nx_graph), \
            "Graph must be a DAG -- cycle detected!"

        self.oracle = oracle
        self.graph = nx_graph
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.mc_samples: int = config.get("oracle", {}).get("mc_samples", 20)
        self.default_base_cost: float = float(
            config.get("planner", {}).get(
                "base_cost",
                config.get("oracle", {}).get("t_base", 60.0),
            )
        )

    def _state_to_mask(self, state: Set[int]) -> torch.Tensor:
        import torch

        mask = torch.zeros(self.num_nodes)
        for n in state:
            if 0 <= n < self.num_nodes:
                mask[n] = 1.0
        return mask

    def _zero_x(self) -> torch.Tensor:
        import torch

        return torch.zeros(self.num_nodes, 2)

    @staticmethod
    def _as_float(value: object) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)  # type: ignore[arg-type]

    def _action_cost(self, action: int, state: Set[int]) -> float:
        """Return geometric one-step expected cost T_v / p(v, s)."""
        frozen_state = frozenset(state)

        if hasattr(self.oracle, "action_cost"):
            cost = self._as_float(self.oracle.action_cost(action, frozen_state))
            if cost <= 0.0:
                raise ValueError(f"action_cost({action}) must be positive, got {cost}")
            return cost

        if hasattr(self.oracle, "success_prob"):
            p = self._as_float(self.oracle.success_prob(action, frozen_state))
            if hasattr(self.oracle, "base_cost"):
                T = self._as_float(self.oracle.base_cost(action))
            else:
                T = self.default_base_cost
        else:
            import torch

            x = self._zero_x()
            mask = self._state_to_mask(state)
            target_t = torch.tensor(action, dtype=torch.long)
            mean_p, _var_p, t_base = self.oracle.predict_mc(
                x,
                self.edge_index,
                target_t,
                mask,
                mc_samples=self.mc_samples,
            )
            p = self._as_float(mean_p)
            T = self._as_float(t_base)

        p = min(max(p, 1e-12), 1.0)
        if T <= 0.0:
            T = self.default_base_cost
        return T / p

    def solve(
        self,
        current_state: Set[int],
        target_lo_nodes: Set[int],
    ) -> Tuple[float, List[int]]:
        """Greedy forward search.

        Returns:
            (total_cost, path)  -- accumulated one-step costs and ordered path.
        """
        state = set(current_state)
        path: List[int] = []
        total_cost = 0.0

        while not target_lo_nodes <= state:
            actions = get_valid_actions(self.graph, state)
            if not actions:
                return (float("inf"), path)

            # Evaluate one-step cost for every candidate
            best_action = -1
            best_cost = float("inf")

            for action in actions:
                cost = self._action_cost(action, state)
                # Tie-break: smaller node_id wins
                if cost < best_cost or (cost == best_cost and action < best_action):
                    best_cost = cost
                    best_action = action

            state.add(best_action)
            path.append(best_action)
            total_cost += best_cost

        return (total_cost, path)


# ==================================================================
# FrequencyOracle
# ==================================================================

class FrequencyOracle:
    """Naive frequency-based oracle (no graph structure, no GNN).

    Computes per-node global average label from training data.
    Ignores prerequisites entirely -- success_prob returns the same
    probability regardless of mask.

    Cold-start policy: unseen nodes get the global mean label.
    Variance estimate: p_hat * (1 - p_hat)  (Bernoulli variance proxy).
    T_base: fixed constant (default 60.0, configurable).

    Interface supports both the current success_prob/base_cost API and the
    legacy MonotonicOracle.predict_mc API for drop-in use.
    """

    def __init__(
        self,
        train_samples: list,
        num_nodes: int,
        node_id_to_idx: Dict[int, int],
        t_base: float = 60.0,
    ):
        self.num_nodes = num_nodes
        self.node_id_to_idx = node_id_to_idx
        self.t_base = t_base

        # Accumulate per-idx statistics
        sums: Dict[int, float] = defaultdict(float)
        counts: Dict[int, int] = defaultdict(int)

        for _hist, tgt_node, label in train_samples:
            if tgt_node in node_id_to_idx:
                idx = node_id_to_idx[tgt_node]
                sums[idx] += label
                counts[idx] += 1

        # Global fallback mean (across all samples)
        total_sum = sum(sums.values())
        total_cnt = sum(counts.values())
        self.global_mean = total_sum / max(total_cnt, 1)

        # Per-idx average
        import torch

        self.p_hat = torch.full((num_nodes,), self.global_mean)
        for idx in range(num_nodes):
            if counts[idx] > 0:
                self.p_hat[idx] = sums[idx] / counts[idx]

    def predict_mc(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_node: torch.Tensor,
        current_prereq_mask: torch.Tensor,
        mc_samples: int = 20,
        t_base: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return frequency-based prediction (ignores mask & graph).

        Returns:
            mean_p:  scalar tensor  -- p_hat[target_node]
            var_p:   scalar tensor  -- p_hat * (1 - p_hat)
            t_base:  scalar tensor  -- fixed constant
        """
        node = target_node.item() if hasattr(target_node, "item") else int(target_node)
        import torch

        p = self.p_hat[node].clone()
        v = p * (1.0 - p)
        tb = self.t_base if t_base < 0 else t_base
        return (
            p.unsqueeze(0).squeeze(),  # ensure scalar
            v.unsqueeze(0).squeeze(),
            torch.tensor(tb),
        )

    def success_prob(self, v: int, mastered: FrozenSet[int]) -> float:
        """Return p(v, s); this baseline ignores the mastered set."""
        return float(self.p_hat[v].item())

    def base_cost(self, v: int) -> float:
        """Return the fixed attempt-time cost for concept v."""
        return float(self.t_base)

    def best_case_success_prob(self, v: int) -> float:
        """Return the best-case success probability for the heuristic."""
        return self.success_prob(v, frozenset())
