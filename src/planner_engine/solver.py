"""
solver.py  --  DAG Planner: DP + memoization over mastery states.

Frozen Spec v0.2:
  - State key: frozenset(mastered_node_ids)
  - Termination: target_lo_nodes is a subset of state -> (0.0, [])
  - Bellman update: min over valid actions of (action_cost + V(next_state))
  - Cost = T_base + (1 - P_succ) * T_penalty + lambda_risk * sigma2
"""

from typing import Dict, FrozenSet, List, Set, Tuple

import networkx as nx
import torch

from src.planner_engine.zpd_utils import get_valid_actions


class DAGPlanner:
    """DP planner on a knowledge prerequisite DAG.

    Args:
        oracle:     object with predict_mc(x, edge_index, target_node,
                    current_prereq_mask, mc_samples=...) -> (mean_p, var_p, t_base)
        nx_graph:   networkx.DiGraph (must be a DAG)
        config:     dict with keys 'planner' and 'oracle'
        edge_index: torch.LongTensor [2, E]  (idx-based, for oracle)
        num_nodes:  int  (total number of nodes)
    """

    def __init__(
        self,
        oracle,
        nx_graph: nx.DiGraph,
        config: dict,
        edge_index: torch.Tensor,
        num_nodes: int,
    ):
        assert nx.is_directed_acyclic_graph(nx_graph), \
            "Graph must be a DAG -- cycle detected!"

        self.oracle = oracle
        self.graph = nx_graph
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.t_penalty: float = config["planner"]["t_penalty"]
        self.lambda_risk: float = config["planner"]["lambda_risk"]
        self.mc_samples: int = config.get("oracle", {}).get("mc_samples", 20)

        self.memo: Dict[FrozenSet[int], Tuple[float, List[int]]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_mask(self, state: FrozenSet[int]) -> torch.Tensor:
        """Convert mastered-node set to a float mask [N]."""
        mask = torch.zeros(self.num_nodes)
        for n in state:
            if 0 <= n < self.num_nodes:
                mask[n] = 1.0
        return mask

    def _zero_x(self) -> torch.Tensor:
        """Return a blank feature matrix [N, 2] (no history)."""
        return torch.zeros(self.num_nodes, 2)

    def _action_cost(self, action: int, state: FrozenSet[int]) -> float:
        """Query oracle for a single action's expected cost."""
        x = self._zero_x()
        mask = self._state_to_mask(state)
        target = torch.tensor(action, dtype=torch.long)

        mean_p, var_p, t_base = self.oracle.predict_mc(
            x, self.edge_index, target, mask,
            mc_samples=self.mc_samples,
        )

        cost = (
            t_base.item()
            + (1.0 - mean_p.item()) * self.t_penalty
            + self.lambda_risk * var_p.item()
        )
        return cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        current_state: Set[int],
        target_lo_nodes: Set[int],
    ) -> Tuple[float, List[int]]:
        """Find minimum-cost learning path.

        Args:
            current_state:   set of already-mastered node IDs
            target_lo_nodes: set of goal node IDs to master

        Returns:
            (total_cost, path)  path = ordered list of node IDs to master
        """
        self.memo.clear()
        return self._dp(frozenset(current_state), frozenset(target_lo_nodes))

    # ------------------------------------------------------------------
    # Recursive DP
    # ------------------------------------------------------------------

    def _dp(
        self,
        state: FrozenSet[int],
        target: FrozenSet[int],
    ) -> Tuple[float, List[int]]:
        # Termination: all target nodes mastered
        if target <= state:
            return (0.0, [])

        if state in self.memo:
            return self.memo[state]

        actions = get_valid_actions(self.graph, set(state))

        if not actions:
            # Should not happen on a well-formed DAG if target is reachable
            result: Tuple[float, List[int]] = (float("inf"), [])
            self.memo[state] = result
            return result

        best_cost = float("inf")
        best_path: List[int] = []

        for action in actions:
            ac = self._action_cost(action, state)
            next_state = state | frozenset([action])
            future_cost, future_path = self._dp(next_state, target)
            total = ac + future_cost

            if total < best_cost:
                best_cost = total
                best_path = [action] + future_path

        self.memo[state] = (best_cost, best_path)
        return (best_cost, best_path)
