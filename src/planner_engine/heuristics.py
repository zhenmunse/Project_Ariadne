"""
heuristics.py  --  Admissible heuristics for Ariadne SSP planning.
"""

from typing import FrozenSet, Protocol

import networkx as nx


class HeuristicOracle(Protocol):
    """Minimal interface required by sum_heuristic."""

    def base_cost(self, v: int) -> float:
        """Return T_v > 0."""

    def best_case_success_prob(self, v: int) -> float:
        """Return an upper bound on p(v, s)."""


def sum_heuristic(
    state: FrozenSet[int],
    target: FrozenSet[int],
    oracle: HeuristicOracle,
) -> float:
    """Return h(s) = sum_{v in target - s} T_v / p*(v).

    p*(v) must be an upper bound on the success probability for concept v.
    This ignores dependencies and optional helper concepts, so it is an
    admissible lower bound on the remaining expected cost.
    """
    h = 0.0
    for v in sorted(target - state):
        p_star = oracle.best_case_success_prob(v)
        if p_star <= 0.0:
            p_star = 1e-12
        elif p_star > 1.0:
            p_star = 1.0
        h += oracle.base_cost(v) / p_star
    return h


def max_heuristic(
    state: FrozenSet[int],
    target: FrozenSet[int],
    oracle: HeuristicOracle,
    graph: nx.DiGraph,
) -> float:
    """Return the largest collapsed-cost sum along a remaining DAG path.

    The calculation is restricted to the subgraph induced by target - state.
    Each node contributes T_v / p*(v), and a topological dynamic program finds
    the maximum path sum.  This critical-path bound is admissible because every
    node on such a path must still be mastered before the target is complete.
    """
    remaining = target - state
    if not remaining:
        return 0.0

    remaining_graph = graph.subgraph(remaining)
    longest_to: dict[int, float] = {}

    for v in nx.topological_sort(remaining_graph):
        p_star = oracle.best_case_success_prob(v)
        if p_star <= 0.0:
            p_star = 1e-12
        elif p_star > 1.0:
            p_star = 1.0

        predecessors = list(remaining_graph.predecessors(v))
        prefix = max((longest_to[u] for u in predecessors), default=0.0)
        longest_to[v] = prefix + oracle.base_cost(v) / p_star

    return max(longest_to.values(), default=0.0)
