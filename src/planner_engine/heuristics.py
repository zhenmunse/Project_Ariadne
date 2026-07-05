"""
heuristics.py  --  Admissible heuristics for Ariadne SSP planning.
"""

from typing import FrozenSet, Protocol


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
