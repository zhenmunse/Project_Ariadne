"""Reusable analyses for the Item 3 extended solver experiments."""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import FrozenSet

import networkx as nx

from src.planner_engine.solver_comparison import (
    AStarSolver,
    FullDPSolver,
    SearchMetrics,
    SolverResult,
    chi,
)
from src.planner_engine.zpd_utils import get_valid_actions


State = FrozenSet[int]


class NumericallyStableAStarSolver(AStarSolver):
    """A* with relative-tolerance g comparisons for synthetic float costs.

    The production comparison solver deliberately uses exact float comparisons.
    Synthetic paths can sum the same mathematical action multiset in different
    orders and differ by one ULP.  Treating that rounding artifact as a reopen
    would incorrectly reject a consistent heuristic.
    """

    @staticmethod
    def _better(candidate: float, incumbent: float) -> bool:
        tolerance = 1e-12 * max(1.0, abs(candidate), abs(incumbent))
        return candidate < incumbent - tolerance

    @staticmethod
    def _equal(candidate: float, incumbent: float) -> bool:
        tolerance = 1e-12 * max(1.0, abs(candidate), abs(incumbent))
        return abs(candidate - incumbent) <= tolerance

    def solve(self, initial_state: set[int], target_nodes: set[int]) -> SolverResult:
        initial, target = self._validate_solve_arguments(initial_state, target_nodes)
        self.metrics = SearchMetrics()
        started = time.perf_counter()
        counter = itertools.count()
        open_heap = []
        heapq.heappush(
            open_heap,
            (
                (self._heuristic(initial), -0.0, chi(initial, self.num_nodes)),
                next(counter),
                initial,
            ),
        )
        self.metrics.peak_open_size = 1
        best_g = {initial: 0.0}
        closed: dict[State, float] = {}
        parent = {}
        path_key = {initial: ()}
        goal = None
        goal_cost = float("inf")
        while open_heap:
            (_, negative_g, _), _, state = heapq.heappop(open_heap)
            popped_g = -negative_g
            if state in closed:
                continue
            closed[state] = popped_g
            self.metrics.expanded_states += 1
            if target <= state:
                goal, goal_cost = state, popped_g
                break
            for action in get_valid_actions(self.graph, set(state)):
                successor = frozenset(set(state) | {action})
                self.metrics.generated_states += 1
                candidate = math.fsum(
                    (popped_g, self._counted_action_cost(action, state))
                )
                if successor in closed:
                    if self._better(candidate, closed[successor]):
                        self.metrics.reopens += 1
                    continue
                candidate_path = path_key[state] + (action,)
                incumbent = best_g.get(successor)
                if incumbent is None or self._better(candidate, incumbent):
                    if incumbent is not None:
                        self.metrics.decrease_key_updates += 1
                    best_g[successor] = candidate
                    parent[successor] = (state, action)
                    path_key[successor] = candidate_path
                    heapq.heappush(
                        open_heap,
                        (
                            (
                                math.fsum((candidate, self._heuristic(successor))),
                                -candidate,
                                chi(successor, self.num_nodes),
                            ),
                            next(counter),
                            successor,
                        ),
                    )
                    self.metrics.peak_open_size = max(
                        self.metrics.peak_open_size, len(open_heap)
                    )
                elif (
                    self._equal(candidate, incumbent)
                    and candidate_path < path_key[successor]
                ):
                    parent[successor] = (state, action)
                    path_key[successor] = candidate_path
        if goal is None:
            raise AssertionError("A* failed to reach the target")
        if self.metrics.reopens:
            raise AssertionError(
                f"Consistent A* heuristic reopened {self.metrics.reopens} states"
            )
        self.metrics.stored_states = len(set(best_g) | set(closed))
        self.metrics.wall_seconds = time.perf_counter() - started
        return SolverResult(
            goal_cost, self._reconstruct(parent, goal, initial), self.metrics
        )


def enumerate_reachable_ideals(
    graph: nx.DiGraph,
    initial_state: State,
    target_nodes: State,
    *,
    timeout_seconds: float | None = None,
) -> tuple[set[State], dict[State, list[tuple[int, State]]], int, int]:
    """Enumerate reachable order ideals and their transitions once.

    Returns ``(states, successors, generated, peak_open)``.  This is shared by
    the delta analysis and the value-exposing DP instead of duplicating BFS.
    """
    started = time.perf_counter()
    queue = deque([initial_state])
    states = {initial_state}
    successors: dict[State, list[tuple[int, State]]] = {}
    generated = 0
    peak_open = 1
    while queue:
        if (
            timeout_seconds is not None
            and time.perf_counter() - started > timeout_seconds
        ):
            raise TimeoutError(
                f"Reachable-ideal enumeration exceeded {timeout_seconds:.1f}s"
            )
        state = queue.popleft()
        if target_nodes <= state:
            successors[state] = []
            continue
        outgoing = []
        for action in get_valid_actions(graph, set(state)):
            successor = frozenset(set(state) | {action})
            generated += 1
            outgoing.append((action, successor))
            if successor not in states:
                states.add(successor)
                queue.append(successor)
                peak_open = max(peak_open, len(queue))
        successors[state] = outgoing
    return states, successors, generated, peak_open


@dataclass(frozen=True)
class ConceptDelta:
    concept: int
    c_min: float
    c_max: float
    delta_v: float
    feasible_state_count: int


class DeltaAnalyzer:
    """Compute per-action cost ranges over every feasible reachable state."""

    def __init__(
        self,
        graph: nx.DiGraph,
        oracle: object,
        initial_state: State,
        target_nodes: State | None = None,
    ) -> None:
        self.graph = graph
        self.oracle = oracle
        self.initial_state = initial_state
        self.target_nodes = target_nodes or frozenset(graph.nodes())

    def analyze(self) -> tuple[list[ConceptDelta], set[State]]:
        states, _, _, _ = enumerate_reachable_ideals(
            self.graph, self.initial_state, self.target_nodes
        )
        costs: dict[int, list[float]] = {
            node: [] for node in self.target_nodes - self.initial_state
        }
        for state in sorted(
            states, key=lambda value: (len(value), tuple(sorted(value)))
        ):
            for action in get_valid_actions(self.graph, set(state)):
                if action in costs:
                    probability = float(self.oracle.success_prob(action, state))
                    if not math.isfinite(probability) or probability <= 0.0:
                        raise ValueError(
                            "Oracle probabilities must be positive and finite"
                        )
                    costs[action].append(
                        float(self.oracle.base_cost(action)) / probability
                    )
        records = []
        for concept in sorted(costs):
            values = costs[concept]
            if not values:
                raise AssertionError(
                    f"Concept {concept} has no feasible reachable state"
                )
            c_min = min(values)
            c_max = max(values)
            records.append(
                ConceptDelta(concept, c_min, c_max, c_max - c_min, len(values))
            )
        return records, states


class FullDPWithValues(FullDPSolver):
    """Full reverse-topological DP that retains every state value."""

    all_values: dict[State, float]

    def __init__(self, *args, timeout_seconds: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.all_values = {}

    def solve(self, initial_state: set[int], target_nodes: set[int]) -> SolverResult:
        initial, target = self._validate_solve_arguments(initial_state, target_nodes)
        self.metrics = type(self.metrics)()
        started = time.perf_counter()
        states, successors, generated, peak_open = enumerate_reachable_ideals(
            self.graph,
            initial,
            target,
            timeout_seconds=self.timeout_seconds,
        )
        self.metrics.generated_states = generated
        self.metrics.peak_open_size = peak_open
        values: dict[State, float] = {}
        policy: dict[State, int] = {}
        for state in sorted(
            states, key=lambda value: (-len(value), chi(value, self.num_nodes))
        ):
            if target <= state:
                values[state] = 0.0
                continue
            self.metrics.expanded_states += 1
            candidates = []
            for action, successor in successors[state]:
                candidates.append(
                    (
                        math.fsum(
                            (
                                self._counted_action_cost(action, state),
                                values[successor],
                            )
                        ),
                        action,
                    )
                )
            best_value, best_action = min(candidates)
            values[state] = best_value
            policy[state] = best_action
        sequence: list[int] = []
        state = initial
        while not target <= state:
            action = policy[state]
            sequence.append(action)
            state = frozenset(set(state) | {action})
        self.all_values = values
        self.metrics.stored_states = len(states)
        self.metrics.wall_seconds = time.perf_counter() - started
        return SolverResult(values[initial], sequence, self.metrics)


@dataclass(frozen=True)
class FrustrationSummary:
    rho_s0: float
    rho_mean: float
    rho_median: float
    rho_min: float
    rho_max: float
    rho_std: float
    reachable_states: int


class FrustrationAnalyzer:
    """Measure residual cost-to-go not captured by ``h_sum``."""

    def __init__(
        self,
        values: dict[State, float],
        initial_state: State,
        target_nodes: State,
        p_bar: dict[int, float],
        base_cost_map: dict[int, float],
    ) -> None:
        self.values = values
        self.initial_state = initial_state
        self.target_nodes = target_nodes
        self.p_bar = p_bar
        self.base_cost_map = base_cost_map

    def _heuristic(self, state: State) -> float:
        return math.fsum(
            self.base_cost_map[node] / self.p_bar[node]
            for node in self.target_nodes - state
        )

    def analyze(self) -> FrustrationSummary:
        import statistics

        rhos: dict[State, float] = {}
        for state, value in self.values.items():
            if value <= 0.0:
                continue
            rho = (value - self._heuristic(state)) / value
            if rho < -1e-10:
                raise AssertionError(
                    f"Heuristic exceeds exact value at {sorted(state)}"
                )
            rhos[state] = max(0.0, rho)
        values = list(rhos.values())
        if self.initial_state not in rhos:
            raise AssertionError("Initial state is absent from nonterminal DP values")
        return FrustrationSummary(
            rho_s0=rhos[self.initial_state],
            rho_mean=statistics.fmean(values),
            rho_median=statistics.median(values),
            rho_min=min(values),
            rho_max=max(values),
            rho_std=statistics.pstdev(values),
            reachable_states=len(self.values),
        )


__all__ = [
    "ConceptDelta",
    "DeltaAnalyzer",
    "FrustrationAnalyzer",
    "FrustrationSummary",
    "FullDPWithValues",
    "NumericallyStableAStarSolver",
    "enumerate_reachable_ideals",
]
