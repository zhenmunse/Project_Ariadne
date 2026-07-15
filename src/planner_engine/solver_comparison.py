"""Instrumented exact/search solvers for the Item 3 comparison.

The classes in this module solve the collapsed deterministic mastery problem
without changing the production planner.  Every transition is
``s -> s union {v}`` with geometric expected cost ``T_v / p(v, s)``.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Protocol, Set, Tuple

import networkx as nx

from src.planner_engine.solver import DAGPlanner
from src.planner_engine.zpd_utils import get_valid_actions


State = FrozenSet[int]
Parent = Tuple[State, int]
RELATIVE_TOLERANCE = 1e-12


class OracleProtocol(Protocol):
    def success_prob(self, v: int, state: State) -> float: ...

    def base_cost(self, v: int) -> float: ...

    def best_case_success_prob(self, v: int) -> float: ...


@dataclass
class SearchMetrics:
    """Counters shared by all four solvers."""

    expanded_states: int = 0
    generated_states: int = 0
    oracle_calls: int = 0
    edge_cost_evaluations: int = 0
    peak_open_size: int = 0
    stored_states: int = 0
    wall_seconds: float = 0.0
    decrease_key_updates: int = 0
    reopens: int = 0
    revision_passes: int = 0
    state_backups: int = 0
    marked_action_changes: int = 0


@dataclass
class SolverResult:
    optimal_cost: float
    optimal_sequence: List[int]
    metrics: SearchMetrics


class SolverProtocol(Protocol):
    def solve(
        self,
        initial_state: Set[int],
        target_nodes: Set[int],
    ) -> SolverResult: ...


def chi(state: State, n_concepts: int) -> Tuple[int, ...]:
    """Characteristic bit tuple under the global concept ordering."""
    return tuple(1 if node in state else 0 for node in range(n_concepts))


def _global_num_nodes(oracle: object, graph: nx.DiGraph) -> int:
    mapping = getattr(oracle, "node_id_to_idx", None)
    if isinstance(mapping, dict) and mapping:
        return max(int(node) for node in mapping) + 1
    model = getattr(oracle, "model", None)
    if model is not None and hasattr(model, "num_nodes"):
        return int(model.num_nodes)
    return max((int(node) for node in graph.nodes()), default=-1) + 1


def _strictly_better(candidate: float, incumbent: float) -> bool:
    return candidate < incumbent


class _ComparisonSolverBase:
    def __init__(
        self,
        oracle: OracleProtocol,
        nx_graph: nx.DiGraph,
        target_nodes: State,
        initial_state: State,
        p_bar: Dict[int, float],
        base_cost_map: Dict[int, float],
    ) -> None:
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise ValueError("Comparison solver requires a DAG")
        if set(target_nodes) != set(nx_graph.nodes()):
            raise ValueError("target_nodes must equal the closure graph nodes")
        if not initial_state <= target_nodes:
            raise ValueError("initial_state must be a subset of target_nodes")
        if set(p_bar) != set(target_nodes):
            raise ValueError("p_bar must cover target_nodes exactly")
        if set(base_cost_map) != set(target_nodes):
            raise ValueError("base_cost_map must cover target_nodes exactly")
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in p_bar.values()
        ):
            raise ValueError("Every p_bar value must be finite and in (0, 1]")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in base_cost_map.values()
        ):
            raise ValueError("Every base cost must be positive and finite")

        self.oracle = oracle
        self.graph = nx_graph.copy()
        self.target_nodes = frozenset(target_nodes)
        self.initial_state = frozenset(initial_state)
        self.p_bar = dict(p_bar)
        self.base_cost_map = dict(base_cost_map)
        self.num_nodes = _global_num_nodes(oracle, nx_graph)
        self.metrics = SearchMetrics()

    def _validate_solve_arguments(
        self,
        initial_state: Set[int],
        target_nodes: Set[int],
    ) -> Tuple[State, State]:
        initial = frozenset(initial_state)
        target = frozenset(target_nodes)
        if initial != self.initial_state or target != self.target_nodes:
            raise ValueError(
                "solve arguments must match the constructor state and target"
            )
        return initial, target

    def _counted_action_cost(self, action: int, state: State) -> float:
        """Count and evaluate one geometric edge cost."""
        self.metrics.edge_cost_evaluations += 1
        self.metrics.oracle_calls += 1
        probability = float(self.oracle.success_prob(action, state))
        if not math.isfinite(probability) or probability <= 0.0:
            raise ValueError(
                f"success_prob({action}, state) must be positive and finite"
            )
        probability = min(1.0, probability)
        return self.base_cost_map[action] / probability

    def _heuristic(self, state: State) -> float:
        return math.fsum(
            self.base_cost_map[node] / self.p_bar[node]
            for node in self.target_nodes - state
        )

    @staticmethod
    def _reconstruct(
        parent: Dict[State, Parent], goal: State, initial: State
    ) -> List[int]:
        sequence: List[int] = []
        state = goal
        while state != initial:
            if state not in parent:
                raise AssertionError("Missing parent while reconstructing optimal path")
            predecessor, action = parent[state]
            sequence.append(action)
            state = predecessor
        sequence.reverse()
        return sequence


class FullDPSolver(_ComparisonSolverBase):
    """Enumerate every reachable ideal, then apply reverse-cardinality DP."""

    def solve(self, initial_state: Set[int], target_nodes: Set[int]) -> SolverResult:
        initial, target = self._validate_solve_arguments(initial_state, target_nodes)
        self.metrics = SearchMetrics()
        started = time.perf_counter()

        queue = deque([initial])
        discovered = {initial}
        successors: Dict[State, List[Tuple[int, State]]] = {}
        self.metrics.peak_open_size = 1

        while queue:
            state = queue.popleft()
            if target <= state:
                successors[state] = []
                continue
            state_successors: List[Tuple[int, State]] = []
            for action in get_valid_actions(self.graph, set(state)):
                successor = frozenset(set(state) | {action})
                self.metrics.generated_states += 1
                state_successors.append((action, successor))
                if successor not in discovered:
                    discovered.add(successor)
                    queue.append(successor)
                    self.metrics.peak_open_size = max(
                        self.metrics.peak_open_size, len(queue)
                    )
            successors[state] = state_successors

        values: Dict[State, float] = {}
        policy: Dict[State, int] = {}
        order = sorted(
            discovered, key=lambda state: (-len(state), chi(state, self.num_nodes))
        )
        for state in order:
            if target <= state:
                values[state] = 0.0
                continue
            self.metrics.expanded_states += 1
            best_value = float("inf")
            best_action: int | None = None
            for action, successor in successors[state]:
                total = math.fsum(
                    (self._counted_action_cost(action, state), values[successor])
                )
                if _strictly_better(total, best_value) or (
                    total == best_value
                    and (best_action is None or action < best_action)
                ):
                    best_value = total
                    best_action = action
            if best_action is None:
                raise AssertionError(
                    f"Nonterminal state has no valid action: {sorted(state)}"
                )
            values[state] = best_value
            policy[state] = best_action

        sequence: List[int] = []
        state = initial
        while not target <= state:
            action = policy[state]
            sequence.append(action)
            state = frozenset(set(state) | {action})

        self.metrics.stored_states = len(discovered)
        self.metrics.wall_seconds = time.perf_counter() - started
        return SolverResult(values[initial], sequence, self.metrics)


class DijkstraSolver(_ComparisonSolverBase):
    """Uniform-cost graph search on the collapsed mastery lattice."""

    def solve(self, initial_state: Set[int], target_nodes: Set[int]) -> SolverResult:
        initial, target = self._validate_solve_arguments(initial_state, target_nodes)
        self.metrics = SearchMetrics()
        started = time.perf_counter()
        counter = itertools.count()
        open_heap: List[Tuple[Tuple[float, Tuple[int, ...]], int, State]] = []
        heapq.heappush(
            open_heap, ((0.0, chi(initial, self.num_nodes)), next(counter), initial)
        )
        self.metrics.peak_open_size = 1
        closed: Dict[State, float] = {}
        best_g: Dict[State, float] = {initial: 0.0}
        parent: Dict[State, Parent] = {}
        path_key: Dict[State, Tuple[int, ...]] = {initial: ()}
        goal: State | None = None
        goal_cost = float("inf")

        while open_heap:
            (popped_g, _), _, state = heapq.heappop(open_heap)
            if state in closed:
                continue
            closed[state] = popped_g
            self.metrics.expanded_states += 1
            if target <= state:
                goal = state
                goal_cost = popped_g
                break
            for action in get_valid_actions(self.graph, set(state)):
                successor = frozenset(set(state) | {action})
                self.metrics.generated_states += 1
                candidate_g = math.fsum(
                    (popped_g, self._counted_action_cost(action, state))
                )
                if successor in closed:
                    continue
                candidate_path = path_key[state] + (action,)
                old_g = best_g.get(successor)
                if old_g is None or candidate_g < old_g:
                    best_g[successor] = candidate_g
                    parent[successor] = (state, action)
                    path_key[successor] = candidate_path
                elif candidate_g == old_g and candidate_path < path_key[successor]:
                    parent[successor] = (state, action)
                    path_key[successor] = candidate_path
                heapq.heappush(
                    open_heap,
                    (
                        (candidate_g, chi(successor, self.num_nodes)),
                        next(counter),
                        successor,
                    ),
                )
                self.metrics.peak_open_size = max(
                    self.metrics.peak_open_size, len(open_heap)
                )

        if goal is None:
            raise AssertionError("Dijkstra failed to reach the target")
        self.metrics.stored_states = len(set(best_g) | set(closed))
        self.metrics.wall_seconds = time.perf_counter() - started
        return SolverResult(
            goal_cost, self._reconstruct(parent, goal, initial), self.metrics
        )


class AStarSolver(_ComparisonSolverBase):
    """Graph-search A* using the precomputed admissible sum heuristic."""

    def solve(self, initial_state: Set[int], target_nodes: Set[int]) -> SolverResult:
        initial, target = self._validate_solve_arguments(initial_state, target_nodes)
        self.metrics = SearchMetrics()
        started = time.perf_counter()
        counter = itertools.count()
        initial_g = 0.0
        initial_f = self._heuristic(initial)
        open_heap: List[Tuple[Tuple[float, float, Tuple[int, ...]], int, State]] = []
        heapq.heappush(
            open_heap,
            (
                (initial_f, -initial_g, chi(initial, self.num_nodes)),
                next(counter),
                initial,
            ),
        )
        self.metrics.peak_open_size = 1
        best_g: Dict[State, float] = {initial: initial_g}
        closed: Dict[State, float] = {}
        parent: Dict[State, Parent] = {}
        path_key: Dict[State, Tuple[int, ...]] = {initial: ()}
        goal: State | None = None
        goal_cost = float("inf")

        while open_heap:
            (_, negative_g, _), _, state = heapq.heappop(open_heap)
            popped_g = -negative_g
            if state in closed:
                continue
            closed[state] = popped_g
            self.metrics.expanded_states += 1
            if target <= state:
                goal = state
                goal_cost = popped_g
                break
            for action in get_valid_actions(self.graph, set(state)):
                successor = frozenset(set(state) | {action})
                self.metrics.generated_states += 1
                candidate_g = math.fsum(
                    (popped_g, self._counted_action_cost(action, state))
                )
                if successor in closed:
                    if candidate_g < closed[successor]:
                        self.metrics.reopens += 1
                    continue
                candidate_path = path_key[state] + (action,)
                old_g = best_g.get(successor)
                if old_g is None or candidate_g < old_g:
                    if old_g is not None:
                        self.metrics.decrease_key_updates += 1
                    best_g[successor] = candidate_g
                    parent[successor] = (state, action)
                    path_key[successor] = candidate_path
                    successor_f = math.fsum((candidate_g, self._heuristic(successor)))
                    heapq.heappush(
                        open_heap,
                        (
                            (successor_f, -candidate_g, chi(successor, self.num_nodes)),
                            next(counter),
                            successor,
                        ),
                    )
                    self.metrics.peak_open_size = max(
                        self.metrics.peak_open_size, len(open_heap)
                    )
                elif candidate_g == old_g and candidate_path < path_key[successor]:
                    parent[successor] = (state, action)
                    path_key[successor] = candidate_path

        if goal is None:
            raise AssertionError("A* failed to reach the target")
        if self.metrics.reopens != 0:
            raise AssertionError(
                f"Consistent A* heuristic reopened {self.metrics.reopens} states"
            )
        self.metrics.stored_states = len(set(best_g) | set(closed))
        self.metrics.wall_seconds = time.perf_counter() - started
        return SolverResult(
            goal_cost, self._reconstruct(parent, goal, initial), self.metrics
        )


class LAOStarWrapper(DAGPlanner):
    """Instrument the production LAO*-derived DAGPlanner without modifying it."""

    def __init__(
        self,
        oracle: OracleProtocol,
        nx_graph: nx.DiGraph,
        target_nodes: State,
        initial_state: State,
        p_bar: Dict[int, float],
        base_cost_map: Dict[int, float],
    ) -> None:
        self.comparison_target = frozenset(target_nodes)
        self.comparison_initial = frozenset(initial_state)
        self.precomputed_p_bar = dict(p_bar)
        self.precomputed_base_cost = dict(base_cost_map)
        self.metrics = SearchMetrics()
        num_nodes = _global_num_nodes(oracle, nx_graph)
        super().__init__(
            oracle=oracle,
            nx_graph=nx_graph,
            config={"planner": {"base_cost": 60.0, "heuristic": "sum"}},
            edge_index=getattr(oracle, "edge_index", None),
            num_nodes=num_nodes,
        )

    def base_cost(self, v: int) -> float:
        return self.precomputed_base_cost[v]

    def best_case_success_prob(self, v: int) -> float:
        return self.precomputed_p_bar[v]

    def _action_cost(self, action: int, state: State) -> float:
        self.metrics.edge_cost_evaluations += 1
        self.metrics.oracle_calls += 1
        probability = float(self.oracle.success_prob(action, state))
        if not math.isfinite(probability) or probability <= 0.0:
            raise ValueError(
                f"success_prob({action}, state) must be positive and finite"
            )
        return self.precomputed_base_cost[action] / min(1.0, probability)

    def _expand_state(self, state: State, target: State) -> None:
        node = self._ensure_state(state, target)
        if node.is_expanded or node.is_terminal:
            return
        self.metrics.expanded_states += 1
        self.metrics.generated_states += len(get_valid_actions(self.graph, set(state)))
        super()._expand_state(state, target)

    def _exact_cost_revision(self) -> None:
        self.metrics.revision_passes += 1
        states = [
            state
            for state, node in self.graph_explicit.expanded.items()
            if node.is_expanded and not node.is_terminal
        ]
        states.sort(key=lambda state: -len(state))
        for state in states:
            node = self.graph_explicit[state]
            old_action = node.best_action
            value, action = self._bellman_update(state)
            self.metrics.state_backups += 1
            if old_action is not None and action != old_action:
                self.metrics.marked_action_changes += 1
            node.value = value
            node.best_action = action

    def solve(self, initial_state: Set[int], target_nodes: Set[int]) -> SolverResult:
        initial = frozenset(initial_state)
        target = frozenset(target_nodes)
        if initial != self.comparison_initial or target != self.comparison_target:
            raise ValueError(
                "solve arguments must match the constructor state and target"
            )
        self.metrics = SearchMetrics()
        started = time.perf_counter()
        result = self.solve_result(set(initial), set(target))
        sequence = self._extract_path(initial, target, result.policy)
        self.metrics.stored_states = len(self.graph_explicit.expanded)
        self.metrics.wall_seconds = time.perf_counter() - started
        if not result.converged:
            raise AssertionError("LAO*-derived solver did not converge")
        return SolverResult(float(result.values[initial]), sequence, self.metrics)


__all__ = [
    "AStarSolver",
    "DijkstraSolver",
    "FullDPSolver",
    "LAOStarWrapper",
    "OracleProtocol",
    "SearchMetrics",
    "SolverProtocol",
    "SolverResult",
    "chi",
]
