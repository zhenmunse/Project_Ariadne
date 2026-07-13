"""Run the Item 3 delta, topology, synthetic, and frustration experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common.frozen_oracle import FrozenMonotonicOracle  # noqa: E402
from experiments.common.manifest import load_manifest, manifest_hash  # noqa: E402
from src.oracle_core.bkt_set_oracle import BKTSetOracle  # noqa: E402
from src.oracle_core.dkt_set_oracle import DKTSetOracle  # noqa: E402
from src.planner_engine.baselines import GreedyPlanner  # noqa: E402
from src.planner_engine.solver_analysis import (  # noqa: E402
    DeltaAnalyzer,
    FrustrationAnalyzer,
    FullDPWithValues,
    NumericallyStableAStarSolver,
)
from src.planner_engine.solver_comparison import (  # noqa: E402
    AStarSolver,
    DijkstraSolver,
    FullDPSolver,
    LAOStarWrapper,
    SolverResult,
)


DELTA_OUTPUT = ROOT / "results" / "delta_analysis"
TOPOLOGY_OUTPUT = ROOT / "results" / "topology_diagnostics"
SYNTHETIC_OUTPUT = ROOT / "results" / "synthetic_scaling"
FRUSTRATION_OUTPUT = ROOT / "results" / "frustration_analysis"
SOLVERS = ("full_dp", "dijkstra", "astar", "lao")
N_VALUES = (10, 15, 20, 25, 30)
W_VALUES = (2, 3, 5)
BETA_VALUES = (0.0, 0.5, 1.0, 2.0)
SEEDS = (42, 43, 44)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def _graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def _frozen(manifest: dict) -> FrozenMonotonicOracle:
    return FrozenMonotonicOracle.from_artifacts(
        base_cost=manifest["base_cost"], device="cpu"
    )


def _maps(
    oracle: object,
    nodes: frozenset[int],
    *,
    mode: str = "oracle",
) -> tuple[dict[int, float], dict[int, float]]:
    base = {node: float(oracle.base_cost(node)) for node in nodes}
    if mode == "loose":
        p_bar = {node: 1.0 for node in nodes}
    elif mode == "tight":
        mapping = getattr(oracle, "node_id_to_idx", None)
        universe = frozenset(mapping) if isinstance(mapping, dict) else nodes
        p_bar = {
            node: float(oracle.success_prob(node, frozenset(universe - {node})))
            for node in nodes
        }
    elif mode == "oracle":
        p_bar = {node: float(oracle.best_case_success_prob(node)) for node in nodes}
    else:
        raise ValueError(f"Unknown p_bar mode: {mode}")
    return p_bar, base


def _solver(
    solver_id: str,
    oracle: object,
    graph: nx.DiGraph,
    nodes: frozenset[int],
    initial: frozenset[int],
    p_bar: dict[int, float],
    base: dict[int, float],
):
    cls = {
        "full_dp": FullDPSolver,
        "dijkstra": DijkstraSolver,
        "astar": AStarSolver,
        "lao": LAOStarWrapper,
    }[solver_id]
    return cls(oracle, graph, nodes, initial, p_bar, base)


def _relative_cost_check(results: dict[str, SolverResult]) -> None:
    reference = next(iter(results.values())).optimal_cost
    tolerance = 1e-12 * max(1.0, abs(reference))
    for solver_id, result in results.items():
        if not abs(result.optimal_cost - reference) < tolerance:
            raise AssertionError(
                f"{solver_id} cost {result.optimal_cost} differs from {reference}"
            )


def _available_set_oracles() -> list[tuple[str, object]]:
    """Load canonical surrogate checkpoints when their actual artifacts exist."""
    candidates = (
        (
            "bkt_set",
            ROOT / "artifacts" / "bkt_set" / "surrogate_checkpoint.pt",
            BKTSetOracle,
        ),
        (
            "dkt_set",
            ROOT / "artifacts" / "dkt_set" / "surrogate_checkpoint.pt",
            DKTSetOracle,
        ),
    )
    loaded = []
    for name, checkpoint, oracle_cls in candidates:
        if not checkpoint.is_file():
            warnings.warn(f"Skipping {name}: missing checkpoint {checkpoint}")
            continue
        loaded.append((name, oracle_cls.from_artifacts(checkpoint_path=checkpoint)))
    return loaded


def run_delta() -> None:
    manifest = load_manifest()
    initial = frozenset(manifest["initial_state"])
    oracle_factories: list[tuple[str, object]] = [
        ("frozen_monotonic", _frozen(manifest))
    ]
    oracle_factories.extend(_available_set_oracles())
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    structured = []
    for closure in manifest["closures"]:
        graph = _graph(closure)
        nodes = frozenset(closure["nodes"])
        for oracle_type, oracle in oracle_factories:
            p_bar, base = _maps(oracle, nodes, mode="oracle")
            optimum = FullDPSolver(oracle, graph, nodes, initial, p_bar, base).solve(
                set(initial), set(nodes)
            )
            deltas, states = DeltaAnalyzer(graph, oracle, initial, nodes).analyze()
            m_delta = math.fsum(record.delta_v for record in deltas)
            normalized = m_delta / optimum.optimal_cost
            max_pct = max(record.delta_v / record.c_min for record in deltas)
            for record in deltas:
                detail_rows.append(
                    {
                        "target_node": closure["target_node"],
                        "concept": record.concept,
                        "oracle_type": oracle_type,
                        "c_min": record.c_min,
                        "c_max": record.c_max,
                        "delta_v": record.delta_v,
                        "feasible_state_count": record.feasible_state_count,
                    }
                )
            row = {
                "target_node": closure["target_node"],
                "oracle_type": oracle_type,
                "m_delta": m_delta,
                "j_star": optimum.optimal_cost,
                "m_delta_normalized": normalized,
                "max_delta_v": max(record.delta_v for record in deltas),
                "mean_delta_v": statistics.fmean(record.delta_v for record in deltas),
                "max_single_concept_variation_pct": max_pct,
            }
            summary_rows.append(row)
            structured.append({**row, "reachable_ideals": len(states)})
            print(
                f"[B] target={closure['target_node']:>2} oracle={oracle_type:<16} "
                f"m_delta={m_delta:8.3f} j_star={optimum.optimal_cost:8.2f} "
                f"normalized={normalized:.6f}"
            )
    frozen_rows = [
        row for row in summary_rows if row["oracle_type"] == "frozen_monotonic"
    ]
    _write_csv(DELTA_OUTPUT / "delta_per_concept.csv", detail_rows)
    _write_csv(DELTA_OUTPUT / "delta_summary.csv", summary_rows)
    _write_json(
        DELTA_OUTPUT / "delta_summary.json",
        {
            "schema_version": 1,
            "manifest_hash": manifest_hash(manifest),
            "frozen_all_below_one_percent": all(
                row["m_delta_normalized"] < 0.01 for row in frozen_rows
            ),
            "targets": structured,
        },
    )


def _poset_width(graph: nx.DiGraph) -> int:
    closure = nx.transitive_closure_dag(graph)
    left = [("l", node) for node in graph]
    right = [("r", node) for node in graph]
    bipartite = nx.Graph()
    bipartite.add_nodes_from(left, bipartite=0)
    bipartite.add_nodes_from(right, bipartite=1)
    bipartite.add_edges_from(
        (("l", source), ("r", target)) for source, target in closure.edges()
    )
    matching = nx.algorithms.bipartite.maximum_matching(bipartite, top_nodes=left)
    matched = sum(1 for node in left if node in matching)
    return len(graph) - matched


def run_topology() -> None:
    manifest = load_manifest()
    initial = frozenset(manifest["initial_state"])
    rows = []
    for closure in manifest["closures"]:
        graph = _graph(closure)
        n = len(graph)
        reduction = nx.transitive_reduction(graph)
        width = _poset_width(graph)
        depth = nx.dag_longest_path_length(graph) + 1
        oracle = _frozen(manifest)
        nodes = frozenset(graph.nodes())
        p_bar, base = _maps(oracle, nodes)
        dp = FullDPSolver(oracle, graph, nodes, initial, p_bar, base).solve(
            set(initial), set(nodes)
        )
        closure_graph = nx.transitive_closure_dag(graph)
        comparable = {
            frozenset((source, target)) for source, target in closure_graph.edges()
        }
        parallel_pairs = math.comb(n, 2) - len(comparable)
        am_gm = (n / width + 1.0) ** width
        row = {
            "target_node": closure["target_node"],
            "n": n,
            "n_sequence": len(set(closure["nodes"]) - set(initial)),
            "edges": graph.number_of_edges(),
            "edges_transitive_reduction": reduction.number_of_edges(),
            "density": graph.number_of_edges() / (n * (n - 1) / 2) if n > 1 else 0.0,
            "width": width,
            "depth": depth,
            "chain_ratio": depth / n,
            "reachable_ideals": dp.metrics.stored_states,
            "theoretical_max": 2**n,
            "am_gm_bound": am_gm,
            "compression_ratio_exp": dp.metrics.stored_states / 2**n,
            "compression_ratio_amgm": dp.metrics.stored_states / am_gm,
            "max_in_degree": max(dict(graph.in_degree()).values(), default=0),
            "max_out_degree": max(dict(graph.out_degree()).values(), default=0),
            "bottleneck_nodes": json.dumps(
                sorted(node for node, degree in graph.in_degree() if degree >= 3),
                separators=(",", ":"),
            ),
            "parallel_pairs": parallel_pairs,
        }
        rows.append(row)
        print(
            f"[D] target={closure['target_node']:>2} n={n:>2} width={width} "
            f"depth={depth:>2} chain_ratio={depth / n:.2f} "
            f"ideals={dp.metrics.stored_states:>4} "
            f"compression={dp.metrics.stored_states / 2**n:.1e}"
        )
    numeric = [key for key in rows[0] if key not in {"target_node", "bottleneck_nodes"}]
    aggregate = {
        key: {
            "min": min(float(row[key]) for row in rows),
            "max": max(float(row[key]) for row in rows),
            "mean": statistics.fmean(float(row[key]) for row in rows),
        }
        for key in numeric
    }
    _write_csv(TOPOLOGY_OUTPUT / "topology.csv", rows)
    _write_json(
        TOPOLOGY_OUTPUT / "topology.json",
        {
            "schema_version": 1,
            "manifest_hash": manifest_hash(manifest),
            "targets": rows,
            "aggregate": aggregate,
        },
    )


class ParametricOracle:
    """Pure deterministic logistic oracle for synthetic DAGs."""

    def __init__(
        self,
        graph: nx.DiGraph,
        alpha: dict[int, float],
        beta: float,
        base_cost: float = 60.0,
        transfer_weights: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self.graph = graph
        self.alpha = alpha
        self.beta = float(beta)
        self.base_cost_value = float(base_cost)
        self.node_id_to_idx = {node: node for node in graph.nodes()}
        self.transfer_weights = transfer_weights or {
            edge: 1.0 for edge in graph.edges()
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def success_prob(self, v: int, state: frozenset[int]) -> float:
        transfer = math.fsum(
            weight
            for (source, target), weight in self.transfer_weights.items()
            if target == v and source in state
        )
        return self._sigmoid(self.alpha[v] + self.beta * transfer)

    def best_case_success_prob(self, v: int) -> float:
        transfer = math.fsum(
            weight
            for (source, target), weight in self.transfer_weights.items()
            if target == v
        )
        return self._sigmoid(self.alpha[v] + self.beta * transfer)

    def base_cost(self, v: int) -> float:
        return self.base_cost_value


def generate_width_poset(
    n: int, width: int, seed: int, p_edge: float = 0.3
) -> nx.DiGraph:
    rng = np.random.RandomState(seed)
    chains = [list(range(index, n, width)) for index in range(width)]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    positions = {}
    for chain in chains:
        for position, node in enumerate(chain):
            positions[node] = position
        graph.add_edges_from(zip(chain, chain[1:]))
    for first in range(width):
        for second in range(first + 1, width):
            for source in chains[first]:
                for target in chains[second]:
                    if positions[source] < positions[target] and rng.rand() < p_edge:
                        graph.add_edge(source, target)
                    elif positions[target] < positions[source] and rng.rand() < p_edge:
                        graph.add_edge(target, source)
    if not nx.is_directed_acyclic_graph(graph):
        raise AssertionError("Synthetic generator produced a cycle")
    return nx.transitive_reduction(graph)


def _synthetic_oracle(graph: nx.DiGraph, beta: float, seed: int) -> ParametricOracle:
    rng = np.random.RandomState(seed)
    alpha = {node: float(rng.normal(0.0, 0.5)) for node in sorted(graph)}
    return ParametricOracle(graph, alpha, beta)


def _ecs32a_tight(manifest: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    targets = []
    initial = frozenset(manifest["initial_state"])
    for closure in manifest["closures"]:
        graph = _graph(closure)
        nodes = frozenset(graph.nodes())
        oracle = _frozen(manifest)
        p_bar, base = _maps(oracle, nodes, mode="tight")
        results = {
            solver_id: _solver(
                solver_id, _frozen(manifest), graph, nodes, initial, p_bar, base
            ).solve(set(initial), set(nodes))
            for solver_id in SOLVERS
        }
        _relative_cost_check(results)
        for solver_id, result in results.items():
            rows.append(
                {
                    "target_node": closure["target_node"],
                    "solver": solver_id,
                    "optimal_cost": result.optimal_cost,
                    "sequence_length": len(result.optimal_sequence),
                    **asdict(result.metrics),
                }
            )
        targets.append(
            {
                "target_node": closure["target_node"],
                "r_prune": results["dijkstra"].metrics.expanded_states
                / results["astar"].metrics.expanded_states,
                "r_dp": results["full_dp"].metrics.stored_states
                / results["astar"].metrics.expanded_states,
            }
        )
    loose_path = ROOT / "results" / "solver_comparison" / "comparison.csv"
    if loose_path.is_file():
        with loose_path.open("r", encoding="utf-8", newline="") as file:
            loose_rows = list(csv.DictReader(file))
        for target in targets:
            selected = {
                row["solver"]: row
                for row in loose_rows
                if int(row["target_node"]) == target["target_node"]
            }
            if set(selected) >= {"full_dp", "dijkstra", "astar"}:
                loose_astar = int(selected["astar"]["expanded_states"])
                target["loose_r_prune"] = (
                    int(selected["dijkstra"]["expanded_states"]) / loose_astar
                )
                target["loose_r_dp"] = (
                    int(selected["full_dp"]["stored_states"]) / loose_astar
                )
                target["tight_vs_loose_astar_expansion_ratio"] = loose_astar / next(
                    int(row["expanded_states"])
                    for row in rows
                    if row["target_node"] == target["target_node"]
                    and row["solver"] == "astar"
                )
    else:
        warnings.warn(
            "Loose Item 3 comparison is unavailable; tight run will be reported alone"
        )
    _write_csv(SYNTHETIC_OUTPUT / "ecs32a_tight" / "comparison.csv", rows)
    return rows, targets


def _run_synthetic_instance(
    n: int, width: int, beta: float, seed: int
) -> tuple[list[dict], dict]:
    graph = generate_width_poset(n, width, seed)
    nodes = frozenset(graph.nodes())
    initial = frozenset()
    rows = []
    results: dict[str, SolverResult] = {}
    timeout = False
    oracle = _synthetic_oracle(graph, beta, seed)
    tight, base = _maps(oracle, nodes, mode="oracle")
    loose, _ = _maps(oracle, nodes, mode="loose")
    solver_ids = ("full_dp", "dijkstra", "lao") if n <= 25 else ("dijkstra",)
    for solver_id in solver_ids:
        try:
            if solver_id == "full_dp":
                solver = FullDPWithValues(
                    oracle,
                    graph,
                    nodes,
                    initial,
                    tight,
                    base,
                    timeout_seconds=120.0,
                )
            else:
                solver = _solver(solver_id, oracle, graph, nodes, initial, tight, base)
            result = solver.solve(set(), set(nodes))
            results[solver_id] = result
            rows.append(
                {
                    "n": n,
                    "width": width,
                    "beta": beta,
                    "seed": seed,
                    "solver": solver_id,
                    "p_bar_mode": "tight",
                    "optimal_cost": result.optimal_cost,
                    "expanded_states": result.metrics.expanded_states,
                    "generated_states": result.metrics.generated_states,
                    "oracle_calls": result.metrics.oracle_calls,
                    "stored_states": result.metrics.stored_states,
                    "wall_seconds": result.metrics.wall_seconds,
                    "timeout": False,
                }
            )
        except TimeoutError:
            timeout = True
            rows.append(
                {
                    "n": n,
                    "width": width,
                    "beta": beta,
                    "seed": seed,
                    "solver": solver_id,
                    "p_bar_mode": "tight",
                    "optimal_cost": "",
                    "expanded_states": "",
                    "generated_states": "",
                    "oracle_calls": "",
                    "stored_states": "",
                    "wall_seconds": 120.0,
                    "timeout": True,
                }
            )
    astar_results = {}
    for mode, bound in (("loose", loose), ("tight", tight)):
        result = NumericallyStableAStarSolver(
            oracle, graph, nodes, initial, bound, base
        ).solve(set(), set(nodes))
        if result.metrics.reopens:
            raise AssertionError("Synthetic A* reopened a state")
        astar_results[mode] = result
        if mode == "tight":
            results["astar"] = result
        rows.append(
            {
                "n": n,
                "width": width,
                "beta": beta,
                "seed": seed,
                "solver": "astar",
                "p_bar_mode": mode,
                "optimal_cost": result.optimal_cost,
                "expanded_states": result.metrics.expanded_states,
                "generated_states": result.metrics.generated_states,
                "oracle_calls": result.metrics.oracle_calls,
                "stored_states": result.metrics.stored_states,
                "wall_seconds": result.metrics.wall_seconds,
                "timeout": False,
            }
        )
    _relative_cost_check(results)
    if (
        astar_results["loose"].metrics.expanded_states
        < astar_results["tight"].metrics.expanded_states
    ):
        raise AssertionError("Tight A* expanded more states than loose A*")
    summary = {
        "n": n,
        "width": width,
        "actual_width": _poset_width(graph),
        "beta": beta,
        "seed": seed,
        "timeout": timeout,
        "r_prune": results["dijkstra"].metrics.expanded_states
        / astar_results["tight"].metrics.expanded_states,
        "r_dp": (
            results["full_dp"].metrics.stored_states
            / astar_results["tight"].metrics.expanded_states
            if "full_dp" in results
            else None
        ),
        "tight_vs_loose_expansion_ratio": astar_results["loose"].metrics.expanded_states
        / astar_results["tight"].metrics.expanded_states,
    }
    return rows, summary


def _trap_graph(width: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    root, sink = 0, width + 1
    graph.add_nodes_from(range(width + 2))
    graph.add_edges_from((root, sibling) for sibling in range(1, width + 1))
    graph.add_edges_from((sibling, sink) for sibling in range(1, width + 1))
    return graph


def _run_traps() -> list[dict]:
    rows = []
    for width in (3, 5, 7, 10, 15):
        graph = _trap_graph(width)
        nodes = frozenset(graph.nodes())
        alpha = {0: 1.0, width + 1: 0.0}
        alpha.update({node: -0.8 + 0.12 * node for node in range(1, width + 1)})
        transfer = {(node, node + 1): 1.0 for node in range(1, width)}
        oracle = ParametricOracle(graph, alpha, 1.5, transfer_weights=transfer)
        p_bar, base = _maps(oracle, nodes, mode="oracle")
        initial = frozenset()
        results = {
            solver_id: _solver(
                solver_id, oracle, graph, nodes, initial, p_bar, base
            ).solve(set(), set(nodes))
            for solver_id in SOLVERS
        }
        _relative_cost_check(results)
        greedy = GreedyPlanner(
            oracle,
            graph,
            {"planner": {"base_cost": 60.0}},
            edge_index=None,
            num_nodes=width + 2,
        )
        greedy_cost, greedy_path = greedy.solve(set(), set(nodes))
        optimum = results["full_dp"].optimal_cost
        regret = (greedy_cost - optimum) / optimum
        for solver_id, result in results.items():
            rows.append(
                {
                    "trap_width": width,
                    "solver": solver_id,
                    "optimal_cost": result.optimal_cost,
                    "expanded_states": result.metrics.expanded_states,
                    "greedy_cost": greedy_cost,
                    "greedy_regret": regret,
                }
            )
        rows.append(
            {
                "trap_width": width,
                "solver": "greedy",
                "optimal_cost": greedy_cost,
                "expanded_states": len(greedy_path),
                "greedy_cost": greedy_cost,
                "greedy_regret": regret,
            }
        )
    return rows


def run_synthetic() -> None:
    manifest = load_manifest()
    _, ecs_targets = _ecs32a_tight(manifest)
    width_rows = []
    instance_summaries = []
    for n in N_VALUES:
        for width in W_VALUES:
            for beta in BETA_VALUES:
                for seed in SEEDS:
                    rows, summary = _run_synthetic_instance(n, width, beta, seed)
                    width_rows.extend(rows)
                    instance_summaries.append(summary)
                    print(
                        f"[A] n={n:>2} w={width} beta={beta:.1f} seed={seed} "
                        f"R_prune={summary['r_prune']:.2f} "
                        f"R_DP={summary['r_dp'] if summary['r_dp'] is not None else 'NA'}"
                    )
    trap_rows = _run_traps()
    _write_csv(SYNTHETIC_OUTPUT / "width_sweep.csv", width_rows)
    _write_csv(SYNTHETIC_OUTPUT / "trap_sweep.csv", trap_rows)
    grouped = {}
    for n in N_VALUES:
        for width in W_VALUES:
            for beta in BETA_VALUES:
                selected = [
                    row
                    for row in instance_summaries
                    if (row["n"], row["width"], row["beta"]) == (n, width, beta)
                ]
                grouped[f"n={n},w={width},beta={beta}"] = {
                    "mean_r_prune": statistics.fmean(
                        row["r_prune"] for row in selected
                    ),
                    "mean_r_dp": (
                        statistics.fmean(
                            row["r_dp"] for row in selected if row["r_dp"] is not None
                        )
                        if any(row["r_dp"] is not None for row in selected)
                        else None
                    ),
                }
    _write_json(
        SYNTHETIC_OUTPUT / "summary.json",
        {
            "schema_version": 1,
            "manifest_hash": manifest_hash(manifest),
            "ecs32a_tight": ecs_targets,
            "instances": instance_summaries,
            "aggregates": grouped,
            "trap_results": trap_rows,
        },
    )


def _frustration_for(
    oracle: object,
    graph: nx.DiGraph,
    nodes: frozenset[int],
    initial: frozenset[int],
    mode: str,
) -> tuple[dict, SolverResult, SolverResult]:
    p_bar, base = _maps(oracle, nodes, mode=mode)
    dp = FullDPWithValues(oracle, graph, nodes, initial, p_bar, base)
    dp_result = dp.solve(set(initial), set(nodes))
    astar = NumericallyStableAStarSolver(
        oracle, graph, nodes, initial, p_bar, base
    ).solve(set(initial), set(nodes))
    dijkstra = DijkstraSolver(oracle, graph, nodes, initial, p_bar, base).solve(
        set(initial), set(nodes)
    )
    summary = asdict(
        FrustrationAnalyzer(dp.all_values, initial, nodes, p_bar, base).analyze()
    )
    summary["r_prune"] = (
        dijkstra.metrics.expanded_states / astar.metrics.expanded_states
    )
    summary["r_dp"] = dp_result.metrics.stored_states / astar.metrics.expanded_states
    return summary, dp_result, astar


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    numerator = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        math.fsum((x - mean_x) ** 2 for x in xs)
        * math.fsum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator


def run_frustration() -> None:
    manifest = load_manifest()
    initial = frozenset(manifest["initial_state"])
    ecs_rows = []
    for closure in manifest["closures"]:
        graph = _graph(closure)
        nodes = frozenset(graph.nodes())
        oracle = _frozen(manifest)
        by_mode = {}
        for mode in ("loose", "tight"):
            analysis, _, _ = _frustration_for(oracle, graph, nodes, initial, mode)
            by_mode[mode] = analysis
            ecs_rows.append(
                {
                    "target_node": closure["target_node"],
                    "p_bar_mode": mode,
                    **{
                        key: analysis[key]
                        for key in (
                            "rho_s0",
                            "rho_mean",
                            "rho_median",
                            "rho_min",
                            "rho_max",
                            "rho_std",
                            "reachable_states",
                            "r_prune",
                        )
                    },
                }
            )
        print(
            f"[C] target={closure['target_node']:>2} "
            f"rho_s0(loose)={by_mode['loose']['rho_s0']:.3f} "
            f"rho_s0(tight)={by_mode['tight']['rho_s0']:.3f}"
        )
    synthetic_rows = []
    width_path = SYNTHETIC_OUTPUT / "width_sweep.csv"
    if width_path.is_file():
        # Recompute values from the frozen sweep definition; the CSV confirms A ran.
        for n in N_VALUES:
            if n > 25:
                continue
            for width in W_VALUES:
                for beta in BETA_VALUES:
                    for seed in SEEDS:
                        graph = generate_width_poset(n, width, seed)
                        nodes = frozenset(graph.nodes())
                        oracle = _synthetic_oracle(graph, beta, seed)
                        for mode in ("loose", "tight"):
                            analysis, _, _ = _frustration_for(
                                oracle, graph, nodes, frozenset(), mode
                            )
                            synthetic_rows.append(
                                {
                                    "n": n,
                                    "width": width,
                                    "beta": beta,
                                    "seed": seed,
                                    "p_bar_mode": mode,
                                    "rho_s0": analysis["rho_s0"],
                                    "rho_mean": analysis["rho_mean"],
                                    "r_prune": analysis["r_prune"],
                                    "r_dp": analysis["r_dp"],
                                }
                            )
    else:
        warnings.warn(
            "Experiment A outputs unavailable; skipping synthetic frustration"
        )
    _write_csv(FRUSTRATION_OUTPUT / "ecs32a_frustration.csv", ecs_rows)
    if synthetic_rows:
        _write_csv(FRUSTRATION_OUTPUT / "synthetic_frustration.csv", synthetic_rows)
    correlation_rows = synthetic_rows or ecs_rows
    correlation = _pearson(
        [float(row["rho_s0"]) for row in correlation_rows],
        [float(row["r_prune"]) for row in correlation_rows],
    )
    _write_json(
        FRUSTRATION_OUTPUT / "summary.json",
        {
            "schema_version": 1,
            "manifest_hash": manifest_hash(manifest),
            "ecs32a": ecs_rows,
            "synthetic_rows": len(synthetic_rows),
            "pearson_rho_s0_vs_r_prune": correlation,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=("delta", "topology", "synthetic", "frustration", "all"),
    )
    args = parser.parse_args()
    actions = {
        "delta": run_delta,
        "topology": run_topology,
        "synthetic": run_synthetic,
        "frustration": run_frustration,
    }
    selected = (
        actions
        if args.experiment == "all"
        else {args.experiment: actions[args.experiment]}
    )
    for name, action in selected.items():
        print(f"Running {name} experiment...")
        action()


if __name__ == "__main__":
    main()
