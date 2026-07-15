"""Compare exact lattice solvers on the ten frozen ECS32A closures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "solver_comparison"
sys.path.insert(0, str(ROOT))

from experiments.common.frozen_oracle import FrozenMonotonicOracle  # noqa: E402
from experiments.common.manifest import (  # noqa: E402
    load_manifest,
    manifest_hash,
    sha256_file,
)
from src.planner_engine.solver_comparison import (  # noqa: E402
    AStarSolver,
    DijkstraSolver,
    FullDPSolver,
    LAOStarWrapper,
    SearchMetrics,
    SolverResult,
)
from src.planner_engine.zpd_utils import get_valid_actions  # noqa: E402


SOLVERS = ("full_dp", "dijkstra", "astar", "lao")
DETERMINISTIC_METRICS = tuple(
    field.name for field in fields(SearchMetrics) if field.name != "wall_seconds"
)


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def _make_solver(
    solver_id: str,
    oracle: object,
    graph: nx.DiGraph,
    target: frozenset[int],
    initial: frozenset[int],
    p_bar: dict[int, float],
    base_cost_map: dict[int, float],
):
    solver_class = {
        "full_dp": FullDPSolver,
        "dijkstra": DijkstraSolver,
        "astar": AStarSolver,
        "lao": LAOStarWrapper,
    }[solver_id]
    return solver_class(oracle, graph, target, initial, p_bar, base_cost_map)


def _assert_valid_sequence(
    sequence: list[int],
    graph: nx.DiGraph,
    initial: frozenset[int],
    target: frozenset[int],
) -> None:
    expected = target - initial
    if len(sequence) != len(expected) or set(sequence) != expected:
        raise AssertionError(
            f"Sequence does not cover target exactly: sequence={sequence}, "
            f"expected={sorted(expected)}"
        )
    state = set(initial)
    for position, action in enumerate(sequence):
        if action not in get_valid_actions(graph, state):
            raise AssertionError(
                f"Invalid action {action} at position {position}; state={sorted(state)}"
            )
        state.add(action)
    if not target <= state:
        raise AssertionError("Sequence did not reach the target state")


def _assert_agreement(
    target_node: int | str,
    results: dict[str, SolverResult],
    graph: nx.DiGraph,
    initial: frozenset[int],
    target: frozenset[int],
) -> None:
    reference = results["full_dp"]
    for solver_id, result in results.items():
        gap = abs(result.optimal_cost - reference.optimal_cost)
        tolerance = 1e-12 * max(1.0, abs(reference.optimal_cost))
        if not gap < tolerance:
            raise AssertionError(
                f"Cost mismatch for target {target_node}: full_dp="
                f"{reference.optimal_cost!r}, {solver_id}={result.optimal_cost!r}, "
                f"gap={gap!r}, tolerance={tolerance!r}"
            )
        if result.optimal_sequence != reference.optimal_sequence:
            raise AssertionError(
                f"Sequence mismatch for target {target_node}: full_dp="
                f"{reference.optimal_sequence}, {solver_id}={result.optimal_sequence}"
            )
        _assert_valid_sequence(result.optimal_sequence, graph, initial, target)
    if results["astar"].metrics.reopens != 0:
        raise AssertionError(f"A* reopened states for target {target_node}")


class _SmokeOracle:
    """Small deterministic, state-dependent oracle used only by --smoke."""

    def __init__(self, nodes: Iterable[int]) -> None:
        nodes = sorted(nodes)
        self.node_id_to_idx = {node: index for index, node in enumerate(nodes)}

    @staticmethod
    def base_cost(v: int) -> float:
        return 60.0

    @staticmethod
    def best_case_success_prob(v: int) -> float:
        return 1.0

    @staticmethod
    def success_prob(v: int, state: frozenset[int]) -> float:
        # A deliberately conservative bound (p_bar=1) and action-dependent
        # probabilities make optimal sequences unique in branching smoke DAGs.
        return min(0.95, 0.42 + 0.055 * len(state) + 0.031 * v)


def _solve_smoke_graph(graph: nx.DiGraph) -> dict[str, SolverResult]:
    oracle = _SmokeOracle(graph.nodes())
    initial = frozenset()
    target = frozenset(graph.nodes())
    p_bar = {node: oracle.best_case_success_prob(node) for node in target}
    base_cost_map = {node: oracle.base_cost(node) for node in target}
    results = {
        solver_id: _make_solver(
            solver_id, oracle, graph, target, initial, p_bar, base_cost_map
        ).solve(set(initial), set(target))
        for solver_id in SOLVERS
    }
    _assert_agreement("smoke", results, graph, initial, target)
    return results


def run_smoke_tests() -> None:
    handmade = nx.DiGraph([(0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    handmade_results = _solve_smoke_graph(handmade)
    if handmade_results["full_dp"].metrics.stored_states != 9:
        raise AssertionError("Handmade DAG must have exactly nine reachable ideals")

    chain = nx.DiGraph()
    chain.add_nodes_from(range(6))
    chain.add_edges_from((node, node + 1) for node in range(5))
    chain_results = _solve_smoke_graph(chain)
    for result in chain_results.values():
        if result.optimal_sequence != list(range(6)):
            raise AssertionError("Every chain solver must return the unique sequence")
    if chain_results["full_dp"].metrics.stored_states != 7:
        raise AssertionError("Length-six chain must have seven reachable ideals")
    if chain_results["dijkstra"].metrics.expanded_states != 7:
        raise AssertionError("Dijkstra must expand all seven chain states")
    if chain_results["astar"].metrics.expanded_states != 7:
        raise AssertionError("A* must expand all seven chain states")

    antichain = nx.DiGraph()
    antichain.add_nodes_from(range(5))
    antichain_results = _solve_smoke_graph(antichain)
    if antichain_results["full_dp"].metrics.stored_states != 2**5:
        raise AssertionError("Five-node antichain must have 32 reachable ideals")
    print("Smoke checks passed: handmade DAG, chain DAG, and antichain DAG.")


def _frozen_oracle(manifest: dict) -> FrozenMonotonicOracle:
    return FrozenMonotonicOracle.from_artifacts(
        base_cost=manifest["base_cost"], device="cpu"
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _ratio_block(
    results: dict[str, SolverResult], metric: str
) -> dict[str, float | None]:
    astar_value = float(getattr(results["astar"].metrics, metric))
    dp_value = (
        float(results["full_dp"].metrics.stored_states)
        if metric == "expanded_states"
        else float(getattr(results["full_dp"].metrics, metric))
    )
    return {
        "r_prune": _safe_ratio(
            float(getattr(results["dijkstra"].metrics, metric)), astar_value
        ),
        "r_dp": _safe_ratio(dp_value, astar_value),
        "r_lao_astar": _safe_ratio(
            float(getattr(results["lao"].metrics, metric)), astar_value
        ),
    }


def _run_once(manifest: dict) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    targets: list[dict] = []
    initial = frozenset(manifest["initial_state"])
    for closure in manifest["closures"]:
        target_node = int(closure["target_node"])
        graph = _closure_graph(closure)
        target = frozenset(closure["nodes"])

        precompute_oracle = _frozen_oracle(manifest)
        p_bar = {
            node: float(precompute_oracle.best_case_success_prob(node))
            for node in target
        }
        base_cost_map = {
            node: float(precompute_oracle.base_cost(node)) for node in target
        }
        results: dict[str, SolverResult] = {}
        for solver_id in SOLVERS:
            oracle = _frozen_oracle(manifest)
            solver = _make_solver(
                solver_id, oracle, graph, target, initial, p_bar, base_cost_map
            )
            results[solver_id] = solver.solve(set(initial), set(target))
        _assert_agreement(target_node, results, graph, initial, target)

        for solver_id in SOLVERS:
            result = results[solver_id]
            rows.append(
                {
                    "target_node": target_node,
                    "solver": solver_id,
                    "optimal_cost": result.optimal_cost,
                    "sequence_length": len(result.optimal_sequence),
                    **asdict(result.metrics),
                }
            )
        targets.append(
            {
                "target_node": target_node,
                "closure_nodes": len(target),
                "optimal_cost": results["full_dp"].optimal_cost,
                "optimal_sequence": results["full_dp"].optimal_sequence,
                "max_absolute_cost_gap": max(
                    abs(result.optimal_cost - results["full_dp"].optimal_cost)
                    for result in results.values()
                ),
                "cost_agreement": True,
                "sequence_agreement": True,
                "ratios": {
                    metric: _ratio_block(results, metric)
                    for metric in (
                        "expanded_states",
                        "generated_states",
                        "oracle_calls",
                        "wall_seconds",
                    )
                },
            }
        )
        print(
            f"target={target_node:>2} nodes={len(target):>2} "
            f"cost={results['full_dp'].optimal_cost:.12f} "
            f"expanded(dp/ucs/a*/lao)="
            f"{results['full_dp'].metrics.expanded_states}/"
            f"{results['dijkstra'].metrics.expanded_states}/"
            f"{results['astar'].metrics.expanded_states}/"
            f"{results['lao'].metrics.expanded_states}"
        )
    return rows, targets


def _deterministic_signature(rows: list[dict]) -> list[dict]:
    keys = (
        "target_node",
        "solver",
        "optimal_cost",
        "sequence_length",
        *DETERMINISTIC_METRICS,
    )
    return [{key: row[key] for key in keys} for row in rows]


def _aggregate(rows: list[dict], targets: list[dict]) -> dict:
    solver_stats = {}
    for solver_id in SOLVERS:
        selected = [row for row in rows if row["solver"] == solver_id]
        solver_stats[solver_id] = {
            metric: {
                "mean": statistics.fmean(float(row[metric]) for row in selected),
                "min": min(float(row[metric]) for row in selected),
                "max": max(float(row[metric]) for row in selected),
            }
            for metric in (
                "expanded_states",
                "generated_states",
                "oracle_calls",
                "edge_cost_evaluations",
                "stored_states",
                "wall_seconds",
            )
        }
    ratio_stats = {}
    for metric in (
        "expanded_states",
        "generated_states",
        "oracle_calls",
        "wall_seconds",
    ):
        ratio_stats[metric] = {}
        for ratio_name in ("r_prune", "r_dp", "r_lao_astar"):
            values = [
                target["ratios"][metric][ratio_name]
                for target in targets
                if target["ratios"][metric][ratio_name] is not None
            ]
            ratio_stats[metric][ratio_name] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
    return {"solver_metrics": solver_stats, "ratios": ratio_stats}


def _write_outputs(rows: list[dict], summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "comparison.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "summary.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the three smoke checks before the full frozen comparison",
    )
    args = parser.parse_args()
    if args.smoke:
        run_smoke_tests()

    manifest = load_manifest()
    print("Running frozen comparison pass 1/2...")
    first_rows, first_targets = _run_once(manifest)
    print("Running frozen comparison pass 2/2 with independent Oracle objects...")
    second_rows, _ = _run_once(manifest)
    if _deterministic_signature(first_rows) != _deterministic_signature(second_rows):
        raise AssertionError(
            "Independent runs differ in costs or deterministic search metrics"
        )

    summary = {
        "schema_version": 1,
        "manifest_hash": manifest_hash(manifest),
        "solver_comparison_source_hash": sha256_file(
            ROOT / "src" / "planner_engine" / "solver_comparison.py"
        ),
        "runner_source_hash": sha256_file(Path(__file__)),
        "targets": first_targets,
        "aggregate": _aggregate(first_rows, first_targets),
        "invariants": {
            "targets_checked": len(first_targets),
            "cost_agreement": True,
            "sequence_agreement": True,
            "sequence_validity": True,
            "astar_reopens_zero": True,
            "independent_runs": 2,
            "deterministic_metrics_match": True,
            "wall_seconds_excluded_from_determinism_check": True,
        },
    }
    _write_outputs(first_rows, summary)
    print(f"Wrote {OUTPUT / 'comparison.csv'}")
    print(f"Wrote {OUTPUT / 'summary.json'}")


if __name__ == "__main__":
    main()
