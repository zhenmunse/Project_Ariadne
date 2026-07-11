"""Canonical sequence validation and scoring under the frozen evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import networkx as nx

from experiments.common.frozen_oracle import FrozenMonotonicOracle
from experiments.common.manifest import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DAG_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SPLIT_PATHS,
    load_manifest,
)
from experiments.common.schema import SequenceRecord
from src.planner_engine.solver import DAGPlannerDP


REGRET_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ScoredSequence:
    method: str
    target_node: int
    run_id: int
    valid: bool
    evaluation_cost: float | None
    optimal_cost: float | None
    normalized_regret: float | None
    sequence_hash: str
    invalid_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sequence_hash(sequence: Iterable[int]) -> str:
    encoded = json.dumps(
        list(sequence), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SequenceEvaluator:
    """Validate and score records using one canonical CPU Oracle."""

    def __init__(
        self,
        manifest: dict,
        oracle: FrozenMonotonicOracle,
    ) -> None:
        if oracle.device.type != "cpu":
            raise ValueError("The canonical evaluator requires a CPU Oracle")
        self.manifest = manifest
        self.oracle = oracle
        self.initial_state = frozenset(manifest["initial_state"])
        self.closures = {
            closure["target_node"]: closure for closure in manifest["closures"]
        }
        self._optimal: dict[int, tuple[float, tuple[int, ...]]] = {}

    @classmethod
    def from_artifacts(
        cls,
        manifest_path=DEFAULT_MANIFEST_PATH,
        *,
        dag_path=DEFAULT_DAG_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        split_paths=DEFAULT_SPLIT_PATHS,
    ) -> "SequenceEvaluator":
        manifest = load_manifest(
            manifest_path,
            dag_path=dag_path,
            checkpoint_path=checkpoint_path,
            split_paths=split_paths,
        )
        oracle = FrozenMonotonicOracle.from_artifacts(
            checkpoint_path=checkpoint_path,
            dag_path=dag_path,
            base_cost=manifest["base_cost"],
            device="cpu",
        )
        return cls(manifest, oracle)

    def _closure_graph(self, target_node: int) -> nx.DiGraph:
        closure = self.closures[target_node]
        graph = nx.DiGraph()
        graph.add_nodes_from(closure["nodes"])
        graph.add_edges_from(tuple(edge) for edge in closure["edges"])
        return graph

    def exact_optimum(self, target_node: int) -> tuple[float, tuple[int, ...]]:
        """Return exact cost and deterministic tie-broken sequence for a target."""
        if target_node not in self.closures:
            raise ValueError(f"Target {target_node} is not in the shared manifest")
        if target_node in self._optimal:
            return self._optimal[target_node]

        graph = self._closure_graph(target_node)
        planner = DAGPlannerDP(
            oracle=self.oracle,
            nx_graph=graph,
            config={"planner": {"base_cost": self.manifest["base_cost"]}},
            edge_index=self.oracle.edge_index,
            num_nodes=self.oracle.model.num_nodes,
        )
        cost, path = planner.solve(
            current_state=set(self.initial_state),
            target_lo_nodes=set(self.closures[target_node]["nodes"]),
        )
        if not math.isfinite(cost):
            raise RuntimeError(f"Exact DP found no finite plan for target {target_node}")
        result = (float(cost), tuple(path))
        self._optimal[target_node] = result
        return result

    def _invalid_reason(self, record: SequenceRecord) -> str | None:
        closure = self.closures.get(record.target_node)
        if closure is None:
            return "target_not_in_manifest"

        closure_nodes = set(closure["nodes"])
        outside = sorted(set(record.sequence) - closure_nodes)
        if outside:
            return f"sequence_contains_nodes_outside_closure:{outside}"

        expected = set(closure["sequence_nodes"])
        actual = set(record.sequence)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            return f"sequence_missing_required_nodes:{missing}"
        if extra:
            return f"sequence_contains_already_mastered_nodes:{extra}"

        if record.target_node in self.initial_state:
            return "target_already_mastered_semantics_not_defined"
        if not record.sequence or record.sequence[-1] != record.target_node:
            return "target_must_be_final_sequence_node"

        state = set(self.initial_state)
        graph = self._closure_graph(record.target_node)
        for position, node in enumerate(record.sequence):
            unmet = sorted(set(graph.predecessors(node)) - state)
            if unmet:
                return (
                    f"prerequisites_not_mastered:position={position},"
                    f"node={node},unmet={unmet}"
                )
            state.add(node)
        return None

    def evaluation_cost(self, record: SequenceRecord) -> float:
        """Score an already validated sequence under the frozen evaluator."""
        state = set(self.initial_state)
        terms = []
        for node in record.sequence:
            probability = self.oracle.success_prob(node, state)
            terms.append(self.oracle.base_cost(node) / probability)
            state.add(node)
        return math.fsum(terms)

    def score(self, record: SequenceRecord) -> ScoredSequence:
        invalid_reason = self._invalid_reason(record)
        digest = sequence_hash(record.sequence)

        if record.target_node not in self.closures:
            return ScoredSequence(
                method=record.method.value,
                target_node=record.target_node,
                run_id=record.run_id,
                valid=False,
                evaluation_cost=None,
                optimal_cost=None,
                normalized_regret=None,
                sequence_hash=digest,
                invalid_reason=invalid_reason,
            )

        optimal_cost, _ = self.exact_optimum(record.target_node)
        if invalid_reason is not None:
            return ScoredSequence(
                method=record.method.value,
                target_node=record.target_node,
                run_id=record.run_id,
                valid=False,
                evaluation_cost=None,
                optimal_cost=optimal_cost,
                normalized_regret=None,
                sequence_hash=digest,
                invalid_reason=invalid_reason,
            )

        cost = self.evaluation_cost(record)
        difference = cost - optimal_cost
        if abs(difference) <= REGRET_TOLERANCE * max(1.0, optimal_cost):
            regret = 0.0
        else:
            regret = difference / optimal_cost
        if regret < -1e-9:
            raise AssertionError(
                f"Negative regret for target {record.target_node}: {regret}"
            )
        return ScoredSequence(
            method=record.method.value,
            target_node=record.target_node,
            run_id=record.run_id,
            valid=True,
            evaluation_cost=cost,
            optimal_cost=optimal_cost,
            normalized_regret=regret,
            sequence_hash=digest,
            invalid_reason=None,
        )

    def score_records(self, records: Iterable[SequenceRecord]) -> list[ScoredSequence]:
        """Score records after enforcing file-level experiment identity uniqueness."""
        records = list(records)
        seen: set[tuple[str, int, int]] = set()
        for record in records:
            identity = (record.method.value, record.target_node, record.run_id)
            if identity in seen:
                raise ValueError(
                    "Duplicate (method, target_node, run_id) identity: "
                    f"{identity}"
                )
            seen.add(identity)
        return [self.score(record) for record in records]
