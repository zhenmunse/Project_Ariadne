"""Compress chronological concept sessions into legal SSP mastery states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ConceptSession:
    """The fields needed to compute mastery from one canonical session."""

    target_node: int
    session_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.target_node, int) or isinstance(self.target_node, bool):
            raise TypeError("target_node must be an integer node ID")
        if isinstance(self.session_score, bool) or not isinstance(
            self.session_score, (int, float)
        ):
            raise TypeError("session_score must be numeric")
        if not 0.0 <= float(self.session_score) <= 1.0:
            raise ValueError("session_score must be in [0, 1]")


def ancestor_map(
    nodes: Iterable[int], edges: Iterable[tuple[int, int]]
) -> dict[int, frozenset[int]]:
    """Return the transitive ancestor set of every node in a validated DAG."""
    node_set = set(nodes)
    predecessors = {node: set() for node in node_set}
    for src, dst in edges:
        if src not in node_set or dst not in node_set:
            raise ValueError("DAG edge references an unknown node")
        predecessors[dst].add(src)

    visiting: set[int] = set()
    completed: dict[int, frozenset[int]] = {}

    def visit(node: int) -> frozenset[int]:
        if node in completed:
            return completed[node]
        if node in visiting:
            raise ValueError("Prerequisite graph must be acyclic")
        visiting.add(node)
        ancestors = set(predecessors[node])
        for predecessor in predecessors[node]:
            ancestors.update(visit(predecessor))
        visiting.remove(node)
        completed[node] = frozenset(ancestors)
        return completed[node]

    for node in sorted(node_set):
        visit(node)
    return completed


def raw_mastery_state_before_prefix(
    sessions: Sequence[ConceptSession],
    prefix_end: int,
    *,
    threshold: float = 0.8,
    consecutive: int = 3,
) -> frozenset[int]:
    """Return irreversible raw mastery using sessions strictly before prefix_end."""
    if not isinstance(prefix_end, int) or isinstance(prefix_end, bool):
        raise TypeError("prefix_end must be an integer")
    if not 0 <= prefix_end <= len(sessions):
        raise ValueError("prefix_end must be between zero and len(sessions)")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not isinstance(consecutive, int) or isinstance(consecutive, bool):
        raise TypeError("consecutive must be an integer")
    if consecutive <= 0:
        raise ValueError("consecutive must be positive")

    streaks: dict[int, int] = {}
    mastered: set[int] = set()
    for session in sessions[:prefix_end]:
        node = session.target_node
        if float(session.session_score) >= float(threshold):
            streaks[node] = streaks.get(node, 0) + 1
            if streaks[node] >= consecutive:
                mastered.add(node)
        else:
            streaks[node] = 0
    return frozenset(mastered)


@dataclass(frozen=True)
class MasteryCompletion:
    """Auditable components of zero-observation prerequisite completion."""

    raw_mastery: frozenset[int]
    retained_mastery: frozenset[int]
    completed_ancestors: frozenset[int]
    state: frozenset[int]


def zero_observation_nodes(
    nodes: Iterable[int], training_observed_nodes: Iterable[int]
) -> frozenset[int]:
    """Return full-DAG nodes with no canonical training observation."""
    node_set = frozenset(nodes)
    observed = frozenset(training_observed_nodes)
    unknown = sorted(observed - node_set)
    if unknown:
        raise ValueError(f"training observations contain unknown nodes: {unknown}")
    return node_set - observed


def zero_observation_prerequisite_completion(
    raw_mastered: Iterable[int],
    ancestors: Mapping[int, frozenset[int]],
    training_observed_nodes: Iterable[int],
) -> MasteryCompletion:
    """Complete only globally training-unobserved prerequisites."""
    raw = frozenset(raw_mastered)
    nodes = frozenset(ancestors)
    unknown_raw = sorted(raw - nodes)
    if unknown_raw:
        raise ValueError(f"raw mastery contains unknown nodes: {unknown_raw}")
    observed = frozenset(training_observed_nodes)
    zero_nodes = zero_observation_nodes(nodes, observed)
    retained = frozenset(
        node for node in raw if (ancestors[node] & observed).issubset(raw)
    )
    completed = frozenset().union(
        *(ancestors[node] & zero_nodes for node in retained)
    ) if retained else frozenset()
    state = retained | completed
    if any(not ancestors[node].issubset(state) for node in state):
        raise AssertionError("completed mastery state is not prerequisite-closed")
    return MasteryCompletion(
        raw_mastery=raw,
        retained_mastery=retained,
        completed_ancestors=completed - retained,
        state=state,
    )


def mastery_completion_before_prefix(
    sessions: Sequence[ConceptSession],
    prefix_end: int,
    *,
    ancestors: Mapping[int, frozenset[int]],
    training_observed_nodes: Iterable[int],
    threshold: float = 0.8,
    consecutive: int = 3,
) -> MasteryCompletion:
    """Return raw and completed mastery components before one prefix."""
    raw = raw_mastery_state_before_prefix(
        sessions,
        prefix_end,
        threshold=threshold,
        consecutive=consecutive,
    )
    return zero_observation_prerequisite_completion(
        raw, ancestors, training_observed_nodes
    )


def mastery_state_before_prefix(
    sessions: Sequence[ConceptSession],
    prefix_end: int,
    *,
    ancestors: Mapping[int, frozenset[int]],
    training_observed_nodes: Iterable[int],
    threshold: float = 0.8,
    consecutive: int = 3,
) -> frozenset[int]:
    """Return the canonical prerequisite-closed state before one prefix."""
    completion = mastery_completion_before_prefix(
        sessions,
        prefix_end,
        ancestors=ancestors,
        training_observed_nodes=training_observed_nodes,
        threshold=threshold,
        consecutive=consecutive,
    )
    return completion.state


def canonical_mastery_tuple(state: Iterable[int]) -> tuple[int, ...]:
    """Return the stable serialized representation of a mastery state."""
    return tuple(sorted(state))

