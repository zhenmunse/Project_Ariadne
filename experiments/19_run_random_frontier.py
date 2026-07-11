"""Generate Random Frontier Policy sequences under the shared protocol."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "random_frontier"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
RUNS_PER_TARGET = 100
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import load_manifest, manifest_hash, sha256_file
from experiments.common.schema import Method, SequenceRecord, write_jsonl


def _closure_graph(closure: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(closure["nodes"])
    graph.add_edges_from(tuple(edge) for edge in closure["edges"])
    return graph


def random_frontier_sequence(
    closure: dict,
    initial_state: set[int],
    seed: int,
) -> list[int]:
    """Sample uniformly from currently available actions at every step."""
    graph = _closure_graph(closure)
    mastered = set(initial_state)
    remaining = set(closure["sequence_nodes"])
    rng = random.Random(seed)
    sequence = []
    while remaining:
        frontier = sorted(
            node
            for node in remaining
            if set(graph.predecessors(node)) <= mastered
        )
        if not frontier:
            raise RuntimeError(
                f"No valid frontier action for target {closure['target_node']}"
            )
        node = rng.choice(frontier)
        sequence.append(node)
        mastered.add(node)
        remaining.remove(node)
    return sequence


def generate_records(manifest: dict) -> list[SequenceRecord]:
    protocol_hash = manifest_hash(manifest)
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")
    initial_state = set(manifest["initial_state"])
    records = []
    for target_index, closure in enumerate(manifest["closures"]):
        for run_id in range(RUNS_PER_TARGET):
            seed = manifest["seed"] + target_index * 1000 + run_id
            sequence = random_frontier_sequence(closure, initial_state, seed)
            records.append(
                SequenceRecord(
                    method=Method.RANDOM_FRONTIER,
                    target_node=closure["target_node"],
                    run_id=run_id,
                    sequence=sequence,
                    internal_cost=None,
                    metadata={
                        "seed": seed,
                        "closure_hash": closure["closure_hash"],
                        "manifest_hash": protocol_hash,
                        "evaluator_hash": evaluator_hash,
                        "policy": "uniform_available_action",
                    },
                )
            )
    return records


def main() -> None:
    manifest = load_manifest()
    first = generate_records(manifest)
    second = generate_records(manifest)
    if first != second:
        raise AssertionError("Random Frontier generation is not reproducible")
    if len(first) != len(manifest["targets"]) * RUNS_PER_TARGET:
        raise AssertionError("Unexpected Random Frontier record count")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first)
    print(f"policy=Random Frontier Policy")
    print(f"targets={len(manifest['targets'])}")
    print(f"runs_per_target={RUNS_PER_TARGET}")
    print(f"records={len(first)}")
    print(f"sequences={SEQUENCES_PATH}")


if __name__ == "__main__":
    main()
