"""Generate Linear Syllabus sequences under the shared protocol."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYLLABUS_PATH = ROOT / "data" / "ecs32a_teaching_order_required_full_v1.csv"
OUTPUT = ROOT / "results" / "linear_syllabus_order"
SEQUENCES_PATH = OUTPUT / "sequences.jsonl"
sys.path.insert(0, str(ROOT))

from experiments.common.manifest import (
    DEFAULT_DAG_PATH,
    load_dag,
    load_manifest,
    manifest_hash,
    sha256_file,
)
from experiments.common.schema import Method, SequenceRecord, write_jsonl


def load_teaching_order(path: str | Path = SYLLABUS_PATH) -> dict[int, int]:
    """Load a strict one-to-one node-to-position teaching order."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "node_id" not in rows[0] or "teaching_order" not in rows[0]:
        raise ValueError("Teaching order CSV requires node_id and teaching_order columns")

    order = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            node = int(row["node_id"])
            position = int(row["teaching_order"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid teaching order value at CSV row {row_number}"
            ) from error
        if str(node) != row["node_id"].strip() or str(position) != row[
            "teaching_order"
        ].strip():
            raise ValueError(f"Teaching order IDs must be canonical integers at row {row_number}")
        if node in order:
            raise ValueError(f"Duplicate teaching-order node ID: {node}")
        order[node] = position
    return order


def validate_teaching_order(
    order: dict[int, int],
    nodes: list[int],
    edges: list[tuple[int, int]],
) -> None:
    if len(nodes) != 61 or len(edges) != 134:
        raise ValueError("Shared ECS32A DAG must contain exactly 61 nodes and 134 edges")
    if set(order) != set(nodes):
        missing = sorted(set(nodes) - set(order))
        extra = sorted(set(order) - set(nodes))
        raise ValueError(f"Teaching order node mismatch; missing={missing}, extra={extra}")
    positions = list(order.values())
    if len(positions) != len(set(positions)):
        raise ValueError("Teaching order contains duplicate positions")
    if sorted(positions) != list(range(1, len(nodes) + 1)):
        raise ValueError("Teaching order positions must be exactly 1 through 61")
    violations = [
        (src, dst) for src, dst in edges if order[src] >= order[dst]
    ]
    if violations:
        raise ValueError(f"Teaching order violates prerequisite edges: {violations}")


def generate_records(manifest: dict, order: dict[int, int]) -> list[SequenceRecord]:
    protocol_hash = manifest_hash(manifest)
    evaluator_hash = sha256_file(ROOT / "experiments" / "common" / "evaluator.py")
    syllabus_hash = sha256_file(SYLLABUS_PATH)
    records = []
    for closure in manifest["closures"]:
        sequence = sorted(closure["sequence_nodes"], key=order.__getitem__)
        if set(sequence) != set(closure["sequence_nodes"]):
            raise AssertionError("Syllabus sequence does not cover sequence_nodes")
        if not sequence or sequence[-1] != closure["target_node"]:
            raise AssertionError(
                f"Target {closure['target_node']} is not final in syllabus sequence"
            )
        records.append(
            SequenceRecord(
                method=Method.LINEAR_SYLLABUS,
                target_node=closure["target_node"],
                run_id=0,
                sequence=sequence,
                internal_cost=None,
                metadata={
                    "closure_hash": closure["closure_hash"],
                    "manifest_hash": protocol_hash,
                    "evaluator_hash": evaluator_hash,
                    "teaching_order_hash": syllabus_hash,
                    "teaching_order_source": "data/ecs32a_teaching_order_required_full_v1.csv",
                },
            )
        )
    return records


def main() -> None:
    manifest = load_manifest()
    nodes, edges = load_dag(DEFAULT_DAG_PATH)
    order = load_teaching_order()
    validate_teaching_order(order, nodes, edges)

    first = generate_records(manifest, order)
    second = generate_records(manifest, order)
    if first != second:
        raise AssertionError("Linear Syllabus generation is not deterministic")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(SEQUENCES_PATH, first)
    print(f"teaching_nodes={len(order)}")
    print(f"validated_edges={len(edges)}")
    print(f"records={len(first)}")
    print(f"sequences={SEQUENCES_PATH}")


if __name__ == "__main__":
    main()
