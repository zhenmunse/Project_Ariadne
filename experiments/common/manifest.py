"""Load and materialize the shared experiment manifest.

The JSON file contains only human-edited experiment choices.  This module adds
deterministic prerequisite closures and hashes for the artifacts that define a
run, so every experiment consumes the same frozen protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("manifest.json")
DEFAULT_DAG_PATH = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
DEFAULT_CHECKPOINT_PATH = ROOT / "data" / "processed" / "oracle_ckpt.pt"
DEFAULT_SPLIT_PATHS = (
    ROOT / "data" / "processed" / "train_sessions.pkl",
    ROOT / "data" / "processed" / "valid_sessions.pkl",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_path(path: Path) -> str:
    """Return a stable repo-relative path where possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def artifact_collection(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Describe and hash a collection, including explicitly missing files."""
    artifacts = []
    for path in sorted((Path(path) for path in paths), key=_protocol_path):
        exists = path.is_file()
        artifacts.append(
            {
                "path": _protocol_path(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    if not artifacts:
        raise ValueError("At least one split artifact path must be specified")
    return {
        "combined_hash": hashlib.sha256(_canonical_json(artifacts)).hexdigest(),
        "artifacts": artifacts,
    }


def _require_node_id(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer node ID")
    return value


def _load_dag(path: Path) -> tuple[list[int], list[tuple[int, int]]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("DAG JSON must contain 'nodes' and 'edges' lists")

    nodes = sorted(
        _require_node_id(
            node["node_id"] if isinstance(node, dict) else node,
            "nodes[].node_id",
        )
        for node in raw_nodes
    )
    edges = sorted(
        (
            _require_node_id(
                edge["src"] if isinstance(edge, dict) else edge[0], "edges[].src"
            ),
            _require_node_id(
                edge["dst"] if isinstance(edge, dict) else edge[1], "edges[].dst"
            ),
        )
        for edge in raw_edges
    )
    if len(nodes) != len(set(nodes)):
        raise ValueError("DAG contains duplicate node IDs")
    if len(edges) != len(set(edges)):
        raise ValueError("DAG contains duplicate edges")
    if any(src not in nodes or dst not in nodes for src, dst in edges):
        raise ValueError("DAG edge references an unknown node")
    return nodes, edges


def _topological_order(nodes: list[int], edges: list[tuple[int, int]]) -> list[int]:
    successors = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        successors[src].append(dst)
        indegree[dst] += 1

    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in sorted(successors[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("Prerequisite graph must be a DAG")
    return order


def _closure(
    target: int, nodes: list[int], edges: list[tuple[int, int]]
) -> dict[str, Any]:
    predecessors = {node: [] for node in nodes}
    for src, dst in edges:
        predecessors[dst].append(src)

    closure_nodes = {target}
    frontier = [target]
    while frontier:
        node = frontier.pop()
        for predecessor in predecessors[node]:
            if predecessor not in closure_nodes:
                closure_nodes.add(predecessor)
                frontier.append(predecessor)

    node_list = sorted(closure_nodes)
    edge_list = [[src, dst] for src, dst in edges if src in closure_nodes and dst in closure_nodes]
    sinks = sorted(set(node_list) - {src for src, _ in edge_list})
    if sinks != [target]:
        raise ValueError(
            f"Target {target} must be the unique sink of its closure; found {sinks}"
        )

    closure_content = {
        "target_node": target,
        "nodes": node_list,
        "edges": edge_list,
    }
    return {
        **closure_content,
        "closure_hash": hashlib.sha256(_canonical_json(closure_content)).hexdigest(),
    }


def load_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    dag_path: str | Path = DEFAULT_DAG_PATH,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    split_paths: Iterable[str | Path] = DEFAULT_SPLIT_PATHS,
) -> dict[str, Any]:
    """Load, validate, and deterministically materialize the shared protocol."""
    manifest_path = Path(manifest_path)
    dag_path = Path(dag_path)
    checkpoint_path = Path(checkpoint_path)

    with manifest_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    required = {"seed", "targets", "initial_state", "base_cost"}
    if set(raw) != required:
        raise ValueError(f"Manifest fields must be exactly {sorted(required)}")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise TypeError("seed must be an integer")
    if (
        isinstance(raw["base_cost"], bool)
        or not isinstance(raw["base_cost"], (int, float))
        or raw["base_cost"] <= 0
    ):
        raise ValueError("base_cost must be positive")
    if not isinstance(raw["targets"], list) or not raw["targets"]:
        raise ValueError("targets must be a non-empty list")
    if not isinstance(raw["initial_state"], list):
        raise TypeError("initial_state must be a list")
    if any(not isinstance(node, int) or isinstance(node, bool) for node in raw["targets"]):
        raise TypeError("every target must be an integer node ID")
    if any(
        not isinstance(node, int) or isinstance(node, bool)
        for node in raw["initial_state"]
    ):
        raise TypeError("every initial_state entry must be an integer node ID")

    targets = list(raw["targets"])
    initial_state = list(raw["initial_state"])
    if len(targets) != len(set(targets)):
        raise ValueError("targets must not contain duplicates")
    if len(initial_state) != len(set(initial_state)):
        raise ValueError("initial_state must not contain duplicates")

    nodes, edges = _load_dag(dag_path)
    _topological_order(nodes, edges)
    unknown = sorted((set(targets) | set(initial_state)) - set(nodes))
    if unknown:
        raise ValueError(f"Manifest references unknown DAG nodes: {unknown}")

    closures = [_closure(target, nodes, edges) for target in targets]
    initial_set = set(initial_state)
    missing_prerequisites = sorted(
        (src, dst) for src, dst in edges if dst in initial_set and src not in initial_set
    )
    if missing_prerequisites:
        raise ValueError(
            "initial_state must be prerequisite-closed; missing prerequisite edges: "
            f"{missing_prerequisites}"
        )
    for closure in closures:
        outside = sorted(initial_set - set(closure["nodes"]))
        if outside:
            raise ValueError(
                f"initial_state must be a subset of target {closure['target_node']} "
                f"closure; outside nodes: {outside}"
            )
        closure["sequence_nodes"] = [
            node for node in closure["nodes"] if node not in initial_set
        ]
    return {
        "seed": raw["seed"],
        "targets": targets,
        "initial_state": initial_state,
        "base_cost": float(raw["base_cost"]),
        "closures": closures,
        "artifact_hashes": {
            "dag": sha256_file(dag_path),
            "oracle_checkpoint": sha256_file(checkpoint_path),
            "train_validation_split": artifact_collection(split_paths),
        },
    }


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Return the canonical hash of a fully materialized manifest."""
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


if __name__ == "__main__":
    materialized = load_manifest()
    print(json.dumps(materialized, indent=2, sort_keys=True))
    print(f"manifest_hash: {manifest_hash(materialized)}")
