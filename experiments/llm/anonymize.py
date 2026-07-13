"""Deterministic per-target opaque mappings and presentation orders."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any

from experiments.common.manifest import load_manifest, manifest_hash
from experiments.llm.artifacts import load_json, sha256_file, value_hash


ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = ROOT / "data" / "ecs32a_dag_required_full_v1.json"
PROTOCOL_PATH = ROOT / "experiments" / "llm" / "protocol.json"

_ORDER_PREFIX = re.compile(
    r"^(?:(?:unit|week|lesson|lecture)\s*\d+\s*[:_\-]\s*|u\d+\s*[-_:]\s*|\d+\s*[_-]\s*)+",
    flags=re.IGNORECASE,
)


def clean_concept_name(name: str) -> str:
    cleaned = _ORDER_PREFIX.sub("", name.strip())
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Concept name is empty after order-prefix cleaning")
    return cleaned


def _seed_digest(protocol_hash: str, materialized_hash: str, target: int, label: str) -> str:
    payload = f"{materialized_hash}{protocol_hash}{target}{label}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rng(digest: str) -> random.Random:
    return random.Random(int(digest, 16))


def build_mappings() -> dict[str, Any]:
    manifest = load_manifest()
    dag = load_json(DAG_PATH)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    materialized_hash = manifest_hash(manifest)
    names = {int(node["node_id"]): clean_concept_name(node["concept_name"]) for node in dag["nodes"]}
    bundles = []
    for closure in manifest["closures"]:
        target = int(closure["target_node"])
        nodes = sorted(int(node) for node in closure["nodes"])
        mapping_seed = _seed_digest(protocol_hash, materialized_hash, target, "opaque-mapping-v1")
        shuffled_nodes = list(nodes)
        _rng(mapping_seed).shuffle(shuffled_nodes)
        real_to_opaque = {str(node): f"C{index:02d}" for index, node in enumerate(shuffled_nodes, 1)}
        opaque_to_real = {opaque: int(node) for node, opaque in real_to_opaque.items()}

        concept_order = [real_to_opaque[str(node)] for node in nodes]
        _rng(_seed_digest(protocol_hash, materialized_hash, target, "concept-order-v1")).shuffle(concept_order)
        edge_order = [
            [real_to_opaque[str(int(src))], real_to_opaque[str(int(dst))]]
            for src, dst in closure["edges"]
        ]
        _rng(_seed_digest(protocol_hash, materialized_hash, target, "edge-order-v1")).shuffle(edge_order)
        concepts = {
            real_to_opaque[str(node)]: names[node]
            for node in nodes
        }
        core = {
            "target_node": target,
            "target_opaque_id": real_to_opaque[str(target)],
            "real_to_opaque": real_to_opaque,
            "opaque_to_real": opaque_to_real,
            "concept_names": concepts,
            "concept_order": concept_order,
            "edge_order": edge_order,
            "mapping_seed": mapping_seed,
            "closure_hash": closure["closure_hash"],
        }
        core["mapping_hash"] = value_hash(core)
        bundles.append(core)
    return {
        "schema_version": 1,
        "manifest_hash": materialized_hash,
        "protocol_hash": protocol_hash,
        "dag_hash": sha256_file(DAG_PATH),
        "generation_source_hash": sha256_file(Path(__file__)),
        "targets": bundles,
    }
