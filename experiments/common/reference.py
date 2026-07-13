"""Strict loading of provenance-matched cross-condition reference records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.common.manifest import manifest_hash
from experiments.common.schema import Method, SequenceRecord, read_jsonl


def load_single_run_reference(
    path: str | Path,
    *,
    expected_method: Method,
    manifest: dict,
    expected_metadata: Mapping[str, Any] | None = None,
    require_internal_cost: bool = False,
    expected_run_id: int = 0,
) -> dict[int, SequenceRecord]:
    """Load exactly one provenance-matched record for every manifest target."""
    records = read_jsonl(path)
    if len(records) != len(manifest["targets"]):
        raise ValueError(
            f"Expected exactly one {expected_method.value} record per target"
        )

    expected_targets = set(manifest["targets"])
    expected_manifest_hash = manifest_hash(manifest)
    closure_hashes = {
        closure["target_node"]: closure["closure_hash"]
        for closure in manifest["closures"]
    }
    required_metadata = dict(expected_metadata or {})
    by_target = {}
    for record in records:
        if record.method is not expected_method:
            raise ValueError(
                f"Reference contains method {record.method.value}; "
                f"expected {expected_method.value}"
            )
        if record.run_id != expected_run_id:
            raise ValueError(
                f"{expected_method.value} reference must use run_id={expected_run_id}"
            )
        if record.target_node not in expected_targets:
            raise ValueError(
                f"Unexpected {expected_method.value} target: {record.target_node}"
            )
        if record.target_node in by_target:
            raise ValueError(
                f"Duplicate {expected_method.value} target: {record.target_node}"
            )
        if require_internal_cost and record.internal_cost is None:
            raise ValueError(
                f"{expected_method.value} reference is missing internal_cost"
            )
        if record.metadata.get("manifest_hash") != expected_manifest_hash:
            raise ValueError(f"{expected_method.value} manifest hash mismatch")
        if record.metadata.get("closure_hash") != closure_hashes[record.target_node]:
            raise ValueError(f"{expected_method.value} closure hash mismatch")
        for key, expected_value in required_metadata.items():
            if record.metadata.get(key) != expected_value:
                raise ValueError(
                    f"{expected_method.value} metadata mismatch for {key}"
                )
        by_target[record.target_node] = record

    if set(by_target) != expected_targets:
        raise ValueError(
            f"{expected_method.value} target set does not match the shared manifest"
        )
    return by_target
