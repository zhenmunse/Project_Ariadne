"""Validate parsed opaque sequences without repair."""

from __future__ import annotations

from typing import Any

from experiments.llm.models import ParseResult, ValidationResult


def _invalid(code: str, detail: str) -> ValidationResult:
    return ValidationResult(False, None, code, detail)


def validate_sequence(parsed: ParseResult, bundle: dict[str, Any]) -> ValidationResult:
    if not parsed.parse_valid or not parsed.schema_valid or parsed.opaque_sequence is None:
        return _invalid(parsed.parse_error_code or "invalid_parse", parsed.parse_error_detail or "")
    sequence = parsed.opaque_sequence
    expected = set(bundle["opaque_to_real"])
    unknown = sorted(set(sequence) - expected)
    if unknown:
        return _invalid("unknown_opaque_id", f"Unknown opaque IDs: {unknown}")
    if len(sequence) != len(set(sequence)):
        return _invalid("duplicate_opaque_id", "Opaque sequence contains duplicates")
    missing = sorted(expected - set(sequence))
    if missing:
        return _invalid("missing_opaque_id", f"Missing opaque IDs: {missing}")
    if sequence[-1] != bundle["target_opaque_id"]:
        return _invalid("target_not_final", "Target opaque ID must be final")
    position = {opaque: index for index, opaque in enumerate(sequence)}
    violations = [
        [source, target]
        for source, target in bundle["edge_order"]
        if position[source] >= position[target]
    ]
    if violations:
        return _invalid("prerequisite_violation", f"Violated edges: {violations}")
    real = tuple(int(bundle["opaque_to_real"][opaque]) for opaque in sequence)
    return ValidationResult(True, real, None, None)
