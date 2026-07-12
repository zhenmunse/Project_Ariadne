"""Typed records shared by deterministic LLM input and parsing stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseResult:
    parse_valid: bool
    schema_valid: bool
    opaque_sequence: tuple[str, ...] | None
    parse_error_code: str | None
    parse_error_detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_valid": self.parse_valid,
            "schema_valid": self.schema_valid,
            "opaque_sequence": (
                list(self.opaque_sequence) if self.opaque_sequence is not None else None
            ),
            "parse_error_code": self.parse_error_code,
            "parse_error_detail": self.parse_error_detail,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    real_sequence: tuple[int, ...] | None
    error_code: str | None
    error_detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "real_sequence": (
                list(self.real_sequence) if self.real_sequence is not None else None
            ),
            "error_code": self.error_code,
            "error_detail": self.error_detail,
        }
