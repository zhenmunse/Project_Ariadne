"""Strict JSONL schema for method-generated learning sequences."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class Method(str, Enum):
    """Stable method identifiers used across all experiment artifacts."""

    ARIADNE_GREEDY = "ariadne_greedy"
    ARIADNE_LAO = "ariadne_lao"
    FREQUENCY_GREEDY = "frequency_greedy"
    FREQUENCY_LAO = "frequency_lao"
    BKT_SET_GREEDY = "bkt_set_greedy"
    BKT_SET_LAO = "bkt_set_lao"
    DKT_SET_GREEDY = "dkt_set_greedy"
    DKT_SET_LAO = "dkt_set_lao"
    RANDOM_FRONTIER = "random_frontier"
    LINEAR_SYLLABUS = "linear_syllabus"
    LLM_ZERO = "llm_zero"
    LLM_FULL = "llm_full"
    GPT56_SOL_ZERO = "gpt56_sol_zero"
    GPT56_SOL_FULL = "gpt56_sol_full"
    DEEPSEEK_V4_ZERO = "deepseek_v4_zero"
    DEEPSEEK_V4_FULL = "deepseek_v4_full"
    EXACT_DP = "exact_dp"


RECORD_FIELDS = {
    "method",
    "target_node",
    "run_id",
    "sequence",
    "internal_cost",
    "metadata",
}


def _require_int(value: Any, field_name: str, *, nonnegative: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def _validate_json_value(value: Any, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            _validate_json_value(item, f"{field_name}.{key}")
        return
    raise TypeError(f"{field_name} contains a non-JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class SequenceRecord:
    """One method output for one target and repetition."""

    method: Method
    target_node: int
    run_id: int
    sequence: tuple[int, ...]
    internal_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            method = self.method if isinstance(self.method, Method) else Method(self.method)
        except (TypeError, ValueError) as error:
            allowed = ", ".join(method.value for method in Method)
            raise ValueError(f"method must be one of: {allowed}") from error
        object.__setattr__(self, "method", method)

        _require_int(self.target_node, "target_node")
        _require_int(self.run_id, "run_id", nonnegative=True)

        if isinstance(self.sequence, (str, bytes)) or not isinstance(
            self.sequence, (list, tuple)
        ):
            raise TypeError("sequence must be a list or tuple of integer node IDs")
        sequence = tuple(
            _require_int(node, f"sequence[{index}]")
            for index, node in enumerate(self.sequence)
        )
        if len(sequence) != len(set(sequence)):
            raise ValueError("sequence must not contain duplicate node IDs")
        object.__setattr__(self, "sequence", sequence)

        if self.internal_cost is not None:
            if isinstance(self.internal_cost, bool) or not isinstance(
                self.internal_cost, (int, float)
            ):
                raise TypeError("internal_cost must be a number or null")
            internal_cost = float(self.internal_cost)
            if not math.isfinite(internal_cost) or internal_cost < 0:
                raise ValueError("internal_cost must be finite and nonnegative")
            object.__setattr__(self, "internal_cost", internal_cost)

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a JSON object")
        metadata = dict(self.metadata)
        _validate_json_value(metadata, "metadata")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceRecord":
        if not isinstance(value, Mapping):
            raise TypeError("record must be a JSON object")
        fields = set(value)
        missing = sorted(RECORD_FIELDS - fields)
        extra = sorted(fields - RECORD_FIELDS)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing fields: {missing}")
            if extra:
                details.append(f"unexpected fields: {extra}")
            raise ValueError("Invalid sequence record; " + "; ".join(details))
        return cls(
            method=value["method"],
            target_node=value["target_node"],
            run_id=value["run_id"],
            sequence=value["sequence"],
            internal_cost=value["internal_cost"],
            metadata=value["metadata"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "target_node": self.target_node,
            "run_id": self.run_id,
            "sequence": list(self.sequence),
            "internal_cost": self.internal_cost,
            "metadata": self.metadata,
        }


def read_jsonl(path: str | Path) -> list[SequenceRecord]:
    """Read and validate every nonblank record in a JSONL artifact."""
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(SequenceRecord.from_dict(value))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {error}") from error
    return records


def write_jsonl(
    path: str | Path,
    records: Iterable[SequenceRecord | Mapping[str, Any]],
) -> None:
    """Validate and write records as canonical one-object-per-line JSONL."""
    validated = [
        record if isinstance(record, SequenceRecord) else SequenceRecord.from_dict(record)
        for record in records
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in validated:
            file.write(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
