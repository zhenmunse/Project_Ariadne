"""Serializable configuration and provenance primitives for synthetic runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "synthetic"
OUTPUT_SUBDIRECTORIES = (
    "dependence_sweep",
    "trap_family",
    "bound_ablation",
    "figures",
    "final",
)


def _json_value(value: Any) -> Any:
    """Convert supported configuration objects to canonical JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return value.resolve().as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Synthetic configuration must not contain NaN or infinity")
        return value
    raise TypeError(f"Unsupported synthetic configuration value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a supported value with stable ordering and no whitespace."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_hash(value: Any) -> str:
    """Return the SHA-256 hash of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 hash of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SeedConfig:
    """Disjoint deterministic seed namespaces used by later tasks."""

    graph: int = 1101
    transfer: int = 1201
    calibration: int = 1301
    evaluation: int = 1401

    def __post_init__(self) -> None:
        values = (self.graph, self.transfer, self.calibration, self.evaluation)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("Synthetic seeds must be integers")
        if any(value < 0 for value in values):
            raise ValueError("Synthetic seeds must be nonnegative")
        if self.calibration == self.evaluation:
            raise ValueError("Calibration and evaluation seeds must be different")


@dataclass(frozen=True)
class GraphArtifactConfig:
    """Serializable hard-prerequisite graph payload used by graph factories."""

    graph_id: str
    family: str
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    target: int
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransferArtifactConfig:
    """Serializable soft-transfer graph payload kept separate from ``G``."""

    transfer_id: str
    graph_id: str
    policy: str
    weights: tuple[tuple[int, int, float], ...]
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OracleParameterConfig:
    """Serializable parameters for one synthetic-oracle instance."""

    oracle_id: str
    graph_id: str
    transfer_id: str
    beta: float
    alpha_by_node: Mapping[int, float]
    base_cost: float = 60.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not math.isfinite(self.base_cost) or self.base_cost <= 0:
            raise ValueError("base_cost must be finite and positive")


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Task S1 configuration shared by every synthetic runner."""

    schema_version: int = 1
    base_cost: float = 60.0
    output_root: Path = DEFAULT_OUTPUT_ROOT
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported synthetic config schema_version")
        if not math.isfinite(self.base_cost) or self.base_cost <= 0:
            raise ValueError("base_cost must be finite and positive")
        object.__setattr__(self, "output_root", Path(self.output_root))

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticExperimentConfig":
        required = {"schema_version", "base_cost", "output_root", "seeds"}
        if set(payload) != required:
            raise ValueError(f"Synthetic config fields must be exactly {sorted(required)}")
        seeds = payload["seeds"]
        if not isinstance(seeds, Mapping):
            raise TypeError("seeds must be a JSON object")
        seed_fields = {"graph", "transfer", "calibration", "evaluation"}
        if set(seeds) != seed_fields:
            raise ValueError(f"Seed fields must be exactly {sorted(seed_fields)}")
        output_root = Path(payload["output_root"])
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        return cls(
            schema_version=payload["schema_version"],
            base_cost=float(payload["base_cost"]),
            output_root=output_root,
            seeds=SeedConfig(**seeds),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise TypeError("Synthetic config must be a JSON object")
        return cls.from_dict(payload)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def ensure_output_directories(config: SyntheticExperimentConfig) -> dict[str, Path]:
    """Create and return the isolated synthetic output directories."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    directories = {"root": config.output_root}
    for name in OUTPUT_SUBDIRECTORIES:
        directory = config.output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        directories[name] = directory
    return directories


def repository_commit() -> str:
    """Return the checked-out repository commit used by a synthetic run."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"Invalid Git commit SHA: {commit!r}")
    return commit


def build_run_record(
    *,
    run_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    config: SyntheticExperimentConfig,
    input_artifacts: Iterable[str | Path] = (),
    code_artifacts: Iterable[str | Path] = (),
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Build one self-hashing run record with complete input provenance."""
    def references(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        refs = []
        for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
            if not path.is_file():
                raise FileNotFoundError(path)
            refs.append({"path": _json_value(path), "sha256": file_hash(path)})
        return refs

    payload = {
        "schema_version": 1,
        "repository_commit_sha": commit_sha or repository_commit(),
        "run_identity": _json_value(run_identity),
        "result": _json_value(result),
        "config": config.to_dict(),
        "config_hash": value_hash(config),
        "input_artifacts": references(input_artifacts),
        "code_artifacts": references(code_artifacts),
    }
    return {**payload, "provenance_hash": value_hash(payload)}


def verify_run_record(record: Mapping[str, Any]) -> bool:
    """Verify the content-addressed provenance hash of a run record."""
    if "provenance_hash" not in record:
        return False
    payload = {key: value for key, value in record.items() if key != "provenance_hash"}
    return record["provenance_hash"] == value_hash(payload)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write canonical run records after verifying every provenance hash."""
    records = list(records)
    for index, record in enumerate(records):
        if not verify_run_record(record):
            raise ValueError(f"Invalid provenance hash for run record {index}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(canonical_json_bytes(record).decode("utf-8") + "\n")


def load_config(path: str | Path | None) -> SyntheticExperimentConfig:
    return SyntheticExperimentConfig.load(path) if path else SyntheticExperimentConfig()


def scaffold_cli(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Task-specific synthetic JSON config")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Create output directories and print the materialized S1 config",
    )
    return parser


def initialize_from_cli(args: argparse.Namespace) -> SyntheticExperimentConfig:
    config = load_config(args.config)
    ensure_output_directories(config)
    if not args.initialize_only:
        raise SystemExit(
            "This Task S1 entry point is initialized; its experiment logic is implemented "
            "by the corresponding later task. Use --initialize-only to inspect it."
        )
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return config


def main() -> None:
    parser = scaffold_cli("Initialize the Task S1 synthetic experiment package")
    initialize_from_cli(parser.parse_args())


if __name__ == "__main__":
    main()
