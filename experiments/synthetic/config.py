"""Configuration, artifact schemas, and provenance for synthetic experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "synthetic"
OUTPUT_SUBDIRECTORIES = (
    "dependence_sweep",
    "trap_family",
    "bound_ablation",
    "figures",
    "final",
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _json_value(value: Any) -> Any:
    """Convert supported objects to canonical JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        resolved = value.resolve()
        try:
            return resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()
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
            raise ValueError("Synthetic values must not contain NaN or infinity")
        return value
    raise TypeError(f"Unsupported synthetic value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    """Defensively copy JSON data into recursively immutable values."""
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_json(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata must not contain NaN or infinity")
        return value
    raise TypeError(f"Metadata contains a non-JSON value: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    """Convert recursively immutable metadata back to ordinary JSON values."""
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw_json(item) for key, item in value}
        return [_thaw_json(item) for item in value]
    return value


def _exact_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    if set(payload) != expected:
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonArtifact:
    """Schema-aware JSON artifact protocol shared by G, H, and oracle data."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JsonArtifact":
        raise NotImplementedError

    @property
    def artifact_hash(self) -> str:
        return value_hash(self.to_dict())

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: str | Path) -> "JsonArtifact":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls.from_dict(payload)


@dataclass(frozen=True)
class SeedConfig:
    """Four distinct deterministic seed namespaces."""

    graph: int = 1101
    transfer: int = 1201
    calibration: int = 1301
    evaluation: int = 1401

    def __post_init__(self) -> None:
        values = (self.graph, self.transfer, self.calibration, self.evaluation)
        for name, value in zip(
            ("graph", "transfer", "calibration", "evaluation"), values
        ):
            _nonnegative_int(value, f"seeds.{name}")
        if len(set(values)) != len(values):
            raise ValueError("Synthetic seed namespaces must all be different")

    def to_dict(self) -> dict[str, int]:
        return {
            "graph": self.graph,
            "transfer": self.transfer,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedConfig":
        fields = {"graph", "transfer", "calibration", "evaluation"}
        _exact_fields(payload, fields, "Seed config")
        return cls(**payload)


@dataclass(frozen=True)
class GraphArtifactConfig(JsonArtifact):
    graph_id: str
    family: str
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    target: int
    seed: int
    metadata: Any = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.graph_id or not self.family:
            raise ValueError("graph_id and family must be non-empty")
        nodes = tuple(_nonnegative_int(node, "graph node") for node in self.nodes)
        if len(nodes) != len(set(nodes)):
            raise ValueError("Graph nodes must be unique")
        edges = tuple(
            (_nonnegative_int(src, "edge source"), _nonnegative_int(dst, "edge target"))
            for src, dst in self.edges
        )
        if len(edges) != len(set(edges)):
            raise ValueError("Graph edges must be unique")
        if any(src not in nodes or dst not in nodes for src, dst in edges):
            raise ValueError("Graph edge references an unknown node")
        if self.target not in nodes:
            raise ValueError("Graph target must be a graph node")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "graph seed"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "graph_id": self.graph_id,
            "family": self.family,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "target": self.target,
            "seed": self.seed,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphArtifactConfig":
        fields = {"schema_version", "graph_id", "family", "nodes", "edges", "target", "seed", "metadata"}
        _exact_fields(payload, fields, "Graph artifact")
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported graph artifact schema_version")
        return cls(
            graph_id=payload["graph_id"], family=payload["family"],
            nodes=tuple(payload["nodes"]), edges=tuple(tuple(edge) for edge in payload["edges"]),
            target=payload["target"], seed=payload["seed"], metadata=payload["metadata"],
        )


@dataclass(frozen=True)
class TransferArtifactConfig(JsonArtifact):
    transfer_id: str
    graph_id: str
    policy: str
    weights: tuple[tuple[int, int, float], ...]
    seed: int
    metadata: Any = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.transfer_id or not self.graph_id or not self.policy:
            raise ValueError("transfer_id, graph_id, and policy must be non-empty")
        weights = []
        for src, dst, weight in self.weights:
            src = _nonnegative_int(src, "transfer source")
            dst = _nonnegative_int(dst, "transfer target")
            if src == dst:
                raise ValueError("Self-transfer is not allowed")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("Transfer weight must be numeric")
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Transfer weight must be finite and nonnegative")
            weights.append((src, dst, weight))
        if len({(src, dst) for src, dst, _ in weights}) != len(weights):
            raise ValueError("Transfer edges must be unique")
        object.__setattr__(self, "weights", tuple(weights))
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "transfer seed"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "transfer_id": self.transfer_id,
            "graph_id": self.graph_id, "policy": self.policy,
            "weights": [list(weight) for weight in self.weights], "seed": self.seed,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransferArtifactConfig":
        fields = {"schema_version", "transfer_id", "graph_id", "policy", "weights", "seed", "metadata"}
        _exact_fields(payload, fields, "Transfer artifact")
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported transfer artifact schema_version")
        return cls(
            transfer_id=payload["transfer_id"], graph_id=payload["graph_id"],
            policy=payload["policy"], weights=tuple(tuple(row) for row in payload["weights"]),
            seed=payload["seed"], metadata=payload["metadata"],
        )


@dataclass(frozen=True)
class OracleParameterConfig(JsonArtifact):
    oracle_id: str
    graph_id: str
    transfer_id: str
    beta: float
    alpha_by_node: Any
    base_cost: float = 60.0
    metadata: Any = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.oracle_id or not self.graph_id or not self.transfer_id:
            raise ValueError("oracle_id, graph_id, and transfer_id must be non-empty")
        beta = float(self.beta)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        alpha_items = self.alpha_by_node.items() if isinstance(self.alpha_by_node, Mapping) else self.alpha_by_node
        alpha = []
        for node, value in alpha_items:
            node = _nonnegative_int(node, "alpha node")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("alpha values must be finite")
            alpha.append((node, value))
        alpha.sort()
        if len({node for node, _ in alpha}) != len(alpha):
            raise ValueError("alpha_by_node contains duplicate nodes")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "alpha_by_node", tuple(alpha))
        object.__setattr__(self, "base_cost", _positive_float(self.base_cost, "base_cost"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "oracle_id": self.oracle_id, "graph_id": self.graph_id,
            "transfer_id": self.transfer_id, "beta": self.beta,
            "alpha_by_node": [list(item) for item in self.alpha_by_node],
            "base_cost": self.base_cost, "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OracleParameterConfig":
        fields = {"schema_version", "oracle_id", "graph_id", "transfer_id", "beta", "alpha_by_node", "base_cost", "metadata"}
        _exact_fields(payload, fields, "Oracle artifact")
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported oracle artifact schema_version")
        return cls(
            oracle_id=payload["oracle_id"], graph_id=payload["graph_id"],
            transfer_id=payload["transfer_id"], beta=payload["beta"],
            alpha_by_node=tuple(tuple(item) for item in payload["alpha_by_node"]),
            base_cost=payload["base_cost"], metadata=payload["metadata"],
        )


class JsonConfig:
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JsonConfig":
        raise NotImplementedError

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "JsonConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


@dataclass(frozen=True)
class SyntheticCommonConfig(JsonConfig):
    schema_version: int = 1
    base_cost: float = 60.0
    output_root: Path = DEFAULT_OUTPUT_ROOT
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported synthetic config schema_version")
        object.__setattr__(self, "base_cost", _positive_float(self.base_cost, "base_cost"))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if not isinstance(self.seeds, SeedConfig):
            object.__setattr__(self, "seeds", SeedConfig.from_dict(self.seeds))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "base_cost": self.base_cost, "output_root": _json_value(self.output_root), "seeds": self.seeds.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticCommonConfig":
        fields = {"schema_version", "base_cost", "output_root", "seeds"}
        _exact_fields(payload, fields, "Synthetic common config")
        output_root = Path(payload["output_root"])
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        return cls(payload["schema_version"], payload["base_cost"], output_root, SeedConfig.from_dict(payload["seeds"]))


# Backward-compatible S1 name; later runners use the explicit common name.
SyntheticExperimentConfig = SyntheticCommonConfig


@dataclass(frozen=True)
class LayeredGraphFamilyConfig:
    sizes: tuple[tuple[int, int], ...] = ((4, 4), (4, 5), (4, 6))
    graph_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    edge_density: float = 0.3
    max_order_ideals: int = 2_000_000

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LayeredGraphFamilyConfig":
        _exact_fields(payload, {"sizes", "graph_seeds", "edge_density", "max_order_ideals"}, "Layered graph config")
        return cls(tuple(tuple(size) for size in payload["sizes"]), tuple(payload["graph_seeds"]), float(payload["edge_density"]), payload["max_order_ideals"])


@dataclass(frozen=True)
class TransferGenerationConfig:
    policy: str = "mixed_transfer"
    densities: tuple[float, ...] = (0.1, 0.2, 0.3)
    max_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]: return _json_value(asdict(self))
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransferGenerationConfig":
        _exact_fields(payload, {"policy", "densities", "max_weight"}, "Transfer generation config")
        return cls(payload["policy"], tuple(float(value) for value in payload["densities"]), float(payload["max_weight"]))


@dataclass(frozen=True)
class CalibrationConfig:
    samples: int = 10_000
    absolute_tolerance: float = 1e-10
    maximum_iterations: int = 200

    def to_dict(self) -> dict[str, Any]: return _json_value(asdict(self))
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationConfig":
        _exact_fields(payload, {"samples", "absolute_tolerance", "maximum_iterations"}, "Calibration config")
        return cls(payload["samples"], float(payload["absolute_tolerance"]), payload["maximum_iterations"])


@dataclass(frozen=True)
class DependenceSweepConfig(JsonConfig):
    common: SyntheticCommonConfig = field(default_factory=SyntheticCommonConfig)
    beta_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    graph: LayeredGraphFamilyConfig = field(default_factory=LayeredGraphFamilyConfig)
    transfer: TransferGenerationConfig = field(default_factory=TransferGenerationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    random_frontier_runs: int = 100
    solver_tolerance: float = 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {"common": self.common.to_dict(), "beta_grid": list(self.beta_grid), "graph": self.graph.to_dict(), "transfer": self.transfer.to_dict(), "calibration": self.calibration.to_dict(), "random_frontier_runs": self.random_frontier_runs, "solver_tolerance": self.solver_tolerance}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DependenceSweepConfig":
        fields = {"common", "beta_grid", "graph", "transfer", "calibration", "random_frontier_runs", "solver_tolerance"}
        _exact_fields(payload, fields, "Dependence sweep config")
        return cls(SyntheticCommonConfig.from_dict(payload["common"]), tuple(float(value) for value in payload["beta_grid"]), LayeredGraphFamilyConfig.from_dict(payload["graph"]), TransferGenerationConfig.from_dict(payload["transfer"]), CalibrationConfig.from_dict(payload["calibration"]), payload["random_frontier_runs"], float(payload["solver_tolerance"]))


@dataclass(frozen=True)
class TrapFamilyConfig(JsonConfig):
    common: SyntheticCommonConfig = field(default_factory=SyntheticCommonConfig)
    p_a: float = 0.8
    q: float = 0.75
    delta_grid: tuple[float, ...] = tuple(round(value * 0.01, 2) for value in range(11))
    tau_grid: tuple[float, ...] = tuple(round(value * 0.1, 1) for value in range(11))
    k_values: tuple[int, ...] = (2, 4, 8, 16)
    solver_tolerance: float = 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {"common": self.common.to_dict(), "p_a": self.p_a, "q": self.q, "delta_grid": list(self.delta_grid), "tau_grid": list(self.tau_grid), "k_values": list(self.k_values), "solver_tolerance": self.solver_tolerance}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrapFamilyConfig":
        fields = {"common", "p_a", "q", "delta_grid", "tau_grid", "k_values", "solver_tolerance"}
        _exact_fields(payload, fields, "Trap family config")
        return cls(SyntheticCommonConfig.from_dict(payload["common"]), float(payload["p_a"]), float(payload["q"]), tuple(float(v) for v in payload["delta_grid"]), tuple(float(v) for v in payload["tau_grid"]), tuple(payload["k_values"]), float(payload["solver_tolerance"]))


@dataclass(frozen=True)
class BoundAblationConfig(JsonConfig):
    common: SyntheticCommonConfig = field(default_factory=SyntheticCommonConfig)
    solver_tolerance: float = 1e-9

    def to_dict(self) -> dict[str, Any]: return {"common": self.common.to_dict(), "solver_tolerance": self.solver_tolerance}
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoundAblationConfig":
        _exact_fields(payload, {"common", "solver_tolerance"}, "Bound ablation config")
        return cls(SyntheticCommonConfig.from_dict(payload["common"]), float(payload["solver_tolerance"]))


@dataclass(frozen=True)
class RunProvenanceInputs:
    graph_artifact: Path
    transfer_artifact: Path
    oracle_artifact: Path
    experiment_config: Path
    runner_code: tuple[Path, ...]

    def __post_init__(self) -> None:
        for name in ("graph_artifact", "transfer_artifact", "oracle_artifact", "experiment_config"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        runner_code = tuple(Path(path) for path in self.runner_code)
        if not runner_code:
            raise ValueError("runner_code must contain at least one code artifact")
        object.__setattr__(self, "runner_code", runner_code)


def _validate_commit_sha(commit_sha: str) -> str:
    if not isinstance(commit_sha, str) or COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise ValueError("commit_sha must be 40 lowercase hexadecimal characters")
    return commit_sha


def repository_state() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8").strip()
    _validate_commit_sha(commit)
    status_bytes = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
    )
    status = [
        item.decode("utf-8", errors="surrogateescape")
        for item in status_bytes.split(b"\0")
        if item
    ]
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT)
    untracked = []
    for line in status:
        if not line.startswith("?? "):
            continue
        path = ROOT / line[3:]
        untracked.append({"path": path, "sha256": file_hash(path) if path.is_file() else None})
    diff_payload = {
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "untracked": untracked,
    }
    return {
        "repository_commit_sha": commit,
        "repository_dirty": bool(status_bytes),
        "git_diff_hash": value_hash(diff_payload),
    }


def _file_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _json_value(path), "sha256": file_hash(path)}


def _resolve_recorded_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_run_record(
    *,
    run_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    config: JsonConfig,
    provenance_inputs: RunProvenanceInputs,
    allow_dirty_repository: bool = False,
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Build a run record only when every mandatory dependency is supplied."""
    state = repository_state()
    if state["repository_dirty"] and not allow_dirty_repository:
        raise RuntimeError("Synthetic formal runs require a clean repository")
    if commit_sha is not None:
        state["repository_commit_sha"] = _validate_commit_sha(commit_sha)

    config_payload = config.to_dict()
    with provenance_inputs.experiment_config.open("r", encoding="utf-8") as file:
        recorded_config = json.load(file)
    if value_hash(recorded_config) != value_hash(config_payload):
        raise ValueError("experiment_config artifact does not match the materialized config")

    runner_refs = [_file_reference(path) for path in provenance_inputs.runner_code]
    runner_refs.sort(key=lambda ref: ref["path"])
    payload = {
        "schema_version": 1,
        **state,
        "run_identity": _json_value(run_identity),
        "result": _json_value(result),
        "config": config_payload,
        "config_hash": value_hash(config_payload),
        "artifacts": {
            "graph": _file_reference(provenance_inputs.graph_artifact),
            "transfer": _file_reference(provenance_inputs.transfer_artifact),
            "oracle": _file_reference(provenance_inputs.oracle_artifact),
            "experiment_config": _file_reference(provenance_inputs.experiment_config),
            "runner_code": runner_refs,
        },
    }
    return {**payload, "provenance_hash": value_hash(payload)}


def verify_run_record(record: Mapping[str, Any], *, verify_artifacts: bool = True) -> bool:
    if "provenance_hash" not in record:
        return False
    payload = {key: value for key, value in record.items() if key != "provenance_hash"}
    if record["provenance_hash"] != value_hash(payload):
        return False
    if COMMIT_PATTERN.fullmatch(str(record.get("repository_commit_sha", ""))) is None:
        return False
    if record.get("config_hash") != value_hash(record.get("config")):
        return False
    if re.fullmatch(r"[0-9a-f]{64}", str(record.get("git_diff_hash", ""))) is None:
        return False
    if not verify_artifacts:
        return True
    try:
        artifacts = record["artifacts"]
        required = {"graph", "transfer", "oracle", "experiment_config", "runner_code"}
        if set(artifacts) != required or not artifacts["runner_code"]:
            return False
        refs = [artifacts[name] for name in ("graph", "transfer", "oracle", "experiment_config")]
        refs.extend(artifacts["runner_code"])
        return all(file_hash(_resolve_recorded_path(ref["path"])) == ref["sha256"] for ref in refs)
    except (KeyError, TypeError, FileNotFoundError, OSError):
        return False


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    records = list(records)
    for index, record in enumerate(records):
        if not verify_run_record(record):
            raise ValueError(f"Invalid provenance for run record {index}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(canonical_json_bytes(record).decode("utf-8") + "\n")


def _common_config(config: JsonConfig) -> SyntheticCommonConfig:
    return config if isinstance(config, SyntheticCommonConfig) else config.common


def ensure_output_directories(config: JsonConfig) -> dict[str, Path]:
    root = _common_config(config).output_root
    root.mkdir(parents=True, exist_ok=True)
    directories = {"root": root}
    for name in OUTPUT_SUBDIRECTORIES:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        directories[name] = directory
    return directories


ConfigT = TypeVar("ConfigT", bound=JsonConfig)


def load_config(path: str | Path | None, config_type: type[ConfigT]) -> ConfigT:
    return config_type.load(path) if path else config_type()


def scaffold_cli(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Task-specific synthetic JSON config")
    parser.add_argument("--initialize-only", action="store_true", help="Create output directories and print the materialized config")
    return parser


def initialize_from_cli(args: argparse.Namespace, *, config_type: type[ConfigT] = SyntheticCommonConfig) -> ConfigT:
    config = load_config(args.config, config_type)
    ensure_output_directories(config)
    if not args.initialize_only:
        raise SystemExit("This Task S1 entry point is initialized; use --initialize-only until its task logic is implemented.")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return config


def main() -> None:
    initialize_from_cli(scaffold_cli("Initialize the Task S1 synthetic experiment package").parse_args())


if __name__ == "__main__":
    main()
