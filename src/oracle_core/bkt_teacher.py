"""Deterministic BKT fitting and read-only prefix-query inference."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


PARAMETER_NAMES = ("prior", "learn", "guess", "slip")
BOUNDS = (
    (1e-6, 1.0 - 1e-6),
    (1e-6, 1.0 - 1e-6),
    (1e-6, 0.5 - 1e-6),
    (1e-6, 0.5 - 1e-6),
)
STARTING_POINTS = (
    (0.20, 0.10, 0.20, 0.10),
    (0.50, 0.10, 0.20, 0.10),
    (0.20, 0.20, 0.10, 0.10),
    (0.20, 0.10, 0.10, 0.20),
)
OPTIMIZER_OPTIONS = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000}
OBJECTIVE_TIE_TOLERANCE = 1e-12
NUMERICAL_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class BKTParameters:
    prior: float
    learn: float
    guess: float
    slip: float

    def __post_init__(self) -> None:
        values = (self.prior, self.learn, self.guess, self.slip)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("BKT parameters must be numeric")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("BKT parameters must be finite")
        for name, value, bounds in zip(PARAMETER_NAMES, values, BOUNDS):
            if not bounds[0] <= float(value) <= bounds[1]:
                raise ValueError(f"{name} must be in {bounds}")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class BKTFitResult:
    parameters: BKTParameters
    objective: float
    selected_restart: int
    restarts: tuple[dict[str, object], ...]


def _require_observation(value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError("BKT observation must be integer 0 or 1")
    result = int(value)
    if result not in (0, 1):
        raise ValueError("BKT observation must be 0 or 1")
    return result


def success_probability(mastery: float, parameters: BKTParameters) -> float:
    """Return P(correct) before observing the next response."""
    probability = float(
        float(mastery) * (1.0 - parameters.slip)
        + (1.0 - float(mastery)) * parameters.guess
    )
    return min(max(probability, NUMERICAL_EPSILON), 1.0 - NUMERICAL_EPSILON)


def update_mastery(
    mastery: float,
    observation: int,
    parameters: BKTParameters,
) -> float:
    """Apply evidence and then the BKT learning transition."""
    outcome = _require_observation(observation)
    probability = success_probability(mastery, parameters)
    if outcome == 1:
        posterior = float(mastery) * (1.0 - parameters.slip) / probability
    else:
        posterior = float(mastery) * parameters.slip / (1.0 - probability)
    updated = float(posterior + (1.0 - posterior) * parameters.learn)
    return min(max(updated, 0.0), 1.0)


def negative_log_likelihood(
    raw_parameters: Sequence[float],
    sequences: Sequence[Sequence[int]],
) -> float:
    """Evaluate the frozen predictive BKT negative log likelihood."""
    parameters = BKTParameters(*(float(value) for value in raw_parameters))
    total = 0.0
    for sequence in sequences:
        mastery = parameters.prior
        for raw_observation in sequence:
            observation = _require_observation(raw_observation)
            probability = success_probability(mastery, parameters)
            likelihood = probability if observation == 1 else 1.0 - probability
            total -= math.log(likelihood)
            mastery = update_mastery(mastery, observation, parameters)
    return float(total)


def fit_bkt_parameters(
    sequences: Mapping[str, Sequence[int]],
) -> BKTFitResult:
    """Fit one BKT vector using the deterministic protocol in Task 11."""
    if not sequences:
        raise ValueError("At least one BKT sequence is required")
    ordered_ids = sorted(sequences)
    ordered_sequences = [tuple(_require_observation(value) for value in sequences[key]) for key in ordered_ids]
    if any(not sequence for sequence in ordered_sequences):
        raise ValueError("BKT sequences must be non-empty")

    restart_records: list[dict[str, object]] = []
    candidates: list[tuple[int, float, BKTParameters]] = []
    for restart_index, starting_point in enumerate(STARTING_POINTS):
        result = minimize(
            negative_log_likelihood,
            np.asarray(starting_point, dtype=np.float64),
            args=(ordered_sequences,),
            method="L-BFGS-B",
            bounds=BOUNDS,
            options=OPTIMIZER_OPTIONS,
        )
        parameters_list = [float(value) for value in result.x]
        objective = float(result.fun)
        finite = math.isfinite(objective) and all(math.isfinite(value) for value in parameters_list)
        converged = bool(result.success and finite)
        record = {
            "restart_index": restart_index,
            "starting_point": list(starting_point),
            "converged": converged,
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": objective if math.isfinite(objective) else None,
            "parameters": dict(zip(PARAMETER_NAMES, parameters_list)) if finite else None,
        }
        restart_records.append(record)
        if converged:
            candidates.append(
                (restart_index, objective, BKTParameters(*parameters_list))
            )

    if not candidates:
        raise RuntimeError("All deterministic BKT optimizer restarts failed")
    best_objective = min(candidate[1] for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if candidate[1] - best_objective <= OBJECTIVE_TIE_TOLERANCE
    ]
    selected_restart, objective, parameters = min(tied, key=lambda item: item[0])
    return BKTFitResult(
        parameters=parameters,
        objective=objective,
        selected_restart=selected_restart,
        restarts=tuple(restart_records),
    )


class BKTStudentState:
    """Independent per-node BKT posteriors for one historical prefix."""

    def __init__(self, parameters: Mapping[int, BKTParameters]) -> None:
        self._parameters = dict(parameters)
        self._posterior = {
            node: node_parameters.prior
            for node, node_parameters in self._parameters.items()
        }

    @property
    def posteriors(self) -> dict[int, float]:
        return dict(self._posterior)

    def query(self, node: int) -> float:
        if not isinstance(node, int) or isinstance(node, bool):
            raise TypeError("node must be an integer node ID")
        if node not in self._parameters:
            raise ValueError(f"No BKT parameters for node {node}")
        return success_probability(self._posterior[node], self._parameters[node])

    def observe(self, node: int, observation: int) -> None:
        if not isinstance(node, int) or isinstance(node, bool):
            raise TypeError("node must be an integer node ID")
        if node not in self._parameters:
            raise ValueError(f"No BKT parameters for node {node}")
        self._posterior[node] = update_mastery(
            self._posterior[node], observation, self._parameters[node]
        )


class BKTTeacher:
    """Factory for deterministic, isolated student-prefix BKT states."""

    def __init__(self, parameters: Mapping[int, BKTParameters]) -> None:
        if not parameters:
            raise ValueError("BKT teacher requires at least one node")
        self.parameters = dict(parameters)

    @classmethod
    def from_artifact(cls, path: str | Path) -> "BKTTeacher":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        entries = payload.get("parameters")
        if not isinstance(entries, list):
            raise ValueError("BKT parameter artifact must contain a parameters list")
        parameters: dict[int, BKTParameters] = {}
        for entry in entries:
            node = entry.get("node_id")
            if not isinstance(node, int) or isinstance(node, bool):
                raise TypeError("BKT artifact node_id must be an integer")
            if node in parameters:
                raise ValueError(f"Duplicate BKT parameters for node {node}")
            parameters[node] = BKTParameters(
                prior=entry["prior"],
                learn=entry["learn"],
                guess=entry["guess"],
                slip=entry["slip"],
            )
        return cls(parameters)

    def new_student_state(self) -> BKTStudentState:
        return BKTStudentState(self.parameters)
