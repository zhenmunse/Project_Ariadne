"""Deterministic DKT teacher for canonical concept-session sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.oracle_core.set_oracle_surrogate import load_deterministic_checkpoint


@dataclass(frozen=True)
class DKTSequence:
    student_id: str
    tokens: tuple[int, ...]
    target_indices: tuple[int, ...]
    outcomes: tuple[int, ...]

    def __post_init__(self) -> None:
        length = len(self.tokens)
        if not length or len(self.target_indices) != length or len(self.outcomes) != length:
            raise ValueError("DKT sequence fields must have the same positive length")
        if any(value not in (0, 1) for value in self.outcomes):
            raise ValueError("DKT outcomes must be binary")


class DKTTeacherModel(nn.Module):
    """122-token embedding -> one-layer LSTM -> 61 target probabilities."""

    def __init__(
        self,
        *,
        num_nodes: int = 61,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(2 * self.num_nodes, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.output = nn.Linear(self.hidden_dim, self.num_nodes)
        self.initial_logits = nn.Parameter(torch.zeros(self.num_nodes))

    def prefix_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return predictions before each token, excluding its outcome."""
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise TypeError("tokens must be a rank-two torch.long tensor")
        if tokens.shape[1] == 0:
            return self.initial_logits.view(1, 1, -1).expand(tokens.shape[0], 0, -1)
        if torch.any(tokens < 0) or torch.any(tokens >= 2 * self.num_nodes):
            raise ValueError("DKT token is outside the interaction vocabulary")
        embedded = self.embedding(tokens)
        hidden, _ = self.lstm(embedded)
        after_event = self.output(hidden)
        initial = self.initial_logits.view(1, 1, -1).expand(tokens.shape[0], 1, -1)
        return torch.cat((initial, after_event[:, :-1, :]), dim=1)

    def prefix_probabilities(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.prefix_logits(tokens))


def pad_sequences(
    sequences: Sequence[DKTSequence],
    *,
    pad_token: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("At least one DKT sequence is required")
    maximum = max(len(sequence.tokens) for sequence in sequences)
    batch = len(sequences)
    tokens = torch.full((batch, maximum), pad_token, dtype=torch.long)
    targets = torch.zeros((batch, maximum), dtype=torch.long)
    outcomes = torch.zeros((batch, maximum), dtype=torch.float32)
    mask = torch.zeros((batch, maximum), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence.tokens)
        tokens[row, :length] = torch.tensor(sequence.tokens, dtype=torch.long)
        targets[row, :length] = torch.tensor(sequence.target_indices, dtype=torch.long)
        outcomes[row, :length] = torch.tensor(sequence.outcomes, dtype=torch.float32)
        mask[row, :length] = True
    return tokens, targets, outcomes, mask


def masked_next_target_bce(
    model: DKTTeacherModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    outcomes: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    probabilities = model.prefix_probabilities(tokens)
    observed = probabilities.gather(2, targets.unsqueeze(2)).squeeze(2)
    losses = F.binary_cross_entropy(observed, outcomes, reduction="none")
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("mask must select at least one observed event")
    return losses[mask].mean()


class FrozenDKTTeacher:
    """CPU-only read-only prefix inference from a deterministic checkpoint."""

    def __init__(self, model: DKTTeacherModel, node_order: Sequence[int]) -> None:
        self.model = model.cpu().eval()
        self.node_order = tuple(int(node) for node in node_order)
        self.node_to_index = {node: index for index, node in enumerate(self.node_order)}
        if len(self.node_to_index) != len(self.node_order):
            raise ValueError("node_order contains duplicates")
        self.model.requires_grad_(False)

    @classmethod
    def from_artifacts(
        cls,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
    ) -> "FrozenDKTTeacher":
        import json

        with Path(config_path).open("r", encoding="utf-8") as file:
            config = json.load(file)
        payload = load_deterministic_checkpoint(checkpoint_path)
        architecture = config["architecture"]
        model = DKTTeacherModel(
            num_nodes=architecture["num_nodes"],
            embedding_dim=architecture["embedding_dim"],
            hidden_dim=architecture["hidden_dim"],
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        if tuple(payload["node_order"]) != tuple(config["node_order"]):
            raise ValueError("DKT checkpoint/config node order mismatch")
        return cls(model, config["node_order"])

    def token(self, node: int, outcome: int) -> int:
        if node not in self.node_to_index:
            raise ValueError(f"Unknown DKT node: {node}")
        if outcome not in (0, 1):
            raise ValueError("DKT outcome must be binary")
        return 2 * self.node_to_index[node] + int(outcome)

    def probabilities_before_events(self, tokens: Sequence[int]) -> torch.Tensor:
        if not tokens:
            return torch.empty((0, len(self.node_order)), dtype=torch.float32)
        tensor = torch.tensor([list(tokens)], dtype=torch.long)
        with torch.inference_mode():
            values = self.model.prefix_probabilities(tensor)[0].cpu()
        if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 1):
            raise ValueError("DKT teacher returned invalid probabilities")
        return values

    def probability_table(self, tokens: Sequence[int]) -> torch.Tensor:
        """Return empty-prefix plus every post-event probability table."""
        initial = torch.sigmoid(self.model.initial_logits.detach()).view(1, -1)
        if not tokens:
            return initial.cpu()
        tensor = torch.tensor([list(tokens)], dtype=torch.long)
        with torch.inference_mode():
            embedded = self.model.embedding(tensor)
            hidden, _ = self.model.lstm(embedded)
            after = torch.sigmoid(self.model.output(hidden))[0]
        result = torch.cat((initial.cpu(), after.cpu()), dim=0)
        if not torch.isfinite(result).all():
            raise ValueError("DKT teacher returned non-finite probabilities")
        return result


def binary_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    pairs = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positives = sum(label == 1 for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in pairs[index:end])
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    if not math.isfinite(auc):
        raise ValueError("AUC is not finite")
    return float(auc)
