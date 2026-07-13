"""Shared deterministic mastery-set surrogate for KT-derived oracles."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SetOracleSurrogate(nn.Module):
    """Frozen 122 -> 128 -> 64 -> 1 set-conditioned probability model."""

    def __init__(self, num_nodes: int = 61) -> None:
        super().__init__()
        if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
            raise TypeError("num_nodes must be an integer")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        self.num_nodes = num_nodes
        self.network = nn.Sequential(
            nn.Linear(2 * num_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        mastery_mask: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        if mastery_mask.ndim == 1:
            mastery_mask = mastery_mask.unsqueeze(0)
        if mastery_mask.ndim != 2 or mastery_mask.shape[1] != self.num_nodes:
            raise ValueError(
                f"mastery_mask must have shape [batch, {self.num_nodes}]"
            )
        if target_index.ndim == 0:
            target_index = target_index.unsqueeze(0)
        if target_index.ndim != 1 or target_index.shape[0] != mastery_mask.shape[0]:
            raise ValueError("target_index must have shape [batch]")
        if target_index.dtype != torch.long:
            raise TypeError("target_index must have dtype torch.long")
        target_one_hot = F.one_hot(
            target_index, num_classes=self.num_nodes
        ).to(dtype=mastery_mask.dtype)
        features = torch.cat((mastery_mask, target_one_hot), dim=1)
        return self.network(features).squeeze(1)


def save_deterministic_checkpoint(
    path: str | Path,
    *,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    """Write a byte-stable, pickle-free tensor checkpoint."""
    tensors = []
    for name in sorted(state_dict):
        array = state_dict[name].detach().cpu().contiguous().numpy()
        tensors.append(
            {
                "name": name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
            }
        )
    payload = {
        "format": "ariadne-deterministic-tensor-checkpoint-v1",
        "metadata": dict(metadata),
        "tensors": tensors,
    }
    rendered = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ) + "\n"
    Path(path).write_text(rendered, encoding="utf-8", newline="\n")


def load_deterministic_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load the deterministic checkpoint into metadata and a state dict."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("format") != "ariadne-deterministic-tensor-checkpoint-v1":
        raise ValueError("Unsupported deterministic checkpoint format")
    if not isinstance(payload.get("metadata"), dict) or not isinstance(
        payload.get("tensors"), list
    ):
        raise ValueError("Malformed deterministic checkpoint")
    state_dict: dict[str, torch.Tensor] = {}
    for entry in payload["tensors"]:
        name = entry.get("name")
        if not isinstance(name, str) or name in state_dict:
            raise ValueError("Checkpoint tensor names must be unique strings")
        dtype = np.dtype(entry["dtype"])
        raw = base64.b64decode(entry["data_base64"], validate=True)
        array = np.frombuffer(raw, dtype=dtype).copy()
        shape = tuple(entry["shape"])
        if int(np.prod(shape, dtype=np.int64)) != array.size:
            raise ValueError(f"Checkpoint tensor shape mismatch: {name}")
        state_dict[name] = torch.from_numpy(array.reshape(shape))
    return {**payload["metadata"], "state_dict": state_dict}
