"""Evaluate the trained ECS32A MonotonicOracle on held-out students."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHECKPOINT = PROCESSED / "oracle_ckpt.pt"
OUTPUT = PROCESSED / "ariadne_oracle_valid_metrics.csv"
sys.path.insert(0, str(ROOT))

from src.oracle_core.dataset import get_dataloader
from src.oracle_core.model import MonotonicOracle


def auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positives = labels == 1
    n_positive = positives.sum()
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUC requires both binary classes")
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def main() -> None:
    with (PROCESSED / "graph.pkl").open("rb") as file:
        graph = pickle.load(file)
    with (PROCESSED / "valid_sessions.pkl").open("rb") as file:
        samples = pickle.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if checkpoint["node_id_to_idx"] != graph["node_id_to_idx"]:
        raise ValueError("checkpoint and validation graph use different node mappings")

    model = MonotonicOracle(
        num_nodes=checkpoint["num_nodes"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
    loader = get_dataloader(
        samples,
        graph["node_id_to_idx"],
        len(graph["node_ids"]),
        batch_size=checkpoint["config"]["batch_size"],
        shuffle=False,
    )

    probabilities = []
    labels = []
    with torch.no_grad():
        for x, target, mask, label in loader:
            probability, _ = model.forward_batch(
                x.to(device),
                edge_index,
                target.to(device),
                mask.to(device),
            )
            probabilities.append(probability.cpu())
            labels.append(label)

    y_prob = torch.cat(probabilities).numpy()
    y_true = torch.cat(labels).numpy()
    binary_mask = np.isin(y_true, [0.0, 1.0])
    binary_true = y_true[binary_mask]
    binary_prob = y_prob[binary_mask]

    auc = np.nan
    accuracy = np.nan
    if len(binary_true) and np.unique(binary_true).size == 2:
        auc = auc_score(binary_true, binary_prob)
        accuracy = float(np.mean((binary_prob >= 0.5) == (binary_true == 1.0)))

    metrics = pd.DataFrame(
        [
            {
                "samples": len(y_true),
                "binary_samples": int(binary_mask.sum()),
                "mse": np.mean((y_true - y_prob) ** 2),
                "rmse": np.sqrt(np.mean((y_true - y_prob) ** 2)),
                "mae": np.mean(np.abs(y_true - y_prob)),
                "auc": auc,
                "accuracy": accuracy,
            }
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT, index=False)
    print(metrics.to_string(index=False))
    print(f"metrics={OUTPUT}")


if __name__ == "__main__":
    main()
