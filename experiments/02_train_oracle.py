"""
experiments/02_train_oracle.py
===============================
Train MonotonicOracle on concept-level session data from Step 1.

Pipeline:
  1. Load graph.pkl  (num_nodes, edge_index, node_id_to_idx)
  2. Load train_sessions.pkl  (samples)
  3. Build DataLoader
  4. Train with MSE loss (prob vs avg_score)
  5. Each epoch: monotonic sanity check  (mask_A superset mask_B => prob_A >= prob_B)
  6. MC Dropout test
  7. Save checkpoint -> data/processed/oracle_ckpt.pt

Usage:
    python experiments/02_train_oracle.py
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------- make project root importable -------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.oracle_core.dataset import OracleDataset, get_dataloader
from src.oracle_core.model import MonotonicOracle


# ------------------------------------------------------------------
# Monotonic sanity check
# ------------------------------------------------------------------

def monotonic_sanity_check(
    model: MonotonicOracle,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    epoch: int,
) -> None:
    """Verify prob(mask_A) >= prob(mask_B) when mask_A is a superset of mask_B.

    Uses x = zeros (empty history), target = last node.
    mask_B = [1, 0, 0, ...]   (only node 0 satisfied)
    mask_A = [1, 1, 0, ...]   (nodes 0 and 1 satisfied -- superset)
    """
    model.eval()

    x = torch.zeros(num_nodes, 2, device=device)
    target = torch.tensor(num_nodes - 1, dtype=torch.long, device=device)

    mask_B = torch.zeros(num_nodes, device=device)
    mask_B[0] = 1.0

    mask_A = mask_B.clone()
    mask_A[1] = 1.0  # superset

    with torch.no_grad():
        prob_A, _ = model.forward(x, edge_index, target, mask_A)
        prob_B, _ = model.forward(x, edge_index, target, mask_B)

    pa = prob_A.item()
    pb = prob_B.item()
    passed = (pa + 1e-6) >= pb

    print(
        f"  [monotonic] epoch {epoch}: "
        f"prob(A)={pa:.6f}  prob(B)={pb:.6f}  "
        f"diff={pa - pb:+.2e}  -> {'PASS' if passed else 'FAIL'}"
    )
    assert passed, (
        f"Monotonicity violated! prob(A)={pa}, prob(B)={pb}"
    )

    model.train()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    # --- Load config -------------------------------------------------
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = os.path.join(ROOT, cfg["data"]["processed_dir"])
    oracle_cfg = cfg["oracle"]
    seed = cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Load graph --------------------------------------------------
    with open(os.path.join(processed_dir, "graph.pkl"), "rb") as f:
        graph = pickle.load(f)

    num_nodes = len(graph["node_ids"])
    node_id_to_idx = graph["node_id_to_idx"]
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    print(f"Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")

    # --- Load training samples ---------------------------------------
    with open(os.path.join(processed_dir, "train_sessions.pkl"), "rb") as f:
        train_samples = pickle.load(f)

    print(f"Training samples: {len(train_samples)}")

    # --- DataLoader --------------------------------------------------
    batch_size = min(oracle_cfg["batch_size"], len(train_samples))
    loader = get_dataloader(
        train_samples, node_id_to_idx, num_nodes,
        batch_size=batch_size, shuffle=True,
    )

    # Print first batch shapes
    first_batch = next(iter(loader))
    x_b, tgt_b, mask_b, y_b = first_batch
    print(f"\nBatch shapes:")
    print(f"  x:      {list(x_b.shape)}")       # [B, N, 2]
    print(f"  target: {list(tgt_b.shape)}")      # [B]
    print(f"  mask:   {list(mask_b.shape)}")     # [B, N]
    print(f"  y:      {list(y_b.shape)}")        # [B]

    # --- Model -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MonotonicOracle(
        num_nodes=num_nodes,
        hidden_dim=oracle_cfg["hidden_dim"],
        dropout=oracle_cfg["dropout"],
    ).to(device)

    edge_index = edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=oracle_cfg["lr"])
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params} parameters")
    print(f"Device: {device}")
    print(f"Loss: MSELoss")

    # --- Training loop -----------------------------------------------
    epochs = max(oracle_cfg["epochs"], 3)
    print(f"\n{'=' * 60}")
    print(f"Training for {epochs} epochs ...")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_b, tgt_b, mask_b, y_b in loader:
            x_b = x_b.to(device)
            tgt_b = tgt_b.to(device)
            mask_b = mask_b.to(device)
            y_b = y_b.to(device)

            probs, _ = model.forward_batch(x_b, edge_index, tgt_b, mask_b)
            loss = criterion(probs, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch:3d}/{epochs} | loss = {avg_loss:.6f}")

        # Monotonic sanity check every epoch
        monotonic_sanity_check(model, edge_index, num_nodes, device, epoch)

    # --- MC Dropout test ---------------------------------------------
    print(f"\n{'=' * 60}")
    print("MC Dropout inference test")
    print(f"{'=' * 60}")

    ds = OracleDataset(train_samples, node_id_to_idx, num_nodes)
    x_test, tgt_test, mask_test, y_test = ds[0]
    x_test = x_test.to(device)
    tgt_test = tgt_test.to(device)
    mask_test = mask_test.to(device)

    mean_p, var_p, t_base = model.predict_mc(
        x_test, edge_index, tgt_test, mask_test,
        mc_samples=oracle_cfg["mc_samples"],
    )

    print(f"  mean_prob = {mean_p.item():.6f}")
    print(f"  var_prob  = {var_p.item():.8f}")
    print(f"  T_base    = {t_base.item():.1f}")
    assert var_p.item() >= 0, "var_prob must be >= 0"
    print("  [OK] var_prob >= 0")

    # --- Save checkpoint ---------------------------------------------
    ckpt_path = os.path.join(processed_dir, "oracle_ckpt.pt")
    ckpt = {
        "state_dict": model.state_dict(),
        "config": oracle_cfg,
        "node_id_to_idx": node_id_to_idx,
        "num_nodes": num_nodes,
    }
    torch.save(ckpt, ckpt_path)
    print(f"\nCheckpoint saved -> {ckpt_path}")

    print("\n=== Step 2 COMPLETE ===")


if __name__ == "__main__":
    main()
