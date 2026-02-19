"""
model.py  --  MonotonicOracle: concept-level success-probability predictor.

Architecture (single-sample forward pass):
    1. h = node_emb + score_proj(x)          -- combine learnable + feature embeddings
    2. z = ReLU( GCN(dropout(h), edge_index) ) -- message passing
    3. difficulty = diff_net(z[target])        -- scalar
    4. ability    = ability_net(score-weighted mean of z)  -- scalar, independent of mask
    5. prereq_strength = |prereq_weight| * mean( softplus(z) * mask )  -- monotonic in mask
    6. gap  = ability - difficulty + prereq_strength
    7. prob = sigmoid(gap)

Monotonicity guarantee:
    When mask_A >= mask_B element-wise (and x, target, weights are identical):
      softplus(z) >= 0  for every element
      => sum(softplus(z) * mask_A) >= sum(softplus(z) * mask_B)
      |prereq_weight| >= 0
      => prereq_strength(A) >= prereq_strength(B)
      ability & difficulty are constant w.r.t. mask
      => gap(A) >= gap(B)
      => sigmoid is monotone => prob(A) >= prob(B)   QED

Frozen-spec interface for Planner (via predict_mc):
    predict_mc(...) -> (P_succ, sigma2, T_base)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MonotonicOracle(nn.Module):
    """GCN-based Oracle with hard monotonicity in prerequisite mask."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # --- Layers ---
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.score_proj = nn.Linear(2, hidden_dim)
        self.gnn = GCNConv(hidden_dim, hidden_dim)
        self.diff_net = nn.Linear(hidden_dim, 1)
        self.ability_net = nn.Linear(hidden_dim, 1)
        self.prereq_weight = nn.Parameter(torch.tensor([1.0]))
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Single-sample forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_node: torch.Tensor,
        current_prereq_mask: torch.Tensor,
    ):
        """
        Args:
            x:                    [N, 2]   node feature matrix
            edge_index:           [2, E]   graph connectivity (idx-based)
            target_node:          scalar   idx of target concept
            current_prereq_mask:  [N]      binary mask (1 = satisfied)

        Returns:
            prob: scalar tensor in (0, 1)  -- P_succ
            gap:  scalar tensor            -- pre-sigmoid logit
        """
        # 1. Node representations
        h = self.node_emb.weight + self.score_proj(x)  # [N, H]
        h = self.drop(h)
        z = self.gnn(h, edge_index)                    # [N, H]
        z = torch.relu(z)
        z = self.drop(z)

        # 2. Difficulty (from target node)
        target_emb = z[target_node]                    # [H]
        difficulty = self.diff_net(target_emb).squeeze(-1)  # scalar

        # 3. User ability (score-weighted mean of z)
        #    Uses x directly, NOT the mask => independent of mask => monotonicity safe
        scores = x[:, 1]                               # [N]
        n_learned = x[:, 0].sum().clamp(min=1.0)
        weighted_z = (z * scores.unsqueeze(-1)).sum(dim=0) / n_learned  # [H]
        ability = self.ability_net(weighted_z).squeeze(-1)              # scalar

        # 4. Prereq strength (MONOTONIC in mask)
        #    softplus(z) >= 0 everywhere, so adding 1s to mask can only increase strength
        z_pos = F.softplus(z)                                           # [N, H] >= 0
        masked_z = z_pos * current_prereq_mask.unsqueeze(-1)            # [N, H]
        prereq_agg = masked_z.sum(dim=0).mean()                         # scalar >= 0
        prereq_strength = self.prereq_weight.abs().squeeze() * prereq_agg  # scalar >= 0

        # 5. Gap & probability
        gap = ability - difficulty + prereq_strength
        prob = torch.sigmoid(gap)

        return prob, gap

    # ------------------------------------------------------------------
    # Batch forward (loop over B; simple for v0.2)
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        x_batch: torch.Tensor,
        edge_index: torch.Tensor,
        target_batch: torch.Tensor,
        mask_batch: torch.Tensor,
    ):
        """
        Args:
            x_batch:      [B, N, 2]
            edge_index:   [2, E]   (shared across batch)
            target_batch: [B]
            mask_batch:   [B, N]

        Returns:
            probs: [B]
            gaps:  [B]
        """
        B = x_batch.size(0)
        probs, gaps = [], []
        for i in range(B):
            p, g = self.forward(
                x_batch[i], edge_index, target_batch[i], mask_batch[i]
            )
            probs.append(p)
            gaps.append(g)
        return torch.stack(probs), torch.stack(gaps)

    # ------------------------------------------------------------------
    # MC Dropout inference (Frozen-spec: returns P_succ, sigma2, T_base)
    # ------------------------------------------------------------------

    def predict_mc(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_node: torch.Tensor,
        current_prereq_mask: torch.Tensor,
        mc_samples: int = 20,
        t_base: float = 60.0,
    ):
        """Run *mc_samples* stochastic forward passes (dropout enabled).

        Args:
            x, edge_index, target_node, current_prereq_mask: same as forward()
            mc_samples:  number of MC forward passes
            t_base:      base time constant (fixed scalar in v0.2)

        Returns:
            mean_prob:   scalar  -- E[P_succ]
            var_prob:    scalar  -- Var[P_succ]  (sigma^2)
            t_base_out:  scalar tensor
        """
        was_training = self.training
        self.train()  # enable dropout

        probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                p, _ = self.forward(
                    x, edge_index, target_node, current_prereq_mask
                )
                probs.append(p)

        probs_t = torch.stack(probs)
        mean_prob = probs_t.mean()
        var_prob = probs_t.var()

        # restore original mode
        if not was_training:
            self.eval()

        return mean_prob, var_prob, torch.tensor(t_base)
