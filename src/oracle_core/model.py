"""
model.py  --  MonotonicOracle: concept-level success-probability predictor.

Architecture:
    1. h = node_emb + score_proj(x)          -- combine learnable + feature embeddings
    2. z = ReLU( GCN(dropout(h), edge_index) ) -- message passing (dense matmul, vectorized over batch)
    3. difficulty = diff_net(z[target])        -- [B]
    4. ability    = ability_net(score-weighted mean of z)  -- [B], independent of mask
    5. prereq_strength = |prereq_weight| * mean( softplus(z) * mask )  -- monotonic in mask, [B]
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

Vectorized batching:
    Since edge_index is shared across all samples (static homogeneous graph),
    the GCN aggregation A_norm @ (H @ W) is computed via dense matmul.
    A_norm is precomputed once and cached.  All ops broadcast over [B, ...].

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

        # Dense normalized adjacency cache (built lazily)
        self._adj_norm: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Adjacency cache
    # ------------------------------------------------------------------

    def _build_adj_norm(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Precompute GCN-normalized dense adjacency: D_hat^{-1/2} A_hat D_hat^{-1/2}.

        A_hat = A + I  (self-loops).  Result shape [N, N].
        """
        N = self.num_nodes
        device = edge_index.device

        adj = torch.zeros(N, N, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + torch.eye(N, device=device)          # add self-loops

        deg = adj.sum(dim=1)                              # [N]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # D^{-1/2} A D^{-1/2}
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return adj_norm

    def _get_adj_norm(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Return cached adj_norm, rebuilding if device changed or first call."""
        if self._adj_norm is None or self._adj_norm.device != edge_index.device:
            self._adj_norm = self._build_adj_norm(edge_index)
        return self._adj_norm

    # ------------------------------------------------------------------
    # Unified forward  (works for single sample AND batched)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_node: torch.Tensor,
        current_prereq_mask: torch.Tensor,
    ):
        """
        Args  (single):
            x:                    [N, 2]
            edge_index:           [2, E]
            target_node:          scalar
            current_prereq_mask:  [N]

        Args  (batched):
            x:                    [B, N, 2]
            edge_index:           [2, E]   (shared)
            target_node:          [B]
            current_prereq_mask:  [B, N]

        Returns:
            prob: scalar / [B]   -- P_succ
            gap:  scalar / [B]   -- pre-sigmoid logit
        """
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)                            # -> [1, N, 2]
            target_node = target_node.unsqueeze(0)        # -> [1]
            current_prereq_mask = current_prereq_mask.unsqueeze(0)  # -> [1, N]

        B, N, _ = x.shape
        H = self.hidden_dim

        # 1. Node representations  [B, N, H]
        #    node_emb.weight [N, H] broadcasts over B
        h = self.node_emb.weight.unsqueeze(0) + self.score_proj(x)
        h = self.drop(h)

        # 2. GCN via dense matmul  (vectorized)
        #    GCNConv linear:  h_proj = h @ W^T + b_lin   [B, N, H]
        #    Aggregation:     z_raw  = A_norm @ h_proj    [B, N, H]
        #    Final bias:      z_raw += bias_gnn
        adj_norm = self._get_adj_norm(edge_index)         # [N, N]

        h_proj = self.gnn.lin(h)                          # [B, N, H]  (nn.Linear handles batch)
        z = torch.matmul(adj_norm, h_proj)                # [N,N] @ [B,N,H] -> [B,N,H]
        if self.gnn.bias is not None:
            z = z + self.gnn.bias                         # [B, N, H]
        z = torch.relu(z)
        z = self.drop(z)

        # 3. Difficulty  [B]
        #    gather target embeddings: z[b, target_node[b], :]
        idx = target_node.view(B, 1, 1).expand(B, 1, H)  # [B, 1, H]
        target_emb = z.gather(1, idx).squeeze(1)          # [B, H]
        difficulty = self.diff_net(target_emb).squeeze(-1) # [B]

        # 4. User ability  [B]   (independent of mask => monotonicity safe)
        scores = x[:, :, 1]                               # [B, N]
        n_learned = x[:, :, 0].sum(dim=1).clamp(min=1.0)  # [B]
        weighted_z = (z * scores.unsqueeze(-1)).sum(dim=1) / n_learned.unsqueeze(-1)  # [B, H]
        ability = self.ability_net(weighted_z).squeeze(-1) # [B]

        # 5. Prereq strength  (MONOTONIC in mask)  [B]
        z_pos = F.softplus(z)                                       # [B, N, H] >= 0
        masked_z = z_pos * current_prereq_mask.unsqueeze(-1)        # [B, N, H]
        prereq_agg = masked_z.sum(dim=1).mean(dim=-1)               # [B]
        prereq_strength = self.prereq_weight.abs().squeeze() * prereq_agg  # [B]

        # 6. Gap & probability
        gap = ability - difficulty + prereq_strength                # [B]
        prob = torch.sigmoid(gap)                                   # [B]

        if single:
            return prob.squeeze(0), gap.squeeze(0)
        return prob, gap

    # ------------------------------------------------------------------
    # Backward-compatible batch entry point
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        x_batch: torch.Tensor,
        edge_index: torch.Tensor,
        target_batch: torch.Tensor,
        mask_batch: torch.Tensor,
    ):
        """Thin wrapper -- delegates to vectorized forward().

        Args / Returns: same shapes as before ([B, ...]).
        """
        return self.forward(x_batch, edge_index, target_batch, mask_batch)

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
