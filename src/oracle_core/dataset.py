"""
dataset.py  --  Convert (user_history, target_node, label) into Oracle training tensors.

Each __getitem__ returns:
    x:                   FloatTensor [N, 2]   (node features)
    target_node_idx:     LongTensor  scalar   (target concept *idx*)
    current_prereq_mask: FloatTensor [N]      (= x[:,0] during training)
    y:                   FloatTensor scalar    (label: avg_score in [0,1])

Collated batch shapes:
    x_batch:      [B, N, 2]
    target_batch: [B]
    mask_batch:   [B, N]
    y_batch:      [B]
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class OracleDataset(Dataset):
    """Concept-level session dataset for Oracle training.

    Few-shot truncation (k):
        When k is not None, keeps only the first k samples **per target_node**.
        This enables few-shot evaluation experiments.
        Default k=None means all samples are used.
    """

    def __init__(
        self,
        samples: List[Tuple[List[Tuple[int, float]], int, float]],
        node_id_to_idx: Dict[int, int],
        num_nodes: int,
        k: Optional[int] = None,
    ):
        self.node_id_to_idx = node_id_to_idx
        self.num_nodes = num_nodes

        # Filter samples whose target_node is not in the graph
        valid: List[Tuple[List[Tuple[int, float]], int, float]] = []
        for hist, tgt, lbl in samples:
            if tgt in node_id_to_idx:
                valid.append((hist, tgt, lbl))

        # Few-shot truncation: keep first k samples per target_node
        if k is not None:
            counts: Dict[int, int] = defaultdict(int)
            truncated: List[Tuple[List[Tuple[int, float]], int, float]] = []
            for hist, tgt, lbl in valid:
                if counts[tgt] < k:
                    truncated.append((hist, tgt, lbl))
                    counts[tgt] += 1
            valid = truncated

        self.samples = valid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hist, tgt_node, label = self.samples[idx]

        # Build x [N, 2]
        #   x[i, 0] = 1.0  if node i appeared in user_history
        #   x[i, 1] = avg_score (latest value if node appears multiple times)
        x = torch.zeros(self.num_nodes, 2)
        for node_id, score in hist:
            if node_id in self.node_id_to_idx:
                nidx = self.node_id_to_idx[node_id]
                x[nidx, 0] = 1.0
                x[nidx, 1] = float(score)  # overwrites with latest

        target_idx = torch.tensor(
            self.node_id_to_idx[tgt_node], dtype=torch.long
        )
        mask = x[:, 0].clone()  # current_prereq_mask = learned flags
        y = torch.tensor(label, dtype=torch.float32)

        return x, target_idx, mask, y


# ------------------------------------------------------------------
# Collate & DataLoader helpers
# ------------------------------------------------------------------

def oracle_collate_fn(
    batch: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack samples into batched tensors.

    Returns (x_batch [B,N,2], target_batch [B], mask_batch [B,N], y_batch [B]).
    """
    xs, targets, masks, ys = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(targets),
        torch.stack(masks),
        torch.stack(ys),
    )


def get_dataloader(
    samples: list,
    node_id_to_idx: Dict[int, int],
    num_nodes: int,
    batch_size: int = 32,
    shuffle: bool = True,
    k: Optional[int] = None,
) -> DataLoader:
    """Convenience wrapper: build OracleDataset + DataLoader."""
    ds = OracleDataset(samples, node_id_to_idx, num_nodes, k=k)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=oracle_collate_fn,
    )
