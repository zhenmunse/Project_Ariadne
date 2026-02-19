"""
zpd_utils.py  --  Zone of Proximal Development (ZPD) action masking.

Given a DAG (nx.DiGraph) and a set of mastered node IDs (state),
returns the list of valid next actions (node IDs to attempt).

Rules (Frozen Spec v0.2):
  Standard:  {v | v not mastered AND all predecessors of v are mastered}
  Fallback:  triggered when Standard is empty and unfinished nodes exist
             pick the node with highest prereqs_met_ratio
             tie-break: score = ratio + 1e-6 * (1 / (node_id + 1))
             return [best_node]  (single-element list)

Note: On a well-formed DAG the fallback is technically unreachable
(there is always at least one source node with all prereqs trivially
met).  The fallback is implemented defensively per spec.
"""

from typing import List, Set

import networkx as nx


def get_valid_actions(graph: nx.DiGraph, state: Set[int]) -> List[int]:
    """Return valid next actions given current mastery state.

    Args:
        graph:  knowledge prerequisite DAG (node IDs as ints)
        state:  set of already-mastered node IDs

    Returns:
        List[int]: candidate node IDs to attempt next.
          - Standard non-empty -> sorted list of all standard candidates
          - Standard empty, unfinished -> [best_fallback_node_id]
          - All mastered -> []
    """
    all_nodes = set(graph.nodes())
    U = all_nodes - state  # unmastered

    if not U:
        return []

    # --- Standard rule ------------------------------------------------
    standard: List[int] = []
    for v in U:
        preds = set(graph.predecessors(v))
        if preds <= state:  # all prereqs satisfied
            standard.append(v)

    if standard:
        return sorted(standard)

    # --- Fallback (defensive; unreachable on a true DAG) --------------
    best_score = -1.0
    best_node = -1

    for v in U:
        preds = set(graph.predecessors(v))
        total = len(preds)
        if total == 0:
            ratio = 1.0
        else:
            mastered = len(preds & state)
            ratio = mastered / total

        score = ratio + 1e-6 * (1.0 / (v + 1))
        if score > best_score:
            best_score = score
            best_node = v

    return [best_node]
