"""
preprocessor.py  --  Raw interaction logs -> concept-level sessions -> training samples.

Pipeline:
  1. Read raw CSV  (user_id, item_id, is_correct, timestamp)
  2. Filter items not present in item2node mapping
  3. Flatten concurrent events  (same user+timestamp -> sort by item_id asc)
  4. Map item_id -> node_id
  5. Session aggregation  (consecutive same-node records per user = one session)
  6. Build training samples:
       List[ (user_history, target_node, label) ]
       user_history = List[ (node_id, avg_score) ]  (sessions before current)
       target_node  = int   (node_id)
       label        = float (avg_score of current session, 0~1)

Output artifacts:
  - sessions.pkl      raw session DataFrame for inspection
  - train_sessions.pkl   frozen training format
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Step 1 : Load & filter
# ------------------------------------------------------------------

def _load_and_filter(
    raw_path: str,
    item2node: Dict[int, int],
) -> pd.DataFrame:
    """Read raw CSV -> filter items not in mapping -> return DataFrame."""
    df = pd.read_csv(raw_path)
    required = {"user_id", "item_id", "is_correct", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}")

    n_before = len(df)
    df = df[df["item_id"].isin(item2node)].copy()
    n_after = len(df)
    print(f"[preprocess] Filtered items not in graph: {n_before} -> {n_after} rows")
    return df


# ------------------------------------------------------------------
# Step 2 : Flatten concurrent events
# ------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Sort so that same (user_id, timestamp) are ordered by item_id asc.

    This enforces the 'no concurrent events' invariant from the spec.
    After sorting, assign a monotonic integer `order` column.
    """
    df = df.sort_values(
        by=["user_id", "timestamp", "item_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    df["order"] = range(len(df))
    return df


# ------------------------------------------------------------------
# Step 3 : Item -> Node mapping
# ------------------------------------------------------------------

def _map_to_nodes(df: pd.DataFrame, item2node: Dict[int, int]) -> pd.DataFrame:
    df["node_id"] = df["item_id"].map(item2node)
    return df


# ------------------------------------------------------------------
# Step 4 : Session aggregation (concept-level)
# ------------------------------------------------------------------

def _aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Concept Session rule (v0.2):
    For each user, walk records in `order`.
    Consecutive records with the same node_id form one session.
    A change in node_id (or a different user) starts a new session.

    Returns a DataFrame with columns:
      user_id, session_id, node_id, avg_score, timestamp, n_items
    where timestamp = first timestamp in the session.
    """
    records: list = []
    prev_uid = None
    prev_nid = None
    sess_id = -1

    # accumulators for current session
    scores: list = []
    ts_first = None

    def _flush():
        if scores:
            records.append({
                "user_id": prev_uid,
                "session_id": sess_id,
                "node_id": prev_nid,
                "avg_score": float(np.mean(scores)),
                "timestamp": ts_first,
                "n_items": len(scores),
            })

    for row in df.itertuples(index=False):
        uid = int(getattr(row, "user_id"))
        nid = int(getattr(row, "node_id"))
        correct = int(getattr(row, "is_correct"))
        ts = getattr(row, "timestamp")

        if uid != prev_uid:
            # new user -> flush previous user's last session & reset
            _flush()
            prev_uid = uid
            prev_nid = nid
            sess_id = 0
            scores = [correct]
            ts_first = ts
        elif nid != prev_nid:
            # same user, different concept -> new session
            _flush()
            prev_nid = nid
            sess_id += 1
            scores = [correct]
            ts_first = ts
        else:
            # same user, same concept -> extend session
            scores.append(correct)

    _flush()  # last session

    sess_df = pd.DataFrame(records)
    return sess_df


# ------------------------------------------------------------------
# Step 5 : Build training samples
# ------------------------------------------------------------------

def _build_train_samples(
    sess_df: pd.DataFrame,
) -> List[Tuple[List[Tuple[int, float]], int, float]]:
    """For each session row, create a training sample:
      (user_history, target_node, label)

    user_history = list of (node_id, avg_score) from *earlier* sessions
                   of the same user (ordered by session_id).
    """
    samples: list = []
    grouped = sess_df.groupby("user_id")

    for uid, udf in grouped:
        udf = udf.sort_values("session_id").reset_index(drop=True)
        history: List[Tuple[int, float]] = []
        for _, row in udf.iterrows():
            target_node = int(row["node_id"])
            label = float(row["avg_score"])
            samples.append((list(history), target_node, label))
            history.append((target_node, label))

    return samples


# ------------------------------------------------------------------
# Public entry-point
# ------------------------------------------------------------------

def run_preprocess(
    raw_path: str,
    item2node: Dict[int, int],
    output_dir: str,
) -> Tuple[pd.DataFrame, list]:
    """Full pipeline: raw csv -> sessions.pkl + train_sessions.pkl.

    Returns (sess_df, train_samples) for programmatic inspection.
    """
    # 1. load & filter
    df = _load_and_filter(raw_path, item2node)

    # 2. flatten
    df = _flatten(df)

    # -- Print a serialization example for verification ---------------
    dup_mask = df.duplicated(subset=["user_id", "timestamp"], keep=False)
    dup_groups = df[dup_mask].groupby(["user_id", "timestamp"])
    if len(dup_groups) > 0:
        first_key = tuple(list(dup_groups.groups.keys())[0])  # type: ignore[arg-type]
        example = dup_groups.get_group(first_key)[["user_id", "timestamp", "item_id", "order"]]
        print(f"[preprocess] Serialization example (user={first_key[0]}, ts={first_key[1]}):")
        print(example.to_string(index=False))
    else:
        print("[preprocess] No concurrent events found in this dataset.")

    # 3. map items -> nodes
    df = _map_to_nodes(df, item2node)

    # 4. session aggregation
    sess_df = _aggregate_sessions(df)
    print(f"[preprocess] Sessions: {len(sess_df)} total, "
          f"{sess_df['user_id'].nunique()} users")

    # 5. training samples
    train_samples = _build_train_samples(sess_df)
    print(f"[preprocess] Training samples: {len(train_samples)}")

    # 6. save
    os.makedirs(output_dir, exist_ok=True)

    sess_path = os.path.join(output_dir, "sessions.pkl")
    with open(sess_path, "wb") as f:
        pickle.dump(sess_df, f)
    print(f"[preprocess] Saved sessions.pkl -> {sess_path}")

    train_path = os.path.join(output_dir, "train_sessions.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_samples, f)
    print(f"[preprocess] Saved train_sessions.pkl -> {train_path}")

    return sess_df, train_samples
