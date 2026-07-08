"""Convert Ariadne ECS32A interactions into the minimal pyKT DKT dataset.

Run with the pyKT environment available:
    python experiments/05_convert_to_pykt.py

The generated pyKT files are written to the sibling pykt-toolkit checkout:
    ../pykt-toolkit/data/output
"""

from pathlib import Path
import json
import os
import shutil
import tempfile

import pandas as pd
from pykt.preprocess.split_datasets import main as split_concept


ROOT = Path(__file__).resolve().parents[1]
PYKT_ROOT = ROOT.parent / "pykt-toolkit"
OUT = PYKT_ROOT / "data" / "output"
DATASET = "ecs32a_ariadne"
CONFIG = PYKT_ROOT / "configs" / "data_config.json"


def main() -> None:
    interactions = pd.read_csv(ROOT / "data" / "processed" / "cleaned_interactions.csv")
    mapping = pd.read_csv(ROOT / "data" / "question_concept_mapping_template.csv")

    needed = {"user_id", "item_id", "is_correct", "timestamp"}
    if not needed <= set(interactions.columns):
        raise ValueError(f"cleaned_interactions.csv missing {needed - set(interactions.columns)}")
    if not {"item_id", "concept_id"} <= set(mapping.columns):
        raise ValueError("question_concept_mapping_template.csv needs item_id,concept_id")
    if mapping["item_id"].duplicated().any():
        raise ValueError("question_concept_mapping_template.csv has duplicate item_id rows")

    df = interactions.merge(mapping[["item_id", "concept_id"]], on="item_id", how="left", validate="many_to_one")
    unmapped = df["concept_id"].isna()
    missing_items = sorted(df.loc[unmapped, "item_id"].drop_duplicates().astype(int).tolist())
    df = df.loc[~unmapped].copy()

    df["is_correct"] = df["is_correct"].astype(int)
    if not set(df["is_correct"]) <= {0, 1}:
        raise ValueError("is_correct must contain only 0/1")

    df["item_id"] = df["item_id"].astype(int)
    df["concept_id"] = df["concept_id"].astype(int)
    df["_time"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values(["user_id", "_time", "item_id"], kind="mergesort")

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        seq_file = Path(tmp) / "ecs32a_sequences.txt"
        with seq_file.open("w", encoding="utf8", newline="\n") as f:
            for uid, group in df.groupby("user_id", sort=False):
                f.write(f"{uid},{len(group)}\n")
                f.write(",".join(group["item_id"].astype(str)) + "\n")
                f.write(",".join(group["concept_id"].astype(str)) + "\n")
                f.write(",".join(group["is_correct"].astype(str)) + "\n")
                f.write("NA\nNA\n")  # ponytail: required only by pyKT raw split format; final files omit these fields.

        old_cwd = Path.cwd()
        os.chdir(PYKT_ROOT / "examples")
        try:
            split_concept("../data/output", str(seq_file), DATASET, str(CONFIG), 3, 200, 5)
        finally:
            os.chdir(old_cwd)

    # NOTE: Avoid deleting other artifacts in ../pykt-toolkit/data/output; it may contain outputs for other datasets.
    # If you need a clean output directory, remove unwanted files manually.

    config = json.loads(CONFIG.read_text(encoding="utf8"))
    old = config[DATASET]
    config[DATASET] = {
        "dpath": "../data/output",
        "num_q": old["num_q"],
        "num_c": old["num_c"],
        "input_type": old["input_type"],
        "max_concepts": old["max_concepts"],
        "min_seq_len": old["min_seq_len"],
        "maxlen": old["maxlen"],
        "emb_path": "",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": old["folds"],
        "test_file": "test_sequences.csv",
    }
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=4), encoding="utf8")

    assert (OUT / "train_valid_sequences.csv").is_file()
    assert (OUT / "test_sequences.csv").is_file()
    print(f"raw_rows={len(interactions)} kept_rows={len(df)} dropped_unmapped={int(unmapped.sum())}")
    print(f"users={df.user_id.nunique()} items={df.item_id.nunique()} concepts={df.concept_id.nunique()}")
    print(f"missing_items={missing_items}")
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
