from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYBKT_ROOT = ROOT.parent / "pyBKT"
INTERACTIONS = ROOT / "data" / "processed" / "cleaned_interactions.csv"
MAPPING = ROOT / "data" / "question_concept_mapping_final.csv"
OUTPUT = PYBKT_ROOT / "data" / "output" / "pybkt_interactions.csv"


def main() -> None:
    if not PYBKT_ROOT.exists():
        raise FileNotFoundError(
            f"pyBKT repo not found at {PYBKT_ROOT}. Expected a sibling checkout named 'pyBKT'."
        )
    interactions = pd.read_csv(INTERACTIONS)
    mapping = pd.read_csv(MAPPING, usecols=["item_id", "concept_id"])

    if mapping["item_id"].duplicated().any():
        raise ValueError("item_id must map to exactly one concept_id")

    data = interactions.merge(mapping, on="item_id", how="left", validate="many_to_one")
    missing_items = sorted(data.loc[data["concept_id"].isna(), "item_id"].unique())
    data = data.dropna(subset=["concept_id"]).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data = data.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)

    output = pd.DataFrame(
        {
            "order_id": range(len(data)),
            "user_id": data["user_id"],
            "skill_name": data["concept_id"].astype(int),
            "correct": data["is_correct"].astype(int),
        }
    )

    if not output["correct"].isin([0, 1]).all():
        raise ValueError("correct must contain only 0 or 1")
    if output.isna().any().any():
        raise ValueError("pyBKT output must not contain missing values")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(f"rows={len(output)} users={output.user_id.nunique()} skills={output.skill_name.nunique()}")
    print(f"dropped_unmapped={len(interactions) - len(output)} missing_items={missing_items}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
