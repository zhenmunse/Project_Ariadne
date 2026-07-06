import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PYBKT_ROOT = ROOT.parent / "pyBKT"
DATA = PYBKT_ROOT / "data" / "output" / "pybkt_interactions.csv"
METRICS = ROOT / "results" / "bkt_metrics.csv"
PARAMETERS = ROOT / "results" / "bkt_concept_parameters.csv"
sys.path.insert(0, str(PYBKT_ROOT / "source-py"))

from pyBKT.models import Model


def main() -> None:
    data = pd.read_csv(DATA)
    users = data["user_id"].drop_duplicates().to_numpy()
    np.random.default_rng(42).shuffle(users)
    test_users = set(users[: round(len(users) * 0.2)])
    train = data[~data["user_id"].isin(test_users)]
    test = data[data["user_id"].isin(test_users)]

    missing_skills = set(test["skill_name"]) - set(train["skill_name"])
    if missing_skills:
        raise ValueError(f"test-only skills: {sorted(missing_skills)}")

    training_seconds = 0.0
    predictions = []
    parameter_rows = []
    concepts = sorted(train["skill_name"].unique())
    for index, concept_id in enumerate(concepts, start=1):
        train_concept = train[train["skill_name"] == concept_id]
        test_concept = test[test["skill_name"] == concept_id]
        print(
            f"[{index}/{len(concepts)}] concept={concept_id} "
            f"train_rows={len(train_concept)}",
            flush=True,
        )
        model = Model(seed=42, parallel=False)
        started = perf_counter()
        model.fit(data=train_concept)
        training_seconds += perf_counter() - started
        if not test_concept.empty:
            predictions.append(model.predict(data=test_concept))
        values = next(iter(model.fit_model.values()))
        parameter_rows.append(
            {
                "concept_id": int(concept_id),
                "p_init": values["prior"],
                "p_learn": values["learns"][0],
                "p_guess": values["guesses"][0],
                "p_slip": values["slips"][0],
            }
        )

    predictions = pd.concat(predictions, ignore_index=True)
    y_true = predictions["correct"].to_numpy()
    y_prob = predictions["correct_predictions"].to_numpy()
    metrics = pd.DataFrame(
        [
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "train_users": train["user_id"].nunique(),
                "test_users": test["user_id"].nunique(),
                "auc": roc_auc_score(y_true, y_prob),
                "accuracy": accuracy_score(y_true, y_prob >= 0.5),
                "rmse": np.sqrt(np.mean((y_true - y_prob) ** 2)),
                "training_seconds": training_seconds,
            }
        ]
    )
    metrics.to_csv(METRICS, index=False)
    pd.DataFrame(parameter_rows).sort_values("concept_id").to_csv(PARAMETERS, index=False)

    print(metrics.to_string(index=False))
    print(f"metrics={METRICS}")
    print(f"parameters={PARAMETERS}")


if __name__ == "__main__":
    main()
