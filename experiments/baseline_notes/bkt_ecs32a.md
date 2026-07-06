# BKT Baseline on ECS32A

## Summary

The pyBKT baseline was trained on cleaned ECS32A interactions.

| metric | value |
|---|---:|
| test AUC | 0.663470 |
| test accuracy | 0.670555 |
| test RMSE | 0.460752 |
| training time | 271.47 seconds |

Required concept parameters are stored in
`results/bkt_concept_parameters.csv`.

## Files

| file | purpose |
|---|---|
| `experiments/07_convert_to_pybkt.py` | Converts Ariadne data to pyBKT format. |
| `experiments/08_reproduce_bkt.py` | Trains and evaluates the BKT baseline. |
| `results/bkt_metrics.csv` | Aggregate metrics. |
| `results/bkt_concept_parameters.csv` | Per-concept BKT parameters. |

## Data

Inputs:

```text
data/processed/cleaned_interactions.csv
data/question_concept_mapping_final.csv
```

The converter joins by `item_id`, drops unmapped rows, sorts interactions by
student and timestamp, and writes pyBKT's required columns:

```text
order_id,user_id,skill_name,correct
```

Conversion result:

```text
raw rows: 71,350
retained rows: 70,893
dropped unmapped rows: 457
students: 294
items: 141
concepts: 34
unmapped items: 9596878, 9597496, 9643536
```

## Training

Students are shuffled with `seed=42` and split 80/20 to avoid student leakage.

| split | students | interactions |
|---|---:|---:|
| train | 235 | 56,062 |
| test | 59 | 14,831 |

One BKT model is trained per concept using pyBKT defaults:

| setting | value |
|---|---|
| initialization fits | 5 |
| forgetting | disabled |
| EM maximum iterations | 100 |
| EM tolerance | 0.005 |
| parallel execution | disabled |

`parallel=False` changes execution strategy only and avoids inefficient process
creation in the pure Python Windows backend.

## Outputs

`results/bkt_metrics.csv`:

```text
AUC: 0.6634698241
accuracy: 0.6705549188
RMSE: 0.4607519785
training time: 271.4659 seconds
```

`results/bkt_concept_parameters.csv` contains:

| column | meaning |
|---|---|
| `p_init` | Initial mastery probability. |
| `p_learn` | Unmastered-to-mastered transition probability. |
| `p_guess` | Correct-response probability while unmastered. |
| `p_slip` | Incorrect-response probability while mastered. |

Only 34 of the 61 DAG concepts have mapped questions. Concepts `3`, `7`, `19`,
`41`, and `57` have no within-student repeated observations, so their transition
parameters are not identifiable and should not be interpreted substantively.

Setup note: the Windows installation used pyBKT's pure Python fallback and
required minor compatibility corrections for current NumPy and scikit-learn.
These corrections affected execution only, not the BKT model or reported data.

## Reproduce

From the Ariadne repository root:

```powershell
../pyBKT/.venv/Scripts/python.exe experiments/07_convert_to_pybkt.py
../pyBKT/.venv/Scripts/python.exe experiments/08_reproduce_bkt.py
```
