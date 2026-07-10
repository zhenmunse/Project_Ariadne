# DKT + Greedy on ECS32A

## Scope

Condition 4 of the planning matrix. The experiment reuses the existing pyKT
ECS32A DKT checkpoint and keeps the same DAG, seed, targets, and cost model as
the other Greedy conditions.

The checkpoint is loaded from the sibling pyKT checkout; no data or model
checkpoint is copied into this repository.

## DKT validation

The DKT model is evaluated on fold 0 using its sequence history. These metrics
match the existing DKT reproduction report.

| Metric | Value |
|---|---:|
| Validation AUC | 0.730437 |
| Validation accuracy | 0.710614 |
| RMSE | 0.435532 |
| Samples | 12,022 |

For the Planner adapter, predictions from the training folds are averaged by
target concept. This avoids using validation labels to choose actions, but it
means the planning probability is a population-level concept estimate rather
than a student-specific DKT hidden state.

## Planning result

The same ten targets were used with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 2258.932923 |
| Mean path length | 25.3 |
| Mean off-target actions | 13.6 |
| Total planning time for 10 targets | 0.0060 s |
| Prerequisite-valid paths | 10 / 10 |

The DKT + Greedy result is close to the FrequencyOracle + Greedy result under
the current population-level adapter. It should not be interpreted as a fully
personalized DKT planner until the Planner state carries the student's DKT
sequence or hidden state.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/14_run_dkt_greedy.py` | Loads the existing DKT checkpoint, evaluates it, and runs Greedy. |
| `results/dkt_greedy/oracle_valid_metrics.csv` | DKT validation metrics. |
| `results/dkt_greedy/planner_trajectories.csv` | Per-target paths and costs. |
| `results/dkt_greedy/summary.json` | Aggregate planning metrics. |
| `../pykt-toolkit/examples/saved_model/.../qid_model.ckpt` | Existing DKT checkpoint used by the script. |

```powershell
..\pykt-toolkit\.venv\Scripts\python.exe experiments\14_run_dkt_greedy.py
```

All ten generated paths passed the prerequisite-validity check.
