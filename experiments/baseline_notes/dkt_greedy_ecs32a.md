# DKT + Greedy on ECS32A

## Scope

Condition 4 of the planning matrix. This experiment uses the DKT prediction
artifacts archived under `data/baselines/pykt/ecs32a_ariadne/` and keeps the
same DAG, data split, targets, seed, and cost model as the other Greedy
conditions. Each target is planned on its induced prerequisite closure, using
the closure as both the planner graph and terminal goal.

The planner uses training-fold DKT predictions averaged by concept. This is a
population-level adapter, not a student-specific DKT hidden-state planner.
The final planner script reads only local archived artifacts and does not
depend on an external pyKT checkout.

## DKT validation

| Metric | Value |
|---|---:|
| Samples | 12,022 |
| AUC | 0.730437 |
| Accuracy | 0.710614 |
| RMSE | 0.435532 |
| MAE | 0.380541 |

## Planning result

The same ten targets were used with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 1075.324149 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Total planning time | 0.0005 s |
| Prerequisite-valid paths | 10 / 10 |

The result is generated from the archived DKT prediction artifacts and is
restricted to the target prerequisite closures, matching the LAO* benchmark
protocol.

## Files and reproduction

| File | Purpose |
|---|---|
| `data/baselines/pykt/ecs32a_ariadne/qid_model.ckpt` | Archived DKT checkpoint. |
| `data/baselines/pykt/ecs32a_ariadne/train_valid_sequences.csv` | Archived DKT sequences. |
| `data/baselines/pykt/ecs32a_ariadne/train_node_probabilities.csv` | Local training-fold DKT predictions. |
| `data/baselines/pykt/ecs32a_ariadne/validation_metrics.json` | Archived DKT validation metrics. |
| `experiments/14_run_dkt_greedy.py` | Runs Greedy from local DKT artifacts. |
| `results/dkt_greedy/` | Planner metrics and trajectories. |

```powershell
.\.venv\Scripts\python.exe experiments\14_run_dkt_greedy.py
```
