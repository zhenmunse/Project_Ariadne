# DKT + LAO* on ECS32A

## Scope

Condition 5 of the planning matrix. This experiment uses the same archived
DKT prediction artifacts, DAG, data split, targets, seed, and cost model as
`DKT + Greedy`; only the Solver changes to LAO*. Each target is planned on
its induced prerequisite closure, using the closure as both graph and goal.

The planner reads training-fold DKT predictions averaged by concept. This is a
population-level adapter, not a student-specific DKT hidden-state planner.

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
| Mean expanded states | 21.9 |
| Total planning time | 0.0069 s |
| Prerequisite-valid paths | 10 / 10 |
| Converged targets | 10 / 10 |

The result is generated from local DKT prediction artifacts and follows the
same prerequisite-closure protocol as the LAO* heuristic benchmark.

## Files and reproduction

| File | Purpose |
|---|---|
| `data/baselines/pykt/ecs32a_ariadne/train_node_probabilities.csv` | Local training-fold DKT predictions. |
| `data/baselines/pykt/ecs32a_ariadne/validation_metrics.json` | DKT validation metrics. |
| `experiments/15_run_dkt_lao.py` | Runs LAO* from local DKT artifacts. |
| `results/dkt_lao/` | Planner metrics and trajectories. |

```powershell
.\.venv\Scripts\python.exe experiments\15_run_dkt_lao.py
```
