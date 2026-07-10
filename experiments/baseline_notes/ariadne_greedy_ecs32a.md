# Ariadne + Greedy on ECS32A

## Scope

This condition combines the local Ariadne `MonotonicOracle` checkpoint with
the repository's one-step `GreedyPlanner`. It uses the full 61-node DAG, seed
42, and the same ten non-root targets as the other ECS32A planner baselines.

## Oracle validation

| Metric | Value |
|---|---:|
| Samples | 3,103 |
| Binary samples | 2,170 |
| AUC | 0.774873 |
| Accuracy | 0.798618 |
| RMSE | 0.351672 |
| MAE | 0.303330 |

## Planning result

| Metric | Value |
|---|---:|
| Mean expected total cost | 4080.197087 |
| Mean path length | 26.2 |
| Mean off-target actions | 14.5 |
| Valid paths | 10 / 10 |
| Probability source | local Ariadne checkpoint |

The paths are valid, but the high cost and off-target count are expected for a
myopic solver: it chooses the lowest immediate cost and does not account for
future prerequisites or whether an action is useful for the target.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/17_run_ariadne_greedy.py` | Runs validation and planning. |
| `results/ariadne_greedy/` | Metrics, trajectories, and summary. |

```powershell
.\.venv\Scripts\python.exe experiments\17_run_ariadne_greedy.py
```

This branch is waiting for review and has not been committed.
