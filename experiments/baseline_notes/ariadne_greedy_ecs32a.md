# Ariadne + Greedy on ECS32A

## Scope

This condition combines the local Ariadne `MonotonicOracle` checkpoint with
the repository's one-step `GreedyPlanner`. It uses seed 42 and the same ten
non-root targets as the other ECS32A planner baselines. Each target is planned
on its induced prerequisite closure, using the closure as graph and goal.

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
| Mean expected total cost | 1762.466515 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Valid paths | 10 / 10 |
| Probability source | local Ariadne checkpoint |

The closure restriction removes target-irrelevant actions. Greedy still chooses
the lowest immediate cost within the closure and does not account for future
costs.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/17_run_ariadne_greedy.py` | Runs validation and planning. |
| `results/ariadne_greedy/` | Metrics, trajectories, and summary. |

```powershell
.\.venv\Scripts\python.exe experiments\17_run_ariadne_greedy.py
```

The run uses the same prerequisite-closure protocol as the LAO* benchmark.
