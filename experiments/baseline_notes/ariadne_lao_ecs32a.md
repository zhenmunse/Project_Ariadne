# Ariadne + LAO* on ECS32A

## Scope

This condition combines the local Ariadne `MonotonicOracle` checkpoint with
the repository's LAO* planner. It uses the full 61-node DAG, seed 42, and the
same ten non-root targets as the other ECS32A planner baselines.

The planner adapter exposes the Oracle's state-dependent probability and a
valid best-case probability computed from the Oracle's monotonic all-mastered
state. The LAO* goal contains the target and its ancestors; this is equivalent
to reaching the target on a prerequisite DAG and gives the search an
informative admissible lower bound.

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
| Mean expected total cost | 1720.655475 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Mean expanded states | 18.3 |
| Total planning time | 9.0582 s |
| Valid paths | 10 / 10 |
| Converged targets | 10 / 10 |
| Probability source | local Ariadne checkpoint |

Compared with Ariadne + Greedy, LAO* avoids the Greedy planner's local
off-target actions and produces the expected lower planning cost.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/18_run_ariadne_lao.py` | Runs validation and LAO* planning. |
| `results/ariadne_lao/` | Metrics, trajectories, and summary. |

```powershell
.\.venv\Scripts\python.exe experiments\18_run_ariadne_lao.py
```

This branch is waiting for review and has not been committed.
