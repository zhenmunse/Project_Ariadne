# Ariadne + LAO* on ECS32A

## Scope

This condition combines the local Ariadne `MonotonicOracle` checkpoint with
the repository's LAO* planner. It uses seed 42 and the same ten non-root
targets as the other ECS32A planner baselines. Each target uses its induced
prerequisite closure as both the planner graph and terminal goal.

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
| Mean expected total cost | 1724.133156 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Mean expanded states | 16.7 |
| Total planning time | 0.8135 s |
| Valid paths | 10 / 10 |
| Converged targets | 10 / 10 |
| Probability source | local Ariadne checkpoint |

Under the closure-restricted protocol, LAO* searches only target-relevant
states and returns a valid long-horizon ordering. The model uses MC Dropout,
so small run-to-run timing and cost variation is expected.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/18_run_ariadne_lao.py` | Runs validation and LAO* planning. |
| `results/ariadne_lao/` | Metrics, trajectories, and summary. |

```powershell
.\.venv\Scripts\python.exe experiments\18_run_ariadne_lao.py
```

The run follows the same prerequisite-closure protocol as the LAO* heuristic
benchmark.
