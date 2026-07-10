# Random Valid Ordering on ECS32A

## Scope

For each target, this baseline samples one random topological ordering of the
target and all of its DAG ancestors. At every step it samples uniformly from
the currently legal prerequisite actions. It never adds nodes outside the
target's prerequisite closure.

The ordering is evaluated with the local Ariadne checkpoint so that expected
cost remains comparable with the planner experiments. The random ordering is
the only randomized component; seed 42 is recorded in the output.

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
| Mean expected total cost | 1760.397629 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Valid paths | 10 / 10 |
| Evaluation Oracle | local Ariadne checkpoint |
| Random seed | 42 |

The mean cost is slightly higher than Ariadne + LAO* (`1720.655475`), as
expected: both use legal prerequisite paths, but LAO* selects the lower-cost
ordering while this baseline samples one ordering uniformly.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/19_run_random_valid_ordering.py` | Generates and evaluates random legal orders. |
| `results/random_valid_ordering/` | Metrics, trajectories, and summary. |

```powershell
.\.venv\Scripts\python.exe experiments\19_run_random_valid_ordering.py
```

This branch is waiting for review and has not been committed.
