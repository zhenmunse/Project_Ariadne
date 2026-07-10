# Linear Syllabus Order on ECS32A

## Scope

For each target, this baseline takes the target and all of its prerequisite
ancestors, then sorts them by the official ECS32A teaching order. The supplied
teaching-order file covers all 61 DAG concepts, and all 134 DAG edges point
forward in that order.

The sequence is evaluated with the local Ariadne checkpoint so its expected
cost is comparable with the other standalone ordering baseline. No nodes
outside the target prerequisite closure are added.

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
| Mean expected total cost | 1764.103351 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Valid paths | 10 / 10 |
| Evaluation Oracle | local Ariadne checkpoint |
| Teaching-order source | `data/ecs32a_teaching_order_required_full_v1.csv` |

The syllabus ordering is a deterministic reference policy. Its cost is close to
the random valid ordering and higher than the LAO* result because it follows
the course order rather than optimizing the model-based expected cost.

## Files and reproduction

| File | Purpose |
|---|---|
| `data/ecs32a_teaching_order_required_full_v1.csv` | Official teaching order. |
| `experiments/20_run_linear_syllabus_order.py` | Generates and evaluates syllabus sequences. |
| `results/linear_syllabus_order/` | Metrics, trajectories, and summary. |

```powershell
.\.venv\Scripts\python.exe experiments\20_run_linear_syllabus_order.py
```

This branch is waiting for review and has not been committed.
