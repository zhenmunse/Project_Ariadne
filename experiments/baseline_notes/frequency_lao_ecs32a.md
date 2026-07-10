# FrequencyOracle + LAO* on ECS32A

## Scope

Condition 2 of the planning matrix. This experiment keeps the FrequencyOracle,
data split, target nodes, random seed, and cost model from
`FrequencyOracle + Greedy`, replacing only the Solver with LAO*.
For each target, both the planner graph and terminal goal are the target's
prerequisite closure.

The Oracle uses 25,089 training sessions and is evaluated on 3,103 validation
sessions. Its metrics are unchanged from the Greedy condition.

| Oracle metric | Value |
|---|---:|
| AUC | 0.662523 |
| Accuracy | 0.779724 |
| RMSE | 0.370814 |
| MAE | 0.320591 |

## Planning result

The same ten targets were used with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 977.292569 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Mean expanded states | 1156.9 |
| Total planning time for 10 targets | 0.0052 s |
| Converged targets | 10 / 10 |
| Prerequisite-valid paths | 10 / 10 |

Compared with the corrected Greedy condition, LAO* has the same mean cost and
path length because FrequencyOracle is state-independent. The closure removes
target-irrelevant actions from both conditions; LAO* still verifies the same
ordering with fewer expanded states than full-DAG search.

The reported cost is the planner's model-based expected cost, not an observed
student learning outcome. The Oracle prediction quality is deliberately held
constant so this condition isolates planning strategy.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/12_run_frequency_lao.py` | Runs FrequencyOracle with LAO*. |
| `results/frequency_lao/oracle_valid_metrics.csv` | Held-out Oracle metrics. |
| `results/frequency_lao/planner_trajectories.csv` | Per-target paths and LAO* diagnostics. |
| `results/frequency_lao/summary.json` | Aggregate planning metrics. |

```powershell
.\.venv\Scripts\python.exe experiments\12_run_frequency_lao.py
.\.venv\Scripts\python.exe experiments\06_verify_lao.py
```

Both the real-data run and the LAO* verification checks completed
successfully with prerequisite-closure planning.
