# FrequencyOracle + Greedy on ECS32A

## Scope

Condition 1 of the planning matrix. `FrequencyOracle` estimates a fixed
per-concept success probability from the training split. `GreedyPlanner` then
repeatedly chooses the currently valid DAG action with the lowest immediate
expected cost, `60 / p(success)`.

The Oracle is fit from 25,089 training sessions. Metrics use the 3,103-session
held-out validation split; AUC and accuracy use its 2,170 binary labels.

| Oracle metric | Value |
|---|---:|
| AUC | 0.662523 |
| Accuracy | 0.779724 |
| RMSE | 0.370814 |
| MAE | 0.320591 |

## Planning result

Ten observed, non-root target concepts were sampled with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`. Each plan starts from an empty
mastery state on the 61-node, 134-edge ECS32A DAG.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 2018.949783 |
| Mean path length | 24.4 |
| Mean off-target actions | 12.7 |
| Total planning time for 10 targets | 0.0049 s |
| Prerequisite-valid paths | 10 / 10 |

`off_target_actions` counts selected nodes outside the target's ancestor
closure. Its nonzero value is expected for this myopic baseline: Greedy ranks
all currently valid actions by immediate frequency-derived cost and does not
look ahead to whether an action advances the selected target. The next
condition, FrequencyOracle + LAO*, uses the same Oracle, graph, targets, and
cost model, changing only the Solver.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/11_run_frequency_greedy.py` | Runs the condition and validates every generated path. |
| `results/frequency_greedy/oracle_valid_metrics.csv` | Held-out Oracle metrics. |
| `results/frequency_greedy/planner_trajectories.csv` | Per-target costs, paths, and legality checks. |
| `results/frequency_greedy/summary.json` | Aggregate planner metrics and fixed targets. |

```powershell
.\.venv\Scripts\python.exe experiments\11_run_frequency_greedy.py
.\.venv\Scripts\python.exe experiments\04a_baseline_check.py
```

The baseline self-check passed: FrequencyOracle returned valid probabilities
for all 61 nodes, and the Greedy trap test remained distinct from the optimal
planner.
