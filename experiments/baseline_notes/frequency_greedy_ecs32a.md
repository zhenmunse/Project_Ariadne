# FrequencyOracle + Greedy on ECS32A

## Reproduction status

This is the first rerun after the planner fallback fix at commit `8f1ba3c`.
The run uses the real ECS32A processed sessions, seed 42, the fixed ten
targets, and the 61-node/134-edge prerequisite DAG.

## Oracle metrics

| Metric | Value |
|---|---:|
| Samples | 3,103 |
| Binary samples | 2,170 |
| AUC | 0.662523 |
| Accuracy | 0.779724 |
| RMSE | 0.370814 |
| MAE | 0.320591 |

FrequencyOracle uses the mean training-session label for each observed node
and the global training mean for unmapped DAG nodes.

## Planning metrics

Targets:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Metric | Value |
|---|---:|
| Mean expected total cost | 2018.949783 |
| Mean path length | 24.4 |
| Mean off-target actions | 12.7 |
| Total planning time | 0.0046 s |
| Prerequisite-valid paths | 10 / 10 |

The nonzero off-target count is expected for Greedy: it selects the cheapest
currently valid action without looking ahead to the target's ancestor closure.

## Reproduction inputs and outputs

| Path | Purpose |
|---|---|
| `data/processed/graph.pkl` | ECS32A DAG and node mapping. |
| `data/processed/train_sessions.pkl` | FrequencyOracle training samples. |
| `data/processed/valid_sessions.pkl` | Oracle validation samples. |
| `experiments/11_run_frequency_greedy.py` | Rerun script. |
| `results/frequency_greedy/oracle_valid_metrics.csv` | Oracle metrics. |
| `results/frequency_greedy/planner_trajectories.csv` | Per-target paths and checks. |
| `results/frequency_greedy/summary.json` | Aggregate planner metrics. |

Verification commands:

```powershell
.\.venv\Scripts\python.exe experiments\11_run_frequency_greedy.py
.\.venv\Scripts\python.exe experiments\04a_baseline_check.py
```

Both commands completed successfully. This result is waiting for review;
later matrix conditions have not been rerun yet.
