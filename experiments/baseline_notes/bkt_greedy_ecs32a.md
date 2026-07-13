# BKT + Greedy on ECS32A

## Scope

Condition 3 of the planning matrix. The planner uses the frozen per-concept
parameters archived from the pyBKT reproduction in
`data/baselines/pybkt/concept_parameters.csv`. It keeps the same DAG, seed,
targets, and cost model as the two FrequencyOracle conditions. Each target is
planned on its induced prerequisite closure.

The standalone pyBKT reproduction reported the following dynamic prediction
metrics:

| Metric | Value |
|---|---:|
| AUC | 0.663470 |
| Accuracy | 0.670555 |
| RMSE | 0.460752 |
| Training time | 271.47 s |

## Planner adapter

The current Planner collapses repeated attempts into an expected action cost.
The BKT adapter now implements that cost directly: it iterates the BKT belief
after an incorrect response and computes the expected number of attempts until
the first correct response, then multiplies by the 60-second base cost. This
uses `p_init`, `p_learn`, `p_guess`, and `p_slip` instead of treating `p_init`
as a fixed success probability.

The Planner still starts from an empty student history, so this is a
population-level BKT prior rather than a personalized BKT belief for each
student. A fully personalized condition would require passing per-student
answer history or BKT belief state through the Planner.

For transparency, evaluating the BKT first-attempt probability on the 3,103
ECS32A validation samples gives:

| Metric | Value |
|---|---:|
| AUC | 0.591798 |
| Accuracy | 0.586175 |
| RMSE | 0.421051 |
| MAE | 0.374297 |

These numbers are still different from the standalone dynamic pyBKT metrics
above because the Planner starts from an empty history and does not model a
particular student's prior answers.

## Planning result

The same ten targets were used with seed 42:
`[6, 7, 12, 18, 29, 36, 39, 42, 46, 52]`.

| Planning metric | Value |
|---|---:|
| Mean expected total cost | 1144.680158 |
| Mean path length | 11.7 |
| Mean off-target actions | 0.0 |
| Total planning time for 10 targets | 0.0015 s |
| Prerequisite-valid paths | 10 / 10 |

The previous `p_init`-only result was not a faithful BKT cost and has been
replaced by the direct BKT expected-attempt calculation. The new result still
uses a population prior, but now uses all four fitted BKT parameters in the
action cost. Closure restriction removes target-irrelevant actions.

## Files and reproduction

| File | Purpose |
|---|---|
| `data/baselines/pybkt/concept_parameters.csv` | Archived BKT parameters. |
| `experiments/13_run_bkt_greedy.py` | Runs the BKT expected-attempt adapter with Greedy. |
| `results/bkt_greedy/oracle_valid_metrics.csv` | Validation metrics for the adapter. |
| `results/bkt_greedy/planner_trajectories.csv` | Per-target paths and costs. |
| `results/bkt_greedy/summary.json` | Aggregate planning metrics. |

```powershell
.\.venv\Scripts\python.exe experiments\13_run_bkt_greedy.py
```

The experiment completed with valid DAG paths for all ten targets and does not
reference files outside this repository. This group should be reviewed before
being submitted as a final matrix condition.
