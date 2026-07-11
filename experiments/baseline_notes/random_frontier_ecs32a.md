# Random Frontier Policy on ECS32A

## Protocol

At each step this baseline samples uniformly from the currently available
prerequisite-valid actions. This produces a valid topological ordering, but it
is **not** a uniform sample over all linear extensions of the closure DAG.

Targets, closures, initial state, and base seed come from the shared manifest.
For target index `i` and repetition `run_id`, the seed is:

```text
manifest_seed + i * 1000 + run_id
```

Each of the ten targets has 100 repetitions. The method generates sequences
only; it has no internal Oracle and therefore records `internal_cost=null`.
All costs and regrets come from the independent canonical scorer.

Every sequence record stores its seed, closure hash, materialized manifest
hash, and evaluator hash. Re-running the generator in the same invocation
produces an identical set of 1,000 records before anything is written.

## Common-scorer results

| Statistic | Normalized regret |
|---|---:|
| Records | 1,000 |
| Valid records | 1,000 / 1,000 |
| Mean | 0.000263322 |
| Median | 0.000256504 |
| Population standard deviation | 0.000214417 |
| 5th percentile | 0.0 |
| 95th percentile | 0.000658391 |
| Best | 0.0 |
| Worst | 0.000755476 |

The small regrets describe behavior under the frozen simulated learner model;
they are not estimates of real-student performance. Per-target statistics are
stored in `regret_summary.csv`.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/19_run_random_frontier.py` | Generates 1,000 deterministic Random Frontier records. |
| `experiments/19_summarize_random_frontier.py` | Summarizes common-scorer regret. |
| `results/random_frontier/sequences.jsonl` | Standard method output. |
| `results/random_frontier/scored_sequences.csv` | Canonical evaluator output. |
| `results/random_frontier/regret_summary.csv` | Overall and per-target statistics. |

```powershell
python experiments\19_run_random_frontier.py
python experiments\score_sequences.py `
  results\random_frontier\sequences.jsonl `
  --output results\random_frontier\scored_sequences.csv
python experiments\19_summarize_random_frontier.py `
  results\random_frontier\scored_sequences.csv `
  --output results\random_frontier\regret_summary.csv
```
