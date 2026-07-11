# Ariadne + LAO* on ECS32A

## Protocol

This condition uses the shared manifest, canonical CPU
`FrozenMonotonicOracle`, and the repository's LAO* planner. Targets and closure
graphs come exclusively from `experiments/common/manifest.json`. The heuristic
uses the certified bound `p_bar(v)=1` for every node; it does not query an
all-mastered stochastic estimate.

Each standard sequence record stores the materialized manifest, evaluator,
Oracle checkpoint, and closure hashes. The validation metrics artifact also
stores the DAG, combined split, and exact `valid_sessions.pkl` hashes.

## Oracle validation modes

| Metric | Full feature | Planning mode (`x=0`) |
|---|---:|---:|
| Samples | 3,103 | 3,103 |
| Binary samples | 2,170 | 2,170 |
| AUC | 0.774873 | 0.610952 |
| Accuracy | 0.798618 | 0.402765 |
| RMSE | 0.351672 | 0.500678 |
| MAE | 0.303330 | 0.436528 |

The planner uses planning mode. Full-feature validation is reported only as a
separate predictor diagnostic.

## LAO*, exact DP, and common scoring

| Metric | Value |
|---|---:|
| Targets | 10 |
| LAO* converged | 10 / 10 |
| `abs(J_LAO - J_DP) < 1e-9` | 10 / 10 |
| Maximum observed LAO*/DP gap | 0.0 |
| Valid sequences | 10 / 10 |
| Normalized regret equal to 0.0 | 10 / 10 |
| Mean evaluation cost | 1789.668001 |
| Mean expanded states | 21.9 |
| Mean iterations | 22.9 |

Zero normalized regret is expected by design: Ariadne + LAO* plans against
the same frozen learner model used by the public evaluator. It is a protocol
consistency result, not evidence of performance on real students.

Two independent runs with newly constructed CPU Oracles produced identical
sequences, LAO* values, DP values, expanded-state counts, and iteration counts.
Runtime metadata remains diagnostic.

## Cross-condition provenance

Ariadne + Greedy and Ariadne + LAO* record identical values for:

```text
manifest_hash:
f03496e703b4e6a04ad46f24c42e55d92bdc88266d92dbd19d77d2892d3c8cad

oracle_checkpoint_hash:
f3ba106238d7113c22705691b51f68b2212b18ce09fc7d9447d2fe9ab31b867a

evaluator_hash:
b0e749046b8f4437bc92f358fe3a644aba4614fc304a3c49168c9a9d8b77a355
```

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/18_run_ariadne_lao.py` | Runs LAO*, exact DP, determinism checks, and both validation modes. |
| `results/ariadne_lao/sequences.jsonl` | Standard method output records and solver diagnostics. |
| `results/ariadne_lao/oracle_valid_metrics.csv` | Validation modes and complete provenance. |
| `results/ariadne_lao/scored_sequences.csv` | Independent common-scorer output. |

```powershell
python experiments\09_prepare_oracle_data.py
python experiments\18_run_ariadne_lao.py
python experiments\score_sequences.py `
  results\ariadne_lao\sequences.jsonl `
  --output results\ariadne_lao\scored_sequences.csv
```
