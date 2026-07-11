# FrequencyOracle + Greedy on ECS32A

## Protocol

This condition uses the shared manifest targets and closures. FrequencyOracle
estimates one population-level mean probability per concept from the training
sessions, with the global training mean as fallback. It is explicitly
state-independent:

```text
oracle_state_dependence = false
```

Greedy retains its FrequencyOracle internal cost for diagnosis, but this cost
is not compared numerically with methods that optimize a different Oracle.
Public evaluation cost and normalized regret come exclusively from the common
frozen evaluator.

Each standard sequence record stores the closure, materialized manifest,
evaluator, and Frequency training-artifact hashes. The Oracle metrics artifact
also stores the actual validation and combined split hashes.

## Oracle metrics

FrequencyOracle validation metrics are retained as an Oracle-quality
diagnostic. They are not directly comparable to differently split BKT/DKT
metrics and are not used as public sequence costs.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/11_run_frequency_greedy.py` | Generates standard records and Frequency diagnostics. |
| `results/frequency_greedy/sequences.jsonl` | Method output and internal costs. |
| `results/frequency_greedy/oracle_valid_metrics.csv` | Frequency validation and provenance. |
| `results/frequency_greedy/scored_sequences.csv` | Canonical evaluator output. |

```powershell
python experiments\09_prepare_oracle_data.py
python experiments\11_run_frequency_greedy.py
python experiments\score_sequences.py `
  results\frequency_greedy\sequences.jsonl `
  --output results\frequency_greedy\scored_sequences.csv
```
