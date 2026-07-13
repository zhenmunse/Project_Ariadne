# FrequencyOracle + LAO* on ECS32A

## Protocol and interpretation

This condition uses the same shared targets, closures, training artifact, and
state-independent FrequencyOracle as Frequency + Greedy, replacing only the
solver with LAO*.

> The FrequencyOracle internal objective is invariant across valid orderings;
> this pair is retained as a state-independent negative control.

Every target is checked directly against the merged Frequency Greedy standard
records. LAO* and Greedy internal costs must differ by less than `1e-9`.
Different valid sequences may nevertheless receive different costs under the
independent public evaluator.

Each record stores `oracle_state_dependence=false`, negative-control status,
the Greedy internal cost and gap, solver diagnostics, and complete manifest,
evaluator, closure, and training-artifact provenance.

Frequency validation metrics are retained as diagnostics and include the same
artifact/evaluator provenance as the Greedy condition. Internal Frequency
costs must not be compared directly with Ariadne internal or public costs.

## Files and reproduction

| File | Purpose |
|---|---|
| `experiments/12_run_frequency_lao.py` | Runs LAO* and verifies the negative-control invariant. |
| `results/frequency_lao/sequences.jsonl` | Standard records, internal costs, and diagnostics. |
| `results/frequency_lao/oracle_valid_metrics.csv` | Frequency validation and provenance. |
| `results/frequency_lao/scored_sequences.csv` | Canonical evaluator output. |

```powershell
python experiments\09_prepare_oracle_data.py
python experiments\11_run_frequency_greedy.py
python experiments\12_run_frequency_lao.py
python experiments\score_sequences.py `
  results\frequency_lao\sequences.jsonl `
  --output results\frequency_lao\scored_sequences.csv
```
