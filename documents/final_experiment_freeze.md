# Final experiment freeze

`results/final/final_freeze_manifest.json` is the single provenance root for
the paper's experimental results. It is generated with:

```powershell
python experiments/finalize_all_results.py
```

The finalizer reads the approved Tasks 6--17 artifacts and writes only
`results/final/`. It does not rerun planners, train an oracle, or call an LLM.
Running it twice without changing an input must leave every output byte
identical.

## Final comparison contract

The aggregate contains 14 method/condition rows: eight Greedy/LAO* planner
conditions, Random Frontier, Linear Syllabus, GPT-5.6 SOL Zero/Full, and
DeepSeek V4 Pro Zero/Full. Structurally invalid LLM responses and the declared
transport-ambiguous run are excluded from sequence scoring but retained in
`all_run_status.jsonl` and `validity_table.csv`.

Cost and normalized regret are computed only for structurally valid sequences.
Repetitions are aggregated within target first; the ten targets are then given
equal weight. The main table's median, standard deviation, p05, and p95 are the
distribution of the ten target-level mean normalized regrets.

`validity_rate` and `pipeline_yield` use all planned runs as their denominator.
`model_validity_rate` is additionally retained for LLM auditability and uses
only obtained provider responses as its denominator.

## Formal LLM configuration

- Models: GPT-5.6 SOL and DeepSeek V4 Pro.
- Conditions: Zero and Full.
- Reasoning: provider-native medium.
- Completion-token ceiling: 32768.
- Temperature and top-p: omitted.
- Repetitions: 20 per model, condition, and target.
- Formal terminal counts: 800 planned, 799 provider responses, 792 valid
  sequences, 7 model-invalid responses, and 1 transport-ambiguous run.

Smoke and pilot artifacts are explicitly excluded from formal analysis. The
manifest binds the formal request, raw-response, and parsed-response
collections separately.

## Statistical contract

The two prespecified comparisons are GPT Full versus GPT Zero and DeepSeek
Full versus DeepSeek Zero. For each model, the sample is the ten paired target
means, not the individual repetitions. `statistical_tests.csv` reports the
Full-minus-Zero paired mean difference, an exact two-sided sign-permutation
test, a two-sided Wilcoxon signed-rank test, a deterministic 100,000-replicate
paired bootstrap interval, Cohen's dz, matched-pairs rank-biserial effect size,
and Holm-adjusted p-values across the two prespecified comparisons.

## Commit identity

The manifest records the repository `HEAD` resolved when the finalizer runs as
`final_code_commit_sha`, together with direct hashes of the finalizer,
aggregator, and public evaluator. After the final Task 18 code commit is
created, rerun the command once so the recorded commit identity names that
frozen source snapshot.
