# BKT-derived Set Oracle reproduction

This document is the authoritative reproduction record for the ECS32A
**BKT-derived Set Oracle** and its Greedy and LAO* planning conditions. The
condition is a deterministic set oracle distilled from a chronological BKT
teacher; neither planner receives student identity, a raw history prefix, or a
BKT posterior.

## Frozen protocol

The protocol specification is `documents/kt_set_adapter_spec.md`. The executed
pipeline is:

```text
cleaned interactions + item mapping + prerequisite DAG
  -> canonical concept sessions and student split
  -> concept-specific BKT plus pooled zero-observation backoff
  -> prefix-level (state, target, teacher probability) examples
  -> grouped distillation tuples
  -> deterministic set-oracle surrogate
  -> Greedy and LAO* planning
  -> independent FrozenMonotonicOracle evaluation
```

Canonical session correctness is `1 iff session_score >= 0.8`. Raw mastery is
irreversible and requires three consecutive session scores at least `0.8`.
State compression uses the Task 11 zero-observation prerequisite-completion
rule: only globally zero-training-observation ancestors may be structurally
completed, while an observed ancestor must independently satisfy mastery.

The shared manifest defines ten targets. Their closures require 27 distinct
nodes. Nineteen nodes use concept-specific BKT parameters. Eight nodes with no
canonical training observations use the training-only pooled BKT parameter
vector while retaining independent `(student, node)` latent posteriors:

```text
0, 1, 2, 5, 11, 32, 37, 51
```

Coverage is 27/27 with no fallback outside this declared pooled rule.

## Environment and line-ending contract

The canonical backend is CPU. Training fixes seed 42, enables deterministic
PyTorch algorithms, disables shuffling and dropout, and uses the stopping rule
recorded in `artifacts/bkt_set/surrogate_config.json`. CUDA is not part of the
canonical protocol.

`.gitattributes` fixes common repository text formats to LF and marks Parquet,
pickle and NumPy artifacts as binary. This is required because provenance uses
raw-file SHA-256. After introducing or changing the attributes in an existing
checkout, the repository maintainer should run:

```powershell
git add .gitattributes
git add --renormalize .
```

Do not omit renormalization when preparing a commit. A fresh checkout must
produce the same raw source hashes on Windows, Linux and macOS.

## Reproduction commands

Run from the repository root. The commands are intentionally stage-separated;
each later stage verifies the artifacts produced by the earlier stages.

```powershell
python experiments\kt\prepare_kt_data.py
python experiments\train_bkt_teacher.py
python experiments\build_bkt_distillation_data.py
python experiments\train_bkt_set_oracle.py
python experiments\13_run_bkt_greedy.py
python experiments\14_run_bkt_set_lao.py
python -m unittest
```

The Greedy and LAO* runners never train the surrogate. Both reject any
checkpoint other than the frozen hash below. LAO* also strictly validates that
the Greedy reference has the same manifest, closure, evaluator, split,
compression, teacher, distillation, surrogate-config and checkpoint identity.

## Canonical data statistics

| Quantity | Value |
| --- | ---: |
| Mapped interactions | 70,893 |
| Canonical sessions | 31,559 |
| Students | 294 |
| Train students / sessions | 236 / 25,089 |
| Validation students / sessions | 29 / 3,103 |
| Test students / sessions | 29 / 3,367 |
| Train raw / completed states | 866 / 11 |
| Validation raw / completed states | 155 / 7 |
| Train non-empty completed states | 10 |
| Validation non-empty completed states | 6 |
| Train raw / grouped tuples | 677,403 / 297 |
| Validation raw / grouped tuples | 83,781 / 189 |

Student split sets are pairwise disjoint. Teacher fitting and distillation
training use no test students. The teacher uses each held-out session outcome
only after the prediction point, so the current label cannot leak into its
probability.

## Surrogate and acceptance result

The shared model is exactly `122 -> 128 -> 64 -> 1`, with ReLU, ReLU and
Sigmoid activations. It was selected at epoch 174 and stopped after epoch 204.

| Metric | Train | Validation |
| --- | ---: | ---: |
| Grouped weighted MSE | 0.0000808524 | 0.0020410674 |
| Prefix-level MSE | 0.0105896525 | 0.0127014464 |
| Prefix-level MAE | 0.0583855201 | 0.0648227769 |
| Pearson correlation | 0.8130330585 | 0.7643174703 |
| Spearman correlation | 0.7649989757 | 0.7229487111 |

State-dependence acceptance passed:

| Diagnostic | Value |
| --- | ---: |
| Packed outputs differ | true |
| Maximum state effect | 0.2983331680 |
| Median per-target maximum effect | 0.0556732714 |
| 95th percentile per-target maximum effect | 0.2316713452 |
| Targets with multiple valid states | 24 |
| Targets with effect at least 0.01 / 0.05 | 24 / 12 |
| Prespecified minimum effect | 0.000001 |

The go/no-go decision is therefore **GO**.

## Planner verification

Both conditions emit ten standard `SequenceRecord` objects and are scored by
the independent shared evaluator. All twenty records are valid. Independent
Oracle instances produce identical planning signatures.

For BKT-set LAO*:

- all ten searches converged;
- every sequence covers exactly its manifest `sequence_nodes` and ends at the
  target;
- `max |J_LAO* - J_DP| = 0.0`;
- `J_LAO* <= J_Greedy + 1e-9` for every target;
- eight of ten LAO* sequences differ from the Greedy sequence.

Internal cost is the BKT-set surrogate objective. Public evaluation cost and
normalized regret come from the separate FrozenMonotonicOracle evaluator. They
answer how a sequence performs if that frozen evaluation model is true; they
are not direct empirical student outcomes.

## Frozen identity

The custom checkpoint is canonical JSON/base64 tensor data despite its `.pt`
suffix. It is not a `torch.save` file and must be loaded through
`BKTSetOracle.from_artifacts()`.

| Identity | SHA-256 |
| --- | --- |
| Student split | `516885f2bc972e20f14939f63d1db14423b2745d3e1fc4c2914161bd0b92d435` |
| Concept sessions | `668f16997fb0bd59d16a174188daaec2164412e3b848305d79d4edd75fdbfe07` |
| BKT parameter values | `6a773d04b9ccd12f4d32736c1b5860ccc3cbc8f1ad20eb577ee655d205bf8f3f` |
| BKT parameter artifact | `d3616777af5d48b8f5bc18d37e3ad11eb892b4c98a1ea52f2283baf8f2f5cd98` |
| Pooled parameter vector | `b91780a500397f5fb457aab4707f7d5769dee5a7da373049bae5fae04c2cd503` |
| Pooled parameter artifact | `5ca92b3ccb665d45f28435b51d305ba8f5e33ba4a64f0f29697f8e97e258245c` |
| Distillation tuple collection | `82a940ecaa922f66e9bb104cf3037acc3f6296920b7fc7ebdb3cdf011b8920c8` |
| Surrogate config | `182b079e2f068fb96434f33e909cbf5b46349e1d1290c757f6ae96917d755778` |
| Surrogate checkpoint | `4a4ae471e06dbeeea46bf09f0502f39455576ccdd7f992e0184912cac7b60791` |
| Evaluator source | `855a4f721d12d42e8fc685aa3638dc9d00a2d4f0aef67cae1528ac2261568de7` |
| Greedy sequences | `b5cb5dbcafa233a0072a232b8e88dc79b52ae8790c9abf64132f809e91409fa6` |
| LAO* sequences | `aab0f38581271eb721520922f27c594c04f66e3a889a7d9f44ec14d7cc2ed73d` |
| LAO* public scores | `303beaf2dc20e861828a4bac0198b200bdab246b245694a8cd85b2bd839be02c` |

`results/bkt_set/planner_comparison.csv` contains measured wall-clock runtime;
its file hash is intentionally not a determinism contract. The deterministic
contract covers sequences, IEEE-754 internal costs, exact-DP costs, convergence,
expanded-state counts and iteration counts.

## Artifact inventory

| Stage | Canonical artifacts |
| --- | --- |
| Preprocessing | `data/kt_set/student_split.json`, `data/kt_set/concept_sessions.parquet`, `data/kt_set/preprocessing_manifest.json` |
| BKT teacher | `artifacts/bkt_set/bkt_parameters.json`, `artifacts/bkt_set/pooled_bkt_parameters.json`, `artifacts/bkt_set/bkt_coverage.json`, `artifacts/bkt_set/bkt_teacher_metadata.json` |
| Distillation | `artifacts/bkt_set/train_prefix_examples.parquet`, `artifacts/bkt_set/validation_prefix_examples.parquet`, `artifacts/bkt_set/train_grouped_tuples.parquet`, `artifacts/bkt_set/validation_grouped_tuples.parquet`, `artifacts/bkt_set/distillation_metadata.json` |
| Surrogate | `artifacts/bkt_set/surrogate_config.json`, `artifacts/bkt_set/surrogate_checkpoint.pt`, `artifacts/bkt_set/surrogate_metrics.json` |
| Diagnostics | `results/bkt_set/oracle_metrics.csv`, `results/bkt_set/state_dependence.csv`, `results/bkt_set/planner_comparison.csv` |
| Greedy | `results/bkt_set_greedy/sequences.jsonl`, `results/bkt_set_greedy/scored_sequences.csv` |
| LAO* | `results/bkt_set_lao/sequences.jsonl`, `results/bkt_set_lao/scored_sequences.csv` |

Legacy history-dependent `bkt_lao` runners and results are not part of this
condition and must not be restored or aggregated with these artifacts.
