# DKT-derived Set Oracle reproduction

This document is the authoritative reproduction record for the ECS32A
**DKT-derived Set Oracle** with Greedy and LAO* planning. The final planner
interface is the deterministic set query `p(v, s)`; neither planner receives a
student identity, chronological prefix, outcome, or DKT hidden state.

## Frozen pipeline

```text
canonical concept sessions and frozen student split
  -> train-only chronological DKT teacher
  -> prefix-level (mastery state, target, teacher probability) examples
  -> grouped deterministic distillation tuples
  -> CPU deterministic set-oracle surrogate
  -> Greedy and LAO* planning
  -> independent public evaluator
```

One DKT event is one canonical concept session. Its skill is the canonical DAG
node and its response is `1 iff session_score >= 0.8`. Token encoding is
`2 * node_index + correctness`, giving 122 input tokens. Every prediction is
made from the complete prefix before the current outcome; only then does that
outcome advance the hidden state. The teacher is an embedding-64,
single-layer LSTM-128 model with a 61-coordinate sigmoid output and learned
empty-prefix logits.

Training is CPU-only and deterministic with seed 42, fixed batch construction,
no shuffling and validation-BCE early stopping. It uses 236 training students
and 25,089 sessions; validation uses 29 students and 3,103 sessions. Test
session rows and outcomes are not inspected: only the frozen split artifact's
test student count is recorded, with `sessions_inspected=false`,
`outcomes_inspected=false`, and `used=false`.

Mastery is irreversible and requires three consecutive session scores at least
0.8. Compression is identical to Task 12, including training-only
zero-observation prerequisite completion. All 27 planning-required targets are
queried at every prefix.

## Commands

Run from the repository root:

```powershell
python experiments\train_dkt_teacher.py
python experiments\build_dkt_distillation_data.py
python experiments\train_dkt_set_oracle.py
python experiments\15_run_dkt_set_greedy.py
python experiments\16_run_dkt_set_lao.py
python experiments\summarize_dkt_set.py
python -m unittest
```

The planning runners load the frozen surrogate and do not retrain it. The LAO*
runner rejects a Greedy reference unless method, run ID, target uniqueness,
closure, manifest, evaluator, split, compression, teacher, distillation,
surrogate-config, and checkpoint identities all match.

Both DKT checkpoints use canonical JSON/base64 tensor serialization despite
their `.pt` suffix; they are not `torch.save` files. Load them through the
repository's frozen loaders. `.gitattributes` fixes these custom `.pt` files
and all hash-sensitive text sources to LF, while Parquet/pickle/NumPy files are
binary. This is required for cross-platform byte-level provenance.

## Teacher and distillation results

The DKT teacher selected epoch 15 and stopped after epoch 35.

| Metric | Train | Validation |
| --- | ---: | ---: |
| BCE | 0.5509275578 | 0.5957606786 |
| AUC (diagnostic) | 0.7864448768 | 0.7400624821 |
| Accuracy (diagnostic) | 0.7191597911 | 0.6793425717 |

| Quantity | Train | Validation |
| --- | ---: | ---: |
| Prefixes | 25,089 | 3,103 |
| Raw tuples | 677,403 | 83,781 |
| Grouped tuples | 297 | 189 |
| Raw mastery states | 866 | 155 |
| Completed states | 11 | 7 |
| Non-empty completed states | 10 | 6 |
| Prefixes changed by completion | 20,113 | 2,456 |

The shared surrogate is exactly `122 -> 128 -> 64 -> 1` with ReLU, ReLU and
Sigmoid. It selected epoch 130 and stopped after epoch 160.

| Metric | Train | Validation |
| --- | ---: | ---: |
| Grouped weighted MSE | 0.0000337389 | 0.0004811091 |
| Prefix-level MSE | 0.0142126719 | 0.0141723517 |
| Prefix-level MAE | 0.0894217144 | 0.0889596517 |
| Pearson correlation | 0.5902324397 | 0.5792326507 |
| Spearman correlation | 0.5761886368 | 0.5697022838 |

State dependence passed the preregistered `1e-6` threshold: packed outputs
differ, maximum effect is 0.1769099832, median per-target maximum effect is
0.0459273905, and the 95th percentile is 0.1520856440. Twenty-four targets have
multiple validation states, all 24 exceed effect 0.01, and 11 exceed 0.05. The
Task 13 decision is therefore **GO**.

## Planner verification

Greedy and LAO* each emit ten standard records, all accepted by the public
evaluator. Independent oracle reloads produce identical planning signatures.
Every LAO* search converged, covered exactly `sequence_nodes`, ended at the
target, and satisfied:

```text
max |J_LAO* - J_DP| = 0.0
max (J_LAO* - J_Greedy) = 0.0
```

LAO* strictly improves eight targets and ties two. Internal costs are measured
under the DKT-derived surrogate. Public evaluation cost and normalized regret
come from the separate FrozenMonotonicOracle and answer performance under that
frozen simulated learner model, not direct student outcomes.

## Frozen identities

| Identity | SHA-256 |
| --- | --- |
| Student split | `516885f2bc972e20f14939f63d1db14423b2745d3e1fc4c2914161bd0b92d435` |
| Concept sessions | `668f16997fb0bd59d16a174188daaec2164412e3b848305d79d4edd75fdbfe07` |
| DKT config | `add41a59081ef6118c5b3f9a96dfcac3ad7e12916702f4b50772d223ca3f9b53` |
| DKT tensor values | `48901d7eeab0580aba90d0fce1d282d300121ea6cbe5abe73791af30299df490` |
| DKT checkpoint artifact | `78f6513776897444445d13d182ec4851baafe84e8c0fbdcd8c4a765a3c26e2d6` |
| Distillation tuple collection | `7e03a1c022f9164fd2c5377ce30b6383e5bdab70c1f2767716a3ac45f2a939c8` |
| Surrogate config | `5333f3064b10c8a25424e49f7030844fc19ad9197519e7ecabe6da6d919d108d` |
| Surrogate checkpoint | `74ee76f29e852b77f3116a6840386342ec256088a245a871a67ff4f4142c012a` |
| Public evaluator | `855a4f721d12d42e8fc685aa3638dc9d00a2d4f0aef67cae1528ac2261568de7` |
| Greedy sequences | `495f2aa7c3013d35abb83a777d89f05624149906be5b13c59f7c1edde2bbb2f2` |
| LAO* sequences | `e889c5df7e7ac2d15492474809489b087569214b3716681e724dc6a9b172087b` |
| LAO* public scores | `f8d2dbb4a7dc888801f140b762d64167900e736b8dd7433c73610acf9fc82f15` |

`results/dkt_set/task13_summary.json` is generated from the artifacts and
performs the final identity and acceptance checks. Planner wall-clock timing in
`planner_comparison.csv` is measured, not a byte-determinism contract; sequence,
cost, exact-DP, convergence, expanded-state and iteration signatures are.
