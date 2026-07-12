# KT Set-Oracle Adapter Specification

## Status and scope

This document freezes the shared distillation protocol used to convert a
history-dependent knowledge-tracing (KT) teacher into a deterministic oracle
over planner states. The two resulting experimental oracles are named:

- **BKT-derived Set Oracle**
- **DKT-derived Set Oracle**

Tasks 12 and 13 must use this protocol without condition-specific changes to
the state representation, student split, surrogate architecture, query API, or
acceptance tests. The BKT and DKT teachers may differ internally, but everything
after teacher-probability extraction is shared.

The adapter is a distillation layer. It does not expose a BKT posterior, a DKT
hidden state, or an ordered interaction history to the planner.

## Frozen mathematical interface

For a DAG node `v` and a prerequisite-closed mastery set `s`, the adapter
returns one scalar:

```text
p(v, s) in (0, 1]
```

The planner-facing call is a pure function of `(v, s)`. The base attempt cost
is read from the shared materialized manifest. The heuristic upper bound is
fixed to:

```text
best_case_success_prob(v) = 1.0
```

This bound must not be estimated from the teacher or surrogate.

## Canonical student split

The split unit is the student, never an interaction, prefix, or distilled
tuple. Starting from `data/processed/cleaned_interactions.csv`, obtain unique
student IDs in the same first-occurrence order used by
`experiments/09_prepare_oracle_data.py`, shuffle them once with NumPy
`default_rng(42)`, and assign:

```text
split_size = floor(0.1 * number_of_students)
test       = users[0 : split_size]
validation = users[split_size : 2 * split_size]
train      = users[2 * split_size :]
```

The implementation must persist a split artifact containing the three exact
student-ID lists and its SHA-256. Reproducing the algorithm without persisting
the membership is not sufficient provenance.

Teacher fitting and surrogate fitting use training students only. Validation
students may be used for checkpoint selection, calibration decisions, and
acceptance metrics. Test students remain untouched until final evaluation.
All prefixes and all tuples derived from one student remain in that student's
split.

## Historical prefix and chronology

For each student, sort interactions by the cleaned timestamp and then by a
stable source-row index to resolve equal timestamps. A prefix contains only
interactions strictly before the prediction point. The current outcome must
never be included in the teacher state used to predict that outcome.

An empty prefix may be used only if the teacher defines an auditable initial
prediction. Otherwise it is excluded for both teachers and the exclusion count
is recorded.

## Prefix-to-mastery compression

The compression rule uses the existing project configuration:

```text
mastery threshold = 0.8
required consecutive concept sessions = 3
```

First map interactions to DAG nodes using
`data/question_concept_mapping_final.csv` and aggregate them into the same
chronological concept sessions used by the canonical preprocessing pipeline.
For each concept, maintain its own sequence of session scores.

A concept becomes raw-mastered at the first prefix for which its last three
observed session scores are all at least `0.8`. Mastery is irreversible: once a
concept enters the raw mastery set, later evidence cannot remove it. This is
required because the SSP state is monotone.

Raw observations can violate the prerequisite DAG. They are therefore mapped
to the largest prerequisite-closed subset without inventing mastery:

```text
s = {v in raw_mastered : every ancestor of v is in raw_mastered}
```

The adapter must not repair a state by adding unobserved ancestors. The final
`s` is serialized as a sorted tuple of canonical integer node IDs and encoded
for the surrogate as a 61-dimensional binary mastery mask in manifest node
order.

## Teacher-probability extraction

### BKT teacher

Fit one BKT model per mapped concept using training students only. Binary
correctness observations are processed chronologically. For each prefix and
target `v`, use the student's current posterior mastery probability for `v`
and the fitted BKT parameters to compute:

```text
p_teacher(v | prefix)
  = P(L_v | prefix) * (1 - slip_v)
    + (1 - P(L_v | prefix)) * guess_v
```

If the student has not attempted `v`, use the fitted initial mastery
probability. Querying a target must not mutate the teacher state; the state is
updated only after the actual next observation is scored.

### DKT teacher

Train or load the frozen DKT model using the canonical student split and a
recorded node-to-model-index mapping. At a prediction point, run the teacher on
the complete ordered prefix and extract the output coordinate corresponding to
target `v`. No current or future outcome may enter that forward pass. The
implementation must include a leakage regression test that changes the held-out
current label while keeping the prefix fixed and verifies that the extracted
teacher probability is unchanged.

The exact DKT checkpoint, configuration, model-index mapping, PyTorch version,
and extraction convention must be hashed or recorded in the artifact metadata.

### Coverage rule

There is no silent target fallback. Each teacher must report the set of DAG
nodes for which it can produce a probability. A condition is eligible for
planning only if it covers every `sequence_node` in all ten manifest closures.
Missing nodes, unknown mappings, and unsupported output coordinates are hard
errors. If full coverage cannot be achieved by the go/no-go date, use the
state-independent fallback described below instead of mixing teacher and
fallback probabilities within one condition.

## Distillation examples

For each student prefix, form teacher examples `(student_id, s, v, y)` where
`y = p_teacher(v | prefix)`. Query all covered DAG targets at each retained
prefix so that targets are not restricted to the student's observed next
action. The teacher state is computed once per prefix and target queries are
read-only.

Within each data split, group identical `(s, v)` examples, store the arithmetic
mean teacher probability and the observation count, and sort groups
lexicographically by the mastery bit vector and target ID. Training loss is
weighted by the stored count, making aggregation mathematically equivalent to
training on the ungrouped examples while removing input-order dependence.

The distilled dataset artifact must store counts and hashes for its raw source,
mapping, split membership, teacher, DAG, compression configuration, and tuple
table.

## Shared surrogate

Both derived oracles use the same surrogate definition:

```text
input  = concat(61-bit mastery mask, 61-way target one-hot vector)
MLP    = Linear(122, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 1)
output = sigmoid(logit)
loss   = count-weighted mean squared error against teacher probability
```

Training uses CPU, seed 42, deterministic PyTorch algorithms, deterministic
data ordering, and no dropout. Model selection uses validation loss with a
fully recorded stopping rule. The selected state dict is loaded with
`map_location="cpu"`, set to evaluation mode, and has gradients disabled.

The returned probability is converted to a Python float and clamped to
`[1e-12, 1.0]` before use in geometric cost. The implementation must reject an
unknown target, a non-integer node ID, a mastery node outside the DAG, or a
mastery set that is not prerequisite-closed.

The surrogate receives no ordered sequence, timestamp, attempt count, score
history, BKT belief, DKT hidden state, student identifier, or teacher-specific
feature. Consequently two historical prefixes compressed to the same `(v, s)`
must yield the same adapter output.

## Determinism contract

The canonical backend is CPU. The following comparisons use packed IEEE-754
bytes, not approximate equality:

1. repeated calls for the same `(v, s)` on one object are identical;
2. querying the same table in different orders produces an identical table;
3. two independently constructed oracle objects produce identical
   probabilities;
4. serialization and reloading do not change any tested probability.

Any cache must be keyed only by canonical `(v, sorted(s))`; the tests must also
pass with an empty cache and after reconstruction.

## Validation and acceptance

Report, separately for train and validation students:

- number of students, prefixes, unique mastery states, and unique `(s, v)`
  pairs;
- target coverage and per-target example counts;
- teacher-versus-surrogate MSE and MAE;
- maximum absolute error and probability range;
- hashes of every input and output artifact.

The state-dependence regression must find at least one target `v` and two valid,
distinct prerequisite-closed states `s1` and `s2` in the validation query table
such that the surrogate's packed outputs differ. This proves that the adapter
has not collapsed to a population prior; it does not claim monotonicity.

The deterministic-query tests above and full target-coverage check are hard
requirements. Tasks 12 and 13 may add teacher-specific quality diagnostics, but
may not weaken these shared requirements.

## Provenance

Every surrogate checkpoint, metrics file, and generated sequence record must
identify at least:

- materialized manifest hash and closure hash;
- DAG hash and evaluator hash;
- cleaned-interaction and question-to-concept mapping hashes;
- canonical split artifact hash;
- compression rule (`threshold=0.8`, `n_consecutive=3`) and its config hash;
- teacher type, checkpoint/parameter hash, configuration, and index mapping;
- distilled tuple-table hash;
- surrogate architecture/configuration and checkpoint hash;
- Python, NumPy, and PyTorch versions and canonical device (`cpu`).

Sequence metadata uses the exact public condition names
`BKT-derived Set Oracle` and `DKT-derived Set Oracle` in addition to the shared
schema method enum selected by Tasks 12 and 13.

## July 18, 2026 go/no-go rule

By July 18, each derived oracle must satisfy full closure-target coverage, the
state-dependence regression, and every deterministic-query test. Failure of any
one is a no-go for that derived set-oracle condition.

The fallback is a separately named state-independent population-prior oracle:
estimate one training-only success rate per node, apply a predeclared smoothing
rule, and return that value for every state. Do not label this fallback as a
BKT-derived or DKT-derived Set Oracle, and do not combine its probabilities
with a partially working KT adapter. Its state independence must be stated in
results metadata and treated as a negative-control condition.

