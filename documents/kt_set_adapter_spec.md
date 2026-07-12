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

### Frozen KT observation encoding

The observation unit for both BKT and DKT is one canonical concept session as
produced by the session aggregation in `experiments/09_prepare_oracle_data.py`.
It is not a raw item interaction. Each session has:

- `student_id`: the session owner;
- `target_node`: the canonical DAG concept ID obtained from the unique
  item-to-concept mapping;
- `timestamp`: the timestamp assigned by the canonical session aggregator and
  used for chronological ordering;
- `session_score`: the canonical continuous session score in `[0, 1]`;
- `correct`: `1` if and only if `session_score >= 0.8`, otherwise `0`.

The continuous `session_score` is retained for mastery compression. Only the
binary `correct` value is supplied as a response observation to BKT or DKT.
Neither model may reinterpret a fractional score as a fractional Bernoulli
observation or choose a different correctness threshold.

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

Raw observations can violate the prerequisite DAG. They are mapped to a legal
planner state using the frozen zero-observation prerequisite completion rule
below. The same rule is shared by the BKT-derived and DKT-derived adapters.

### Zero-observation prerequisite completion

Let `O` be the set of DAG nodes with at least one canonical **training** concept
session, and let `Z = V - O`. Validation and test sessions must not affect `O`
or `Z`. For a historical prefix `h`, let `M(h)` be the irreversible raw mastery
set produced by the three-consecutive-session rule above.

First retain evidence-supported mastered nodes whose observable ancestors have
also independently reached raw mastery:

```text
R(h) = {
    v in M(h) : ancestors(v) intersect O is a subset of M(h)
}
```

Then add only the globally zero-training-observation ancestors required by
those retained descendants:

```text
s(h) = R(h) union (ancestors(R(h)) intersect Z)
```

Here `ancestors(R)` is the union of the full-DAG transitive ancestor sets of all
nodes in `R`. The result must be checked as a full-DAG order ideal:

```text
for every v in s(h): ancestors(v) is a subset of s(h)
```

The rule has these frozen consequences:

- empty raw mastery maps to the empty state;
- only nodes in `Z` may be structurally completed;
- a node in `O` is never added unless it independently satisfies raw mastery;
- an evidence-supported descendant is removed whenever any observed ancestor
  has not reached raw mastery;
- zero-observation nodes are not mastered at the empty prefix and enter a state
  only when required by a retained evidence-supported descendant;
- unrelated zero-observation nodes are never added.

The adapter may add only globally zero-training-observation ancestors required
by evidence-supported mastered descendants. It must never add an ancestor that
has canonical training observations but has not independently satisfied the
mastery rule.

For the frozen canonical training split, the full-DAG observed set contains 34
nodes and `Z` contains these 27 nodes:

```text
[0, 1, 2, 5, 9, 11, 21, 22, 24, 25, 26, 28, 30, 31, 32, 34, 37,
 45, 48, 49, 50, 51, 53, 54, 58, 59, 60]
```

The SHA-256 of its canonical compact JSON list is:

```text
47ca6d6d085a531a2ce866021b51c4b1bd95f647190e5c676a5c258b33358992
```

The eight required planning nodes `[0, 1, 2, 5, 11, 32, 37, 51]` are the
intersection of this full-DAG `Z` with the union of the ten manifest closures;
they remain the distinct BKT pooled-parameter backoff subset. State completion
and BKT parameter backoff use related evidence audits but are not the same
operation.

The final `s` is serialized as a sorted tuple of canonical integer node IDs and
encoded for the surrogate as a 61-dimensional binary mastery mask in manifest
node order.

## Teacher-probability extraction

### BKT teacher

For every required node with at least one canonical training observation, fit
one concept-specific BKT model using training students only. Required nodes with
zero canonical training observations use the frozen pooled BKT backoff below.
Binary correctness observations are the frozen session-level `correct` values
defined above and are processed chronologically. For each prefix and target
`v`, use the student's current posterior mastery probability for `v` and the
assigned BKT parameters to compute:

```text
p_teacher(v | prefix)
  = P(L_v | prefix) * (1 - slip_v)
    + (1 - P(L_v | prefix)) * guess_v
```

If the student has not attempted `v`, use the fitted initial mastery
probability. Querying a target must not mutate the teacher state; the state is
updated only after the actual next observation is scored.

#### Frozen BKT maximum-likelihood fitting

Concept-specific and pooled models use the same deterministic fitting
algorithm. Each input sequence begins with latent mastery probability `prior`.
For an observation `y` and current mastery probability `L`, compute:

```text
p_correct = L * (1 - slip) + (1 - L) * guess
posterior = L * P(y | mastered) / P(y)
next_L    = posterior + (1 - posterior) * learn
```

where `P(y | mastered)` is `1-slip` for a correct response and `slip` for an
incorrect response, and the analogous unmastered probabilities are `guess` and
`1-guess`. The fitted objective is the sum of negative log predictive
probabilities before each observation. All arithmetic is CPU float64.

Use `scipy.optimize.minimize(method="L-BFGS-B")` with parameter order
`(prior, learn, guess, slip)`, bounds:

```text
prior: [1e-6, 1 - 1e-6]
learn: [1e-6, 1 - 1e-6]
guess: [1e-6, 0.5 - 1e-6]
slip:  [1e-6, 0.5 - 1e-6]
```

and these four deterministic starting points, in order:

```text
(0.20, 0.10, 0.20, 0.10)
(0.50, 0.10, 0.20, 0.10)
(0.20, 0.20, 0.10, 0.10)
(0.20, 0.10, 0.10, 0.20)
```

Set `ftol=1e-12`, `gtol=1e-8`, and `maxiter=2000`. Record every restart result;
discard a restart that does not converge or returns a non-finite parameter or
objective. Select the remaining restart with the lowest final negative log
likelihood; objectives within `1e-12` are tied and the earliest listed restart
wins. Failure of all restarts is a hard error. Sequence and observation ordering
must be deterministic, and the SciPy version is recorded. No random
initialization or additional unrecorded restart is allowed.

#### Pooled zero-observation BKT backoff

Fit one pooled BKT model from all canonical training-student concept sessions.
For parameter estimation only, treat each `(student_id, target_node)` pair as
an independent sequence with skill name `"**pooled**"`. Sort sequences by
`(student_id, target_node)` and retain frozen chronological session order within
each sequence. A student's observations from one concept must never form a
latent trajectory with observations from another concept.

The exact required nodes with zero canonical training observations are:

```text
[0, 1, 2, 5, 11, 32, 37, 51]
```

The 19 required nodes with concept-specific training observations are:

```text
[3, 4, 6, 7, 8, 10, 12, 13, 15, 17, 18, 29, 36, 39, 40, 41, 42, 46, 52]
```

This list is frozen for student split hash
`516885f2bc972e20f14939f63d1db14423b2745d3e1fc4c2914161bd0b92d435`
and concept-session artifact hash
`668f16997fb0bd59d16a174188daaec2164412e3b848305d79d4edd75fdbfe07`.
If either input hash changes, the zero-observation audit must be rerun and this
protocol amended before fitting.

Each zero-observation node uses the pooled parameter vector
`(prior, learn, guess, slip)`, but every `(student, node)` retains an independent
latent mastery posterior initialized from the pooled prior. Updating node `A`
must not alter node `B`, including when both use pooled parameters. Every
required node with one or more canonical training observations must use its
concept-specific parameters and is forbidden from using pooled backoff.

This is the only permitted BKT coverage backoff. Parameters may not be copied
from a neighboring prerequisite, set manually, obtained by averaging fitted
parameter vectors, inferred from the DAG, estimated from validation/test data,
or replaced with FrequencyOracle probabilities. The public condition remains
**BKT-derived Set Oracle**, with metadata:

```text
teacher_parameterization = concept_specific_with_pooled_zero_observation_backoff
```

Define `pooled_parameter_hash` as SHA-256 of canonical sorted-key JSON containing
the selected `(prior, learn, guess, slip)` values, all frozen optimizer settings,
the four restart results, the ordered pooled sequence IDs, their observation
counts, and all training input hashes. The hash field itself is excluded from
its input.

### DKT teacher

The DKT checkpoint must be retrained using the canonical student split in this
document. An existing checkpoint trained under a different or unverifiable
split is not eligible for Task 13, even if it has the same architecture.

One DKT time step is one canonical concept session. The skill is the canonical
DAG `target_node`, translated through a persisted `node_id_to_model_idx`
mapping; it is never the raw item ID. The response is the frozen binary
session-level `correct` value defined above. Sessions are ordered by the frozen
chronology rule, and ties retain the stable source order.

At a prediction point, run the teacher on the complete ordered prefix and
extract the output coordinate corresponding to target `v`. No current or future
outcome may enter that forward pass. DKT sequences are not truncated: the
maximum sequence length is the maximum complete prefix length in the canonical
training/validation data. Mini-batches may right-pad shorter sequences with a
dedicated padding value and an explicit mask; padded steps must not affect the
hidden state, loss, or extracted probability. If the implementation or library
cannot process the observed maximum length, this specification must be amended
and re-approved before training; Task 13 may not silently introduce a truncation
rule. In particular, neither earliest-prefix nor recent-prefix truncation is
currently permitted.

The implementation must include a leakage regression test that changes the
held-out current label while keeping the prefix fixed and verifies that the
extracted teacher probability is unchanged.

The exact DKT checkpoint, configuration, model-index mapping, PyTorch version,
and extraction convention must be hashed or recorded in the artifact metadata.

### Coverage rule

There is no silent target fallback. Each teacher must report the set of DAG
nodes for which it can produce a probability. A condition is eligible for
planning only if it covers every `sequence_node` in all ten manifest closures.

For BKT, concept-specific parameters are required for every required node with
at least one canonical training observation. A required node with zero training
observations must use the frozen pooled zero-observation backoff above. No other
fallback is permitted. The coverage artifact must distinguish
`concept_specific_nodes`, `pooled_backoff_nodes`, and `missing_nodes`; the last
must be empty.

For DKT, unknown mappings and unsupported output coordinates remain hard errors
unless a separate amendment is approved. If full coverage cannot be achieved by
the go/no-go date, use the state-independent fallback described below instead
of silently mixing incompatible probability sources within one condition.

## Distillation examples

For each student prefix, form teacher examples `(student_id, s, v, y)` where
`y = p_teacher(v | prefix)`. Query all covered DAG targets at each retained
prefix so that targets are not restricted to the student's observed next
action. The teacher state is computed once per prefix and target queries are
read-only.

Within each data split, group identical `(s, v)` examples, store the arithmetic
mean teacher probability and the observation count, and sort groups
lexicographically by the mastery bit vector and target ID. Training loss is
weighted by the stored count. This preserves the parameter-dependent part of
the ungrouped squared-error objective and therefore gives identical gradients
and minimizers. The grouped loss value differs from the raw ungrouped loss by a
parameter-independent within-group variance term. Validation must consequently
report both count-weighted grouped surrogate MSE and prefix-level
teacher-versus-surrogate MSE computed against the ungrouped examples.

The distilled dataset artifact must store counts and hashes for its raw source,
mapping, split membership, teacher, DAG, compression configuration, and tuple
table. Its metadata must additionally store:

```text
zero_observation_nodes
zero_observation_nodes_hash
raw_mastery_state_count
completed_state_count
states_changed_by_completion
per_node_completion_frequency
```

`zero_observation_nodes` is the full-DAG training-derived `Z`, not merely the
eight required BKT pooled-backoff nodes. `states_changed_by_completion` counts
prefixes for which `s(h)` differs from raw mastery `M(h)`. Completion frequency
counts how often each node in `Z` is structurally added across prefixes. These
statistics are reported separately for train and validation.

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
- count-weighted grouped surrogate MSE;
- prefix-level teacher-versus-surrogate MSE and MAE on ungrouped examples;
- maximum absolute error and probability range;
- state-effect maximum, median, and 95th percentile;
- hashes of every input and output artifact.

For each target represented in at least two distinct valid validation states,
compute all pairwise absolute effects `|p(v, s1) - p(v, s2)|`. Report the
maximum across all targets and pairs, plus the median and 95th percentile over
the per-target maximum effects. Targets with fewer than two observed states are
reported separately and excluded from the percentile calculation.

The state-dependence regression has two simultaneous requirements:

1. at least one target has two prerequisite-closed validation states whose
   packed outputs differ; and
2. `max_state_effect >= 1e-6`.

The first detects exact collapse to a population prior; the second prevents a
single floating-point ULP from satisfying the July 18 go/no-go rule. These tests
do not claim that the learned function is monotone. The observed effect sizes,
including the maximum, median, and 95th percentile, remain reported experimental
results rather than additional post hoc acceptance thresholds.

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
- compression rule (`threshold=0.8`, `n_consecutive=3`), the
  zero-observation prerequisite-completion rule, and their config hashes;
- full-DAG observed and zero-observation node lists, their hashes, and the
  training-only source used to derive them;
- teacher type, checkpoint/parameter hash, configuration, and index mapping;
- distilled tuple-table hash;
- surrogate architecture/configuration and checkpoint hash;
- Python, NumPy, and PyTorch versions and canonical device (`cpu`).

The BKT teacher additionally produces and hashes:

- `bkt_parameters.json`, containing all concept-specific parameters;
- `pooled_bkt_parameters.json`, containing the pooled parameter vector,
  optimizer settings, restart outcomes, source sequence counts, and pooled
  parameter hash;
- `bkt_coverage.json`, containing all 27 required nodes, the 19
  concept-specific nodes, the exact eight pooled-backoff nodes, an empty missing
  list, coverage fraction `1.0`, backoff rule
  `pooled_zero_observation_bkt`, and the pooled parameter hash.

Every later BKT-derived sequence record stores `bkt_parameter_hash`,
`pooled_bkt_parameter_hash`, and `pooled_backoff_nodes_hash`.

The BKT regression suite must prove that all observed required nodes use
concept-specific parameters, exactly the eight frozen zero-observation nodes
use pooled parameters, pooled fitting contains training students only, and no
other node can enter the backoff list. It must also prove independent
`(student,node)` posteriors, cross-node update isolation, legal probabilities
for all eight pooled nodes, 27/27 coverage, and byte-stable pooled artifacts
across repeated runs.

The shared state-compression regression suite, reused by BKT and DKT, must also
prove:

1. empty raw mastery produces the empty state;
2. a mastered descendant whose complete ancestor chain is in `Z` causes that
   chain, and only that chain, to be completed;
3. a descendant is removed if any ancestor in `O` has not reached raw mastery;
4. when all observed ancestors have reached raw mastery, they are retained and
   any intervening ancestors in `Z` are completed;
5. unrelated nodes in `Z` are never added;
6. every output is prerequisite-closed in the full DAG;
7. `Z` is computed from training sessions only and cannot change when
   validation/test sessions are modified; and
8. repeated state-table construction is byte-identical.

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

## Evaluation interpretation and paper-language constraints

The public evaluator and Ariadne planning oracle are the same frozen
`FrozenMonotonicOracle`. Therefore Ariadne + LAO* normalized regret is zero by
construction, subject only to numerical tolerance: it globally optimizes the
evaluator's own cost function. The main table answers, "If the frozen simulated
learner model is true, how far does each sequencing method deviate from its
optimum?" It does not establish superior outcomes on real students.

The Ariadne + LAO* row must be marked **planner matches evaluator by design**.
The paper must not present its zero regret as an empirical discovery. The
comparative result is the regret incurred by oracle mismatch for Frequency,
BKT-derived, and DKT-derived planners, and by the absence or limitation of
optimization for Random, Syllabus, and LLM conditions.

After Task 15 freezes uniform `T_v = 60`, Section 3 must state that experiments
use a uniform per-attempt cost of 60 seconds, so expected cost is proportional
to expected attempt count. It must not claim that `T_v` is instantiated from
empirical instructional time. Add this Section 3 correction to the Task 18
wording checklist alongside Section 4.

Because canonical oracle inference is deterministic, Section 4 must remove or
rewrite the phrase "even when the underlying predictor uses stochastic
inference" so it does not imply an inference mode absent from the experiments.
