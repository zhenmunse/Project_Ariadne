# LLM experiment protocol

## Status and scope

This document freezes the LLM-Zero and LLM-Full comparison protocol for
Project Ariadne. It is the human-readable counterpart of
`experiments/llm/protocol.json`. If prose and machine-readable configuration
ever disagree, execution must stop; neither file may be silently preferred.

The LLM conditions are curriculum-sequencing baselines, not agentic planning
systems. They use fresh, stateless, single-turn provider API requests without
tools, search, retrieval, code execution, files, MCP, Connectors, Skills, or
persistent memory. Their outputs are evaluated by the same frozen CPU evaluator
used by the non-LLM conditions.

## Frozen objective and cost

Every concept has the same nominal per-attempt cost:

\[
T_v=60.0\quad\text{for every concept }v.
\]

The objective is **expected uniform-cost attempts**. Because every attempt has
the same nominal cost, minimizing model-based expected cost is equivalent to
minimizing the model-based expected number of attempts. The value 60 has the
unit "nominal seconds per attempt" only to retain the repository's cost scale;
it is not an empirically estimated concept duration.

LLM-Full's aggregate statistics provide evidence that may influence the chosen
ordering. They do not change the objective, concept costs, evaluator, or exact
optimal-cost calculation. All planners and the public evaluator continue to
use `base_cost = 60.0` for every concept.

The paper's Section 3 must describe this as:

> In our experiments, we use a uniform per-attempt cost of 60 seconds, so expected cost is proportional to expected attempt count.

It must not state that the experiments instantiate concept cost using empirical
instructional or completion time.

## Shared curriculum input

Both conditions use the ten targets, prerequisite closures, empty initial
state, DAG, and hashes from `experiments/common/manifest.json`. For each target,
both receive exactly:

- anonymized concepts, consisting of opaque IDs and semantic concept names;
- anonymized prerequisite edges over those opaque IDs.

Opaque mappings and presentation order are fixed once per target and shared
across models, conditions, and repetitions. Original node IDs or order-revealing
name prefixes are not exposed.

LLM-Zero receives only these two curriculum fields. "Zero" means zero student
data, not zero prerequisite structure.

## LLM-Full aggregate fields

LLM-Full receives the same curriculum input and exactly two additional fields
for every concept:

1. `attempt_count`
2. `success_rate`

No other student-derived field is allowed.

### Statistical population and unit

Both fields are computed only from canonical concept sessions belonging to
students in the frozen **training split**. Validation and test students are
excluded. One canonical concept session targeting node \(v\) is one statistical
attempt for \(v\).

For concept \(v\), let \(D_{\mathrm{train},v}\) be its canonical training
concept sessions. Then:

\[
\operatorname{attempt\_count}(v)=|D_{\mathrm{train},v}|.
\]

The frozen correctness rule is:

\[
\operatorname{correct}(j)=\mathbf{1}[\operatorname{session\_score}_j\ge0.8].
\]

For a concept with at least one observation:

\[
\operatorname{success\_rate}(v)=
\frac{\sum_{j\in D_{\mathrm{train},v}}\operatorname{correct}(j)}
     {\operatorname{attempt\_count}(v)}.
\]

Thus the numerator is the number of canonical training concept sessions whose
session score is at least 0.8, and the denominator is the number of canonical
training concept sessions for that concept. This is a binary success fraction,
not the mean continuous session score.

For a zero-observation concept, the only permitted representation is:

```json
{"attempt_count": 0, "success_rate": null}
```

No evaluator prediction, FrequencyOracle mean, pooled prior, neighboring
concept value, or other imputation may replace `null`. Non-null rates are
rendered with exactly four digits after the decimal point in prompts. Counts
are rendered as base-10 integers.

## Explicitly excluded inputs

Neither condition may receive:

- `median_completion_time`;
- `mean_completion_time`;
- empirical per-concept duration;
- raw student histories or interactions;
- student identifiers, timestamps, or trajectories;
- validation/test aggregates;
- mastery-state-conditioned probabilities;
- evaluator or Oracle queries;
- Ariadne sequences, exact-DP paths, or planner descriptions.

Completion-time fields are excluded because they are not part of the frozen
uniform-cost objective. Adding them would define a different experiment and
would require changing and rerunning every planner and evaluator condition.

## Model and run matrix

The frozen logical conditions are:

```text
gpt56_sol_zero
gpt56_sol_full
deepseek_v4_zero
deepseek_v4_full
```

Each is run for 20 fresh API calls on each of the ten manifest targets, for 800
calls total. Exact provider model IDs, response-reported IDs, endpoint versions,
reasoning settings, sampling settings, request IDs, timestamps, token usage,
and latency are frozen with the raw responses.

## Output validity and evaluation

The model returns one JSON object containing a complete opaque-ID sequence.
The parser may trim whitespace, remove one outer code fence, parse one
unambiguous object, and extract `sequence`. It may not retry, repair, complete,
deduplicate, rerank, or use evaluator feedback.

Invalid runs remain invalid raw-run records with null cost and regret. Valid
runs map back to real node IDs and are scored by the shared
`FrozenMonotonicOracle`. Report validity over all calls and cost/regret only
conditional on valid runs. Aggregate repetitions within each target before
averaging equally over the ten targets.

All reported costs and regrets are model-based. Ariadne + LAO* is the
evaluator-matched oracle-optimal reference by construction; this evaluation
does not establish realized human-student outcomes.

## Change control

Before the first API call, freeze the protocol JSON hash, manifest hash, DAG
hash, evaluator hash, training-split hash, aggregate-statistics hash, mapping
hashes, prompt hashes, and provider configurations. A change to the cost,
objective, field set, correctness threshold, split, missing-value rule, or
rendering precision creates a new experiment version and invalidates comparison
with runs produced under this protocol.
