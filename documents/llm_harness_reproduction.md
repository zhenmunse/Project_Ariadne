# LLM harness reproduction

## Scope

This document describes the Task 16 harness that prepares and executes the
Project Ariadne LLM baselines. Task 16 validation uses only the deterministic
mock provider. Formal provider calls and the resulting 800-response dataset
belong to Task 17.

The human protocol is `documents/llm_experiment_protocol.md`. Its cost and
input contract is machine-readable in `experiments/llm/protocol.json`.
Execution, retry, parser, aggregation and provenance rules are frozen in
`experiments/llm/run_config.json`.

## Provider configuration

The two logical model families are configured under:

```text
run_config.json -> models.closed_frontier
run_config.json -> models.open_weight
```

Before Task 17, each entry must have a provider-verified exact:

- `requested_model_id`;
- `endpoint`;
- native `reasoning` setting;
- supported sampling configuration.

Task 16 intentionally leaves model IDs and endpoints null and marks them
`must_be_frozen_before_task17`. Both formal adapters fail their capability gate
while either value is unset. No request can be sent in that state.

API secrets are read only from these environment variables:

```text
OPENAI_API_KEY
DEEPSEEK_API_KEY
```

Secrets, authorization headers, cookies and API-key fields are forbidden in
canonical artifacts. The adapters pass authorization directly to the HTTP
transport; it is never included in request or response artifacts.

`closed_frontier.py` uses a Responses-style single-turn payload.
`open_weight.py` uses a chat-completions-style single-turn payload. Provider
API documentation and payload compatibility must be reverified when the exact
Task 17 endpoints are frozen. Both adapters expose the same typed
`LLMProvider.complete(ProviderRequest) -> ProviderResponse` interface and save
the response-reported model ID and complete JSON provider payload.

## Deterministic preparation and dry-run

From the repository root:

```powershell
python experiments\llm\run_llm.py --dry-run
```

Dry-run performs no network access and creates no response artifacts. It
rebuilds:

```text
experiments/llm/generated/mappings.json
experiments/llm/generated/full_statistics.json
experiments/llm/generated/prompt_manifest.json
experiments/llm/generated/run_manifest.json
experiments/llm/generated/prompts/<target>/<condition>.json
```

The run manifest contains exactly:

```text
2 model families x 2 conditions x 10 targets x 20 repetitions = 800 runs
```

Running dry-run twice must leave every generated JSON byte-identical.

## Mock execution

Mock mode never accesses the network. Use a temporary or explicitly mock-only
output root so test responses cannot be confused with formal Task 17 data:

```powershell
python experiments\llm\run_llm.py `
  --provider mock `
  --model closed_frontier `
  --condition zero `
  --target 42 `
  --run-id 7 `
  --output-root local\llm_mock
```

The test suite exercises valid, code-fenced, outer-text, invalid-JSON,
duplicate, missing, unknown-ID, prerequisite-violation, target-not-final,
empty-response, simulated 429 and simulated timeout behavior. It also completes
the full 800-run matrix in a temporary directory.

## Formal subset commands

These commands remain blocked until the selected provider's model ID and
endpoint pass the capability gate:

```powershell
python experiments\llm\run_llm.py `
  --provider closed_frontier `
  --condition zero `
  --target 42 `
  --run-id 7

python experiments\llm\run_llm.py `
  --provider open_weight `
  --condition full `
  --target 42
```

Filters never change logical identity. A run key remains:

```text
model_key/condition/target_node/run_id
```

## Request isolation and retry policy

Every logical run sends one system prompt and one user prompt in a fresh,
single-turn request. No previous response, thread, conversation, tool, search,
retrieval, file search or code-execution facility is registered. Native
reasoning is requested through provider parameters, not through a request for
visible chain of thought.

Transport retry and experimental retry are distinct:

- a retryable transport failure before any provider response may create a new
  attempt under the same logical run key;
- every transport attempt is durably recorded;
- once a provider response is obtained, the logical run ends, even if parsing
  or sequence validation fails;
- invalid model output is never repaired, supplemented, reordered or sent back
  to the model.

The default maximum transport attempt count is three.

## Artifact layers

Each attempt uses three append-only paths:

```text
results/llm/requests/<logical_run_key>/<attempt>.json
results/llm/raw/<logical_run_key>/<attempt>.json
results/llm/parsed/<logical_run_key>/<attempt>.json
```

Request artifacts contain the exact prompts, requested model configuration,
logical identity and source hashes. Raw artifacts contain the verbatim response
text, response-reported model ID, provider request ID, usage, finish reason,
latency and complete provider payload. Parsed artifacts bind the raw byte hash
to parser and structural-validation results.

Only structurally valid parsed runs may later become canonical
`SequenceRecord` objects. Invalid runs remain parsed all-run records with no
evaluation cost or regret.

## Resume and force-rerun

Normal execution is resume-safe:

- a parsed attempt is skipped without a provider call;
- a durable raw response lacking a parsed artifact is parsed locally without a
  provider call;
- an explicitly recorded retryable transport error may be retried;
- a dispatched request with no durable raw response is ambiguous and fails
  closed, because automatically repeating it could duplicate billing.

Use `--force-rerun` only after an explicit audit decision. It creates a new
numbered attempt and never overwrites old request, raw or parsed artifacts. The
first successfully returned provider response remains
`selected_for_analysis=true`; later forced attempts are preserved for audit but
are not silently substituted.

## Crash recovery contract

The test suite simulates crashes:

- before request persistence: no artifact exists and a later run is safe;
- after provider return but before raw persistence: resume fails closed as
  ambiguous;
- after raw persistence but before parsing: resume parses locally;
- after parsed persistence: resume skips the completed attempt.

All JSON writes use flush, fsync and atomic rename.

## Tests

Run the Task 16 suite:

```powershell
python -m unittest `
  tests.test_llm_protocol `
  tests.test_llm_anonymization `
  tests.test_llm_statistics `
  tests.test_llm_prompts `
  tests.test_llm_parser `
  tests.test_llm_harness `
  tests.test_llm_provider_contract `
  tests.test_llm_runtime
```

Then run the full repository suite:

```powershell
python -m unittest
```

## Task 17 freeze checklist

Before the first formal request:

- merge the reviewed Task 16 harness and record the repository commit;
- set and review exact provider endpoints and requested model IDs;
- record response-reported model identifiers in smoke tests that are explicitly
  excluded from analysis;
- verify both capability reports are ready;
- verify native reasoning and accepted sampling parameters against current
  provider documentation;
- verify no tools, previous response IDs or storage are enabled;
- freeze protocol, run-config, manifest, DAG, split, statistics, mapping,
  prompt, parser and adapter hashes;
- rerun dry-run twice and confirm byte-identical inputs;
- ensure the formal `results/llm` root contains no mock data;
- configure API keys only through environment variables;
- record the UTC execution window and preserve every request ID and raw payload.

Any prompt, mapping, objective, model version, provider configuration, parser or
scoring change after comparative results are inspected requires a new declared
experiment version.
