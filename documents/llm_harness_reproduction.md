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

Task 17 preflight freezes OpenAI as `gpt-5.6-sol` at
`https://api.openai.com/v1/responses` with no multi-agent mode, and DeepSeek as
`deepseek-v4-pro` at `https://api.deepseek.com/chat/completions` with thinking
enabled. The original candidate used the highest reasoning setting and 4096
completion tokens. A preregistered pilot showed that DeepSeek consumed all 4096
tokens as reasoning and returned an empty final response even on the smallest
closure. The effective configuration therefore uses `medium` reasoning and
32,768 completion tokens for both models. This remains within the protocol's
highest-stable-setting rule: a setting producing budget-exhausted empty outputs
is not stable. Both omit temperature and top-p. A formal request remains prohibited until the account-level smoke test
also confirms access and the response metadata contract.

The medium/16,384 DeepSeek curriculum pilots ended naturally with finish reason
`stop` for both the smallest and largest closures. They remain excluded pilot
evidence. The hard ceiling was then raised to 32,768 for both providers to
protect repeated runs from upper-tail truncation without changing the medium
reasoning setting. Each provider must complete a new configuration-bound smoke
before formal execution. Smoke requests are marked `smoke_test=true` and
`excluded_from_analysis=true`. Restricted raw artifacts are ignored by Git;
`experiments/llm/generated/provider_preflight.json` records their hashes and
sanitized audit metadata.

DeepSeek's 32,768 ceiling verification is explicitly inherited from its
medium/16,384 generic smoke and its two naturally completed curriculum pilots.
No additional API call was made for the ceiling-only increase. The preflight
records `verification_basis=inherited_nonbinding_ceiling_increase` and the
verified lower request ceiling, rather than claiming an exact 32,768 smoke.

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

Input regeneration requires the repository's full data environment, including
`pandas` and parquet support. The normal execution path loads the frozen JSON
inputs and does not import those preparation-only dependencies, so `--help`
and an individual provider or mock run work from any Python installation that
has the lightweight runtime dependencies installed.

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
  --single-run `
  --provider closed_frontier `
  --model closed_frontier `
  --condition zero `
  --target 42 `
  --run-id 7

python experiments\llm\run_llm.py `
  --provider open_weight `
  --condition full `
  --target 42
```

Use `--single-run` for an individually controlled formal experiment. This
safety mode requires explicit `--provider`, `--model`, `--condition`,
`--target`, and `--run-id` values and aborts unless they resolve to exactly one
entry in the frozen run manifest. For example, the completed DeepSeek smoke-in
formal-grid invocation is reproducible with:

```powershell
D:\anaconda3\python.exe experiments\llm\run_llm.py `
  --single-run `
  --provider open_weight `
  --model open_weight `
  --condition zero `
  --target 6 `
  --run-id 0
```

Omit `--single-run` only when a reviewed batch selection is intentional. Batch
filters may select multiple targets or repetitions.

## Concurrent batch execution

Batch runs can overlap independent API waits with `--workers`. The default is
`1`; a conservative initial formal setting is `5`, increased only after
observing provider rate limits and account quotas. For example, all pending
DeepSeek Zero runs can be resumed with:

```powershell
D:\anaconda3\python.exe experiments\llm\run_llm.py `
  --provider open_weight `
  --condition zero `
  --only pending `
  --workers 5
```

The scheduler rejects non-positive worker counts and duplicate logical run
identities. Threads share only the stateless provider adapter; each logical run
retains a distinct request/raw/parsed path and its existing retry, resume,
first-success, and invalid-no-retry semantics. Do not launch overlapping shell
processes over the same run range. Provider HTTP 429 responses remain governed
by the frozen transport-retry policy.

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

Each attempt uses three attempt-indexed paths:

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

Within one attempt, the request artifact is atomically updated from
`request_prepared` to `request_dispatched` and, when applicable, to
`transport_error`. It is never overwritten across attempt numbers. Raw and
parsed artifacts are written once per attempt and are the strictly append-only
response layers.

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
