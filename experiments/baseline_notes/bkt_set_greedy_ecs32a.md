# BKT-derived Set Oracle + Greedy on ECS32A

## Condition

This condition uses the deterministic **BKT-derived Set Oracle** approved in
Task 12-3. Historical student prefixes and BKT posteriors are not available to
the planner. At every step, Greedy queries only the current prerequisite-closed
mastery set and candidate node `(v, s)`.

The runner does not train or modify the surrogate. It hard-requires checkpoint
SHA-256:

```text
b00a8184babd0280f979af41d1403c7c0ea0fe4b4bb70c05c71be3fb5ccff920
```

The checkpoint uses the project's custom deterministic tensor format, not the
`torch.save` format; load it through `BKTSetOracle.from_artifacts()` rather than
`torch.load()`.

## Planning and evaluation

Greedy selects the currently valid action with minimum immediate BKT-set cost
`60 / p(v, s)`, with node ID as the deterministic tie-break. `internal_cost`
records that BKT-set objective. Public `evaluation_cost` and normalized regret
are computed separately by the shared frozen evaluator and must not be
substituted with the internal objective.

Every record identifies the closure, manifest, evaluator, student split,
state-compression rule, BKT numerical parameters and artifacts, distillation
table, surrogate config, and exact checkpoint.

## Reproduction

This commit assumes Tasks 12-1 through 12-3 artifacts already exist:

```powershell
python experiments\13_run_bkt_greedy.py
```

Outputs:

- `results/bkt_set_greedy/sequences.jsonl`
- `results/bkt_set_greedy/scored_sequences.csv`

