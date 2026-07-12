# ECS32A BKT-derived Set Oracle + LAO*

## Condition

This condition uses LAO* to optimize the state-dependent success probabilities
returned by the frozen **BKT-derived Set Oracle**. The planner observes only the
pair `(target_node, prerequisite-closed mastery set)`; it never receives a
student identifier, interaction prefix, BKT posterior, or chronological event
history.

Both the Greedy reference and LAO* load the same custom deterministic tensor
checkpoint on CPU. The checkpoint SHA-256 is:

```
4a4ae471e06dbeeea46bf09f0502f39455576ccdd7f992e0184912cac7b60791
```

The runner does not train or modify the surrogate. It rejects a different
checkpoint and rejects Greedy references whose manifest, closure, teacher,
distillation, surrogate, or evaluator provenance differs.

## Planning checks

For every shared-manifest target, the runner requires:

- LAO* convergence;
- a prerequisite-valid sequence covering exactly `sequence_nodes` and ending
  at the target;
- LAO* internal cost no greater than the matched Greedy cost plus `1e-9`;
- absolute agreement between LAO* and exhaustive memoized DP below `1e-9`;
- acceptance by the independent FrozenMonotonicOracle public evaluator.

The LAO* heuristic is the admissible sum heuristic with
`best_case_success_prob(v) = 1.0`. Internal BKT-set costs compare planners under
the distilled surrogate. Public evaluation costs and normalized regret are
separately computed by the shared frozen evaluator and should not be
interpreted as direct outcomes on real students.

## Reproduction

From the repository root, after the frozen BKT-set artifacts and Greedy
reference are present:

```powershell
python experiments\14_run_bkt_set_lao.py
```

Outputs:

- `results/bkt_set_lao/sequences.jsonl`
- `results/bkt_set_lao/scored_sequences.csv`
- `results/bkt_set/planner_comparison.csv`

`planner_comparison.csv` contains timing diagnostics, so its byte hash is not a
determinism contract. Sequence, cost, convergence, expansion, and iteration
signatures are checked across two independently loaded Oracle objects.
