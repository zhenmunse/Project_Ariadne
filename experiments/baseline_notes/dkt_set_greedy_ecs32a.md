# ECS32A DKT-derived Set Oracle + Greedy

This condition greedily plans over the shared manifest closures using the
frozen **DKT-derived Set Oracle**. The planner observes only `(target, mastery
set)`; student identity, chronological prefix and DKT hidden state are confined
to the train-only teacher/distillation pipeline.

The CPU deterministic surrogate is shared infrastructure with Task 12 and uses
the same `122 -> 128 -> 64 -> 1` architecture. This runner never trains it and
accepts only checkpoint:

```
74ee76f29e852b77f3116a6840386342ec256088a245a871a67ff4f4142c012a
```

Run from the repository root:

```powershell
python experiments\15_run_dkt_set_greedy.py
```

Outputs are `results/dkt_set_greedy/sequences.jsonl` and
`results/dkt_set_greedy/scored_sequences.csv`. Internal cost uses the DKT-set
surrogate; public evaluation is independently performed by the shared frozen
evaluator and is not a direct empirical student outcome.
