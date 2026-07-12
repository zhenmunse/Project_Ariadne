# ECS32A DKT-derived Set Oracle + LAO*

This condition globally plans over the shared manifest closures using the
frozen **DKT-derived Set Oracle**. The planner observes only `(target, mastery
set)`; student identity, chronological prefix and DKT hidden state remain
confined to the train-only teacher/distillation pipeline.

The runner never trains or modifies the surrogate. Both this condition and the
DKT-set Greedy reference must use checkpoint:

```text
74ee76f29e852b77f3116a6840386342ec256088a245a871a67ff4f4142c012a
```

Run from the repository root:

```powershell
python experiments\16_run_dkt_set_lao.py
python experiments\summarize_dkt_set.py
```

All ten LAO* searches converged. Each internal cost exactly matched the exact
DP cost (`max |J_LAO* - J_DP| = 0.0`) and none exceeded its strictly validated
Greedy reference. Eight targets improved and two tied. Outputs are in
`results/dkt_set_lao/`; the cross-planner table is
`results/dkt_set/planner_comparison.csv`.

Internal cost uses the DKT-set surrogate. Public evaluation uses the separate
shared FrozenMonotonicOracle and is not a direct empirical student outcome.
