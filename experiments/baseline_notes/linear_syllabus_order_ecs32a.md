# Linear Syllabus Order on ECS32A

## Protocol

For each shared-manifest target, this deterministic baseline restricts the
official ECS32A teaching order to the target's `sequence_nodes`. It generates
standard sequence records only and has no internal Oracle, so
`internal_cost=null`. Public cost and regret are produced exclusively by the
canonical common scorer.

The runner validates before generation that:

- the teaching-order artifact covers all 61 DAG nodes exactly once;
- positions are exactly the integers 1 through 61;
- all 134 prerequisite edges point to a later teaching position;
- every target closure sequence covers `sequence_nodes` exactly once;
- every target is the final sequence node.

Each record stores the closure, materialized manifest, evaluator, and teaching
order artifact hashes. No manually maintained summary is generated; Task 14's
aggregator will derive final tables from the scored records.

## Files and reproduction

| File | Purpose |
|---|---|
| `data/ecs32a_teaching_order_required_full_v1.csv` | Official teaching order. |
| `experiments/20_run_linear_syllabus_order.py` | Validates the order and generates standard records. |
| `results/linear_syllabus_order/sequences.jsonl` | Standard method output. |
| `results/linear_syllabus_order/scored_sequences.csv` | Canonical evaluator output. |

```powershell
python experiments\20_run_linear_syllabus_order.py
python experiments\score_sequences.py `
  results\linear_syllabus_order\sequences.jsonl `
  --output results\linear_syllabus_order\scored_sequences.csv
```
