# Project Ariadne — Week 3 Tasks (June 29 – July 5)

**AAAI-27 Abstract: July 21 (22 days)**
**AAAI-27 Full Paper: July 28 (29 days)**

---

## Tongan

### Task 1: Run DKT on real ECS 32A data (P0)

The cleaned dataset is at `data/processed/cleaned_interactions.csv` (71,350 rows, 296 students, 144 questions).

1. Convert `cleaned_interactions.csv` into pykt's expected input format. Document the conversion in a script at `experiments/05_convert_to_pykt.py`.
2. Train DKT using the same hyperparameters that worked on the sample dataset.
3. Report: training AUC, validation AUC, best epoch, training time.
4. If val AUC is significantly lower than on the sample data, investigate whether it's a data format issue before tuning hyperparameters.

**Deliverable:** Results in `experiments/baseline_notes/dkt_ecs32a.md` + conversion script. Due: July 2.

### Task 2: Reproduce BKT baseline (P1)

1. Set up pyBKT: `https://github.com/CAHLR/pyBKT`
2. Run on the same cleaned dataset.
3. Report: AUC, per-concept parameter estimates (p_init, p_learn, p_guess, p_slip).

**Deliverable:** Results in `experiments/baseline_notes/bkt_ecs32a.md`. Due: July 5.

---

## Jiawen

### Task 1: Build question → concept mapping template (P0)

1. From `cleaned_interactions.csv`, extract all 144 unique `item_id` values.
2. From the PrairieLearn export, find which assessment each question belongs to (you have this in `assessment_title` from the raw data).
3. Produce a CSV at `data/question_concept_mapping_template.csv` with the following columns:

```
item_id,assessment_title,question_label,concept_id
```

`item_id` and `assessment_title`: fill from data.
`question_label`: if you can find the human-readable question name from PL (e.g., "Q3: for loop counting"), include it. Otherwise leave blank.
`concept_id`: leave blank — Zonglin will fill this by mapping to the DAG.

4. Sort by assessment_title, then by item_id.

**Deliverable:** `data/question_concept_mapping_template.csv` + a short note in the PR describing how you extracted the data. Due: July 2.

### Task 2: Per-assessment cleaning statistics (P1)

Add a per-assessment breakdown to your cleaning report:

| assessment_title | raw_rows | dropped_rows | drop_rate | cleaned_rows |
|---|---|---|---|---|

This helps us identify if specific assessments have abnormal data quality.

**Deliverable:** Updated `data/processed/cleaning_report.md`. Due: July 3.

### Task 3: Document high-retry outliers (P2)

Write up your findings on high-retry behavior (the 119-attempt student, ≥50 and ≥100 attempt pairs, attempt interval analysis) as a section in the cleaning report. Include:
- Count of student-question pairs at various retry thresholds (≥10, ≥20, ≥50, ≥100)
- The attempt interval distribution (median, p25, p75)
- Your recommendation on whether these should be filtered (answer: no, keep in cleaned data, handle at preprocessor stage)

**Deliverable:** New section in `data/processed/cleaning_report.md`. Due: July 5.

---

## Yichen

### Task 1: Read and summarize LAO* (P0)

Read Hansen & Zilberstein (2001), "LAO\*: A heuristic search algorithm that finds solutions with loops." The PDF is at `documents/7_LAOstar.pdf`.

Write a summary document covering:
1. The three-phase loop: Expand → Test for convergence → Cost revision
2. What a "solution graph" is and how it differs from a search tree
3. How LAO\* handles stochastic transitions (AND-nodes vs OR-nodes)
4. What "admissible heuristic" means in this context and why it matters
5. The convergence guarantee (Theorem 1): under what conditions does LAO\* find the optimal solution?

Keep it concise (1–2 pages). Use your own words — this is for your understanding, not for the paper.

**Deliverable:** `documents/lao_star_summary.md`. Due: July 3.

### Task 2: Read the Ariadne SSP formulation (P1)

After finishing the LAO\* summary, read:
- `AGENT.md` — project context and formulation summary
- `Introduction.md` — project motivation
- The convergence proof document (will be shared separately)

Write a short note (half page) on how Ariadne's SSP maps onto LAO\*: what are the states, actions, transitions, and how does ZPD masking constrain the search.

**Deliverable:** Append to `documents/lao_star_summary.md`. Due: July 5.

---

## Zonglin

### LAO* solver implementation (P0, critical path)

Implement the LAO\* solver in `src/planner_engine/solver.py`, replacing the current memoized DP. Use the geometric cost function: $c(s,v) = T_v / p(v,s)$.

Target: a working solver that can plan on the 61-node DAG with a synthetic oracle by July 5. Correctness verification against exact DP on 15-node subgraphs.

### Question → concept mapping (P0)

Fill in `concept_id` column in Jiawen's mapping template once she delivers it. Estimated time: 30 min.

---

## Communication

- Weekly sync: TBD (coordinate in WeChat group)
- All code via PR to `main`. No direct pushes.
- If blocked, post in WeChat group or open a GitHub Issue.
- Tag @zonglin for anything that needs a decision.
