# Onboarding Guide for Research Assistants

Last updated: June 13, 2026

## Project Overview

Project Ariadne frames personalized learning path optimization as a **Stochastic Shortest Path (SSP)** problem on a prerequisite DAG. Given a student's current mastery state and a target set of concepts, the system finds the expected-cost-minimizing sequence in which to study those concepts.

**Key components:**

- A **prerequisite DAG** (61 concepts, 134 edges) representing the concept structure of UC Davis ECS 32A (introductory Python).
- An **Oracle** that estimates the probability a student will master a concept given their current state.
- A **Planner** (LAO\* solver) that searches for the optimal learning path under the SSP formulation.
- **Baselines** including greedy planning, random valid ordering, linear syllabus order, and LLM-based recommendation.

**Target venue:** AAAI-27 Main Technical Track  
**Abstract deadline:** July 21, 2026  
**Full paper deadline:** July 28, 2026

Read `README.md` for repo structure, module descriptions, and how to run the pipeline.

---

## Getting Started (All Positions)

### 1. Environment Setup

```bash
git clone <repo_url>
cd Project_Ariadne
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify installation:

```bash
python experiments/03_smoke_test.py
```

This should complete without errors. If it does, your environment is correct.

### 2. Read These First

- `README.md` — repo structure and pipeline overview
- `documents/Introduction.md` — project motivation and research context
- `documents/papers.md` — annotated reading list (start with 6_SSP.pdf and 7_LAOstar.pdf)
- `documents/ecs32a_dag_full.pdf` — concept DAG reference

### 3. Communication

- Primary channel: WeChat group (link will be shared separately)
- Weekly sync: TBD (will accommodate UTC+8 / UTC-7 time difference)
- Async work rules:
  - Create a feature branch for your work. **Never push directly to `main`.**
  - Commit messages in English, descriptive (e.g., `add dedup logic to preprocessing pipeline`, not `update`)
  - When blocked, open a GitHub Issue describing: what you tried, what you expected, what happened. Tag @zonglin.
  - If unsure whether a design decision is correct, ask before implementing — don't guess and build on top of a guess.

---

## Position A: Data & Pipeline

**Your role:** Turn raw PrairieLearn interaction logs into the clean, structured dataset that the Oracle and Planner consume.

### Background

PrairieLearn logs record every student submission: which question they attempted, whether they got it right, and when. Each question maps to one or more concepts in our DAG. Your job is to build and validate the pipeline that converts these logs into concept-level learning sessions.

### Week 1 Deliverable (Due: June 21)

**Task: Build the data cleaning pipeline.**

Input: a CSV file exported from PrairieLearn with the following columns:

| Column | Type | Description |
|---|---|---|
| `student_id` | string | Anonymized student identifier |
| `question_id` | string | PrairieLearn question identifier |
| `timestamp` | ISO 8601 | Submission time |
| `is_correct` | 0 or 1 | Whether the submission was correct |
| `attempt_number` | int | Which attempt this was for this question |

Output: a cleaned CSV matching the schema expected by `src/data_engine/preprocessor.py` (columns: `user_id`, `item_id`, `is_correct`, `timestamp`).

**Cleaning rules:**

1. **Deduplication:** If a student has multiple rows for the same `(question_id, timestamp)` pair, keep the row with the highest `attempt_number`.
2. **Missing values:** Drop rows where `is_correct` is missing. Log the count and distribution of dropped rows (by student and by question).
3. **ID normalization:** Strip leading zeros from `student_id`. Ensure consistent string format.
4. **Orphan filtering:** Drop rows whose `question_id` does not appear in the concept mapping file (`data/ecs32a_concepts_required_full_v1.csv`). Log dropped question IDs.
5. **Validation:** After cleaning, assert:
   - No duplicate `(student_id, question_id, timestamp)` triples
   - No null values in any column
   - All `question_id` values exist in the concept mapping
   - Print summary statistics: total rows before/after, number of unique students, number of unique questions

**Deliverable format:** A Python script at `experiments/00_clean_raw_data.py` that reads from `data/raw/` and writes to `data/processed/cleaned_interactions.csv`. Include a `--dry-run` flag that prints statistics without writing the output file.

### Week 2+ Roadmap

- Validate the concept mapping: verify that every question in the PL export maps to exactly one concept in the DAG.
- Run `experiments/01_preprocess.py` on the cleaned data and verify output integrity.
- Build a data summary report: per-concept attempt counts, success rates, temporal distribution.

---

## Position B: ML Experiments

**Your role:** Reproduce baseline knowledge tracing models and run the experimental comparison matrix.

### Background

Our paper compares the Ariadne planner against multiple baselines across two dimensions: **Oracle quality** (how accurately we estimate success probability) and **planning strategy** (how we use those estimates to pick the next concept). You will reproduce the Oracle baselines and run them through our evaluation pipeline.

### Week 1 Deliverable (Due: June 21)

**Task: Reproduce the DKT baseline.**

1. Clone the pykt-toolkit repository: `https://github.com/pykt-team/pykt-toolkit`
2. Follow their setup instructions and verify you can train DKT on their provided sample dataset.
3. Report: training AUC, validation AUC, number of epochs to convergence, and training time.
4. Document any setup issues, version mismatches, or deviations from their README in a short markdown file (`experiments/baseline_notes/dkt_setup.md`).

**Do not** attempt to run DKT on our data yet — that requires the cleaned dataset from Position A. Week 1 is purely about verifying you can reproduce an existing result.

### Week 2+ Roadmap

Priority-ordered baseline reproduction list:

| Priority | Model | Source | Notes |
|---|---|---|---|
| P0 | DKT | pykt-toolkit | Week 1 target |
| P1 | BKT | pyBKT (`https://github.com/CAHLR/pyBKT`) | Simpler model, should be fast |
| P1 | Frequency Oracle | Already in `src/oracle_core/` | Just needs to be wired into the new experiment runner |
| P2 | SAINT+ | pykt-toolkit | More complex, may need GPU |
| P3 | LLM baseline | To be designed | Will use Claude/GPT-4 API, detailed spec coming later |

After baselines are individually verified, we integrate them into our 10-condition experiment matrix:

|  | Greedy | LAO\* (Optimal) |
|---|---|---|
| FrequencyOracle | Freq + Greedy | Freq + LAO\* |
| DKT | DKT + Greedy | DKT + LAO\* |
| MonotonicOracle | Mono + Greedy | **Mono + LAO\* (Ours)** |

Plus standalone conditions: Random Valid Ordering, Linear Syllabus Order, LLM-Full, LLM-Zero.

---

## Sample Data for Testing

Until the real PrairieLearn data is cleaned and anonymized, use the synthetic data generator:

```bash
python experiments/01_preprocess.py
```

This creates toy interaction data in `data/processed/` that exercises the full pipeline. Use this to verify your code works end-to-end before the real data arrives.

---

## Questions?

If something in this document is unclear or contradicts the README, **this document takes precedence for task assignments**. Open a GitHub Issue or message the WeChat group.
