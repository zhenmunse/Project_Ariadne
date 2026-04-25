# Project Ariadne — Briefing Notes

Zonglin Han · April 2026

---

## The Problem

We all know UCD's CS courses form a long prerequisite chain — ECS 32A → 32B → 32C → 34, then 154A, 170, etc. Every student enters with different background: some have AP credit that covers 32A, some have never seen a variable. But right now, every non-major student walks the same path, in the same order.

**Question: Is this path optimal for every student?** If a student already has partial mastery, could they skip certain intermediate steps, or reorder their learning sequence, to reach their target more efficiently?

We've talked before about how ECS 32B is a rough transition for non-majors — data structures jumps too steeply from 32A. That kind of sequencing problem wastes student time and causes dropouts. My research goal is to turn "what's a good learning sequence" from instructor intuition into a **computable optimization problem**.

---

## Modeling the Problem

Course prerequisites form a **Directed Acyclic Graph (DAG)**. This structure is already explicit in the course catalog. A student's current knowledge state is the set of "lit up" nodes on this graph — the things they've already mastered.

The optimization question: **Given a student's current state, what is the optimal path to reach a target course/skill?**

This is NOT a deterministic shortest path. Taking a course doesn't guarantee mastery — a student might fail, might need to retake, might need supplementary work. So this is a **Stochastic Shortest Path (SSP)** problem: at each step, there's a probability of success (move forward) and a probability of failure (stay and retry), and we want to minimize the **expected total time** to reach the goal.

---

## What Existing Work Does (and Doesn't Do)

### Category 1: Knowledge Tracing — Diagnosis without Prescription

- **BKT** (Corbett & Anderson, 1995) and **DKT** (Piech et al., 2015) can predict whether a student currently knows a skill, but they don't answer "what should the student learn next."
- Analogy: a hospital lab tells you your blood sugar level, but doesn't prescribe medication.

### Category 2: RL / POMDP for Teaching Decisions — Too Data-Hungry

- **Rafferty et al. (2016)** modeled teaching as a POMDP. Theoretically elegant, but POMDPs require massive online interaction data to train policies, and the computation is expensive. Not practical for a classroom of ~200 students.
- **Doroudi et al. (2019)** surveyed RL for instructional sequencing and found that most RL approaches optimize short-term metrics (next-item accuracy) rather than long-term efficiency.
- **Settles & Meeder (2016)** at Duolingo optimized spaced repetition for individual vocabulary items — but only for single items, no prerequisite dependencies between concepts.
- **Ye et al. (KDD 2022)** used SSP for spaced repetition scheduling — again single-item review interval optimization, no graph structure, no prerequisite dependencies.

### Category 3: Computational ZPD — Greedy Only

- **ZPDES** (Clement et al., 2015) used multi-armed bandits to select exercises within a student's Zone of Proximal Development. But bandit methods are **myopic** — they pick the best next action without considering long-term consequences. No global optimality guarantee.
- **E-Gotsky** (Ronen et al., 2019) also used ZPD for content sequencing, but again rule-based / ML heuristics, not optimal planning.

### The Gap

**Nobody has combined these three things:**

1. Prerequisite DAG (graph structure with dependencies)
2. SSP (globally optimal stochastic planning, not greedy)
3. ZPD as a formal action constraint (not just a heuristic)

That's what Ariadne does.

---

## My Approach

### Formulation

- **State**: the subset of knowledge points a student has mastered. (If there are |V| nodes in the DAG, the state space is theoretically 2^|V|.)
- **Action**: at any state, the student can attempt any knowledge point whose prerequisites are ALL already mastered. This is the **ZPD constraint** — formalized as action masking on the SSP.
- **Transition**: attempting knowledge point v succeeds with probability p(v, s), moving to a new state; fails with probability 1 - p(v, s), staying in the current state (student retries).
- **Objective**: minimize expected total time from current state to target mastery.

### Theoretical Foundation

This formulation maps directly onto **Bertsekas & Tsitsiklis (1991)**'s SSP framework. Under standard conditions (a proper policy exists — i.e., the student can eventually reach the goal), the Bellman equation has a unique solution and value iteration converges. This is not new theory — the contribution is **the mapping itself** and what follows from it.

### Core Contributions

1. **Problem formulation**: mapping educational sequencing onto SSP with prerequisite DAG structure.
2. **ZPD pruning preserves optimality**: proving that restricting actions to the ZPD frontier doesn't cut off optimal paths. (Because on a DAG, you can't reach a node without first mastering its prerequisites anyway — so ZPD masking removes only infeasible paths, not useful ones.)
3. **Scalability analysis**: the raw state space is 2^|V|, but ZPD + DAG topology makes the reachable state space much sparser. Quantifying this compression is part of the contribution.
4. **Empirical validation on PrairieLearn data**.

---

## Why PrairieLearn

The framework works at the course-prerequisite level conceptually, but **validating it requires fine-grained data**. A course-level dataset only gives one grade per course — not enough to estimate transition probabilities at the concept level.

PrairieLearn is different:

- Every question maps to a specific concept.
- Every student attempt (correct/incorrect) is logged.
- The concept structure within ECS 32A is already a DAG — concepts have prerequisite dependencies built into the course design.

So PL naturally provides both the **knowledge graph** and the **interaction data** needed to estimate transition probabilities. No need to build a separate knowledge graph from scratch.

**What Ariadne uses from PL**: student answer logs (correct/incorrect per concept per attempt). That's it.

**What Ariadne does NOT use**: Dashboard analytics, Live CCTV, participation tracking — those are valuable for the pedagogy paper's assessment dimensions, but Ariadne only needs the raw answer data.

Two papers, two different layers of PL data, no resource conflict.

---

## Key References

| # | Paper | Venue | Role in this project |
|---|-------|-------|---------------------|
| 1 | Corbett & Anderson, "Knowledge Tracing" | UMUAI 1995 | Baseline: diagnosis only |
| 2 | Piech et al., "Deep Knowledge Tracing" | NIPS 2015 | Baseline: black-box diagnosis |
| 3 | Rafferty et al., "Faster Teaching via POMDP Planning" | Cognitive Science 2016 | Prior art: POMDP approach |
| 4 | Doroudi et al., "Where's the Reward?" | IJAIED 2019 | Survey of RL for sequencing |
| 5 | Settles & Meeder, "Half-Life Regression" | ACL 2016 | Single-item spaced repetition |
| 6 | Bertsekas & Tsitsiklis, "Analysis of Stochastic Shortest Path Problems" | Math. Oper. Res. 1991 | Theoretical foundation |
| 7 | Hansen & Zilberstein, "LAO*" | Artificial Intelligence 2001 | Heuristic search for SSP |

---

## Current Status & Timeline

- **Done**: MVP codebase (data pipeline, oracle prototype, DP planner, baseline comparisons, smoke tests). IEEE SoutheastCon 2026 paper (curriculum alignment — feeds into this project as prior work).
- **In progress**: Locking down the mathematical formulation. Supplementing probability and optimization background.
- **Summer 2026**: Core proofs (NP-hardness, ZPD optimality preservation), experiments on PL data.
- **August 2026**: Target submission (AAAI / ICAPS main track if the theory is strong enough; EDM / AIED as fallback with more applied framing).
