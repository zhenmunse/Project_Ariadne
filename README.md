# Project Ariadne

Project Ariadne is a research prototype for adaptive instructional sequencing over a prerequisite knowledge graph. The repository combines data preprocessing, a monotonic neural success-probability oracle, and a dynamic-programming planner for selecting learning paths on a DAG.

This repository is currently private and intended for internal research use. The README is written for research assistants who need to understand the code layout, reproduce the experimental pipeline, and safely extend the implementation.

## Research Goal

The project studies how to plan individualized learning paths when concepts are connected by prerequisite constraints. Given a learner state and a target learning objective, Ariadne estimates the probability and uncertainty of success for candidate concepts, then searches for a low-cost path through the prerequisite DAG.

The current prototype focuses on:

- Building concept-level sessions from raw interaction logs.
- Representing course concepts as a directed acyclic prerequisite graph.
- Training a graph-based Oracle that predicts concept success probability.
- Enforcing monotonic behavior with respect to prerequisite mastery.
- Comparing long-horizon DAG planning against myopic and no-prior baselines.

## Repository Structure

Only files tracked in the code repository are documented here. The `local/` directory is intentionally excluded because it is for private scratch work, local notes, or machine-specific artifacts.

```text
Project_Ariadne/
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   ├── ecs32a_concepts_required_full_v1.csv
│   ├── ecs32a_dag_edges_required_full_v1.csv
│   ├── ecs32a_dag_required_full_v1.json
│   ├── ecs32a_teaching_order_required_full_v1.csv
│   └── processed/
│       ├── .gitkeep
│       └── oracle_ckpt.pt
├── documents/
│   ├── 1_BKT.pdf
│   ├── 2_DKT.pdf
│   ├── 3_POMDP.pdf
│   ├── 4_RL_for_Instructional_Sequencing.pdf
│   ├── 5_Half_Life_Regression_Spaced_Repitition.pdf
│   ├── 6_SSP.pdf
│   ├── 7_LAOstar.pdf
│   ├── ecs32a_dag_full.pdf
│   └── papers.md
├── experiments/
│   ├── .gitkeep
│   ├── 01_preprocess.py
│   ├── 02_train_oracle.py
│   ├── 03_smoke_test.py
│   ├── 04_run_experiments.py
│   └── 04a_baseline_check.py
├── notebooks/
│   └── 01_ablation_analysis.ipynb
├── results/
│   ├── fig1_dp_vs_greedy.png
│   ├── fig2_risk_sensitivity.png
│   ├── fig3_oracle_comparison.png
│   ├── fig3_oracle_comparison.svg
│   ├── metrics.csv
│   └── trajectories.json
├── src/
│   ├── data_engine/
│   ├── oracle_core/
│   └── planner_engine/
├── Introduction.md
├── LICENSE
├── README.md
├── ecs32a_dag_interactive.html
└── requirements.txt
```

## Core Modules

### `src/data_engine`

This module prepares the graph and learning-session data used by the rest of the pipeline.

- `graph_builder.py` loads an item-to-concept mapping and prerequisite edges, checks that the graph is a DAG, builds NetworkX and tensor-friendly graph representations, and saves `graph.pkl`.
- `preprocessor.py` converts raw interaction logs into concept-level sessions and training samples. It filters items outside the graph, serializes concurrent events, aggregates repeated concept interactions into sessions, and produces learner-history samples for Oracle training.

Expected raw log columns are `user_id`, `item_id`, `is_correct`, and `timestamp`. If real raw files are missing, `experiments/01_preprocess.py` can generate toy data for a full smoke run.

### `src/oracle_core`

This module estimates whether a learner is likely to succeed on a target concept.

- `dataset.py` converts `(user_history, target_node, label)` samples into tensors with node-level learned flags and score features.
- `model.py` defines `MonotonicOracle`, a GCN-based model with a hard monotonicity design: if a learner has a superset of prerequisite mastery, the predicted success probability should not decrease under otherwise identical inputs.

The Oracle exposes `predict_mc(...)`, which uses MC Dropout to return:

- `P_succ`: estimated success probability.
- `sigma2`: uncertainty estimate.
- `T_base`: base time/cost constant used by the planner.

### `src/planner_engine`

This module turns Oracle predictions into learning paths.

- `zpd_utils.py` identifies valid next actions under prerequisite constraints.
- `solver.py` implements `DAGPlanner`, a memoized dynamic-programming planner over mastered-node states.
- `baselines.py` implements ablation baselines, including a one-step greedy planner and a frequency-only Oracle.

The planner minimizes the current cost function:

```text
Cost = T_base + (1 - P_succ) * T_penalty + lambda_risk * sigma2
```

`lambda_risk` controls how much uncertainty is penalized during planning.

## Pipeline

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run the pipeline from the repository root:

```bash
python experiments/01_preprocess.py
python experiments/02_train_oracle.py
python experiments/03_smoke_test.py
python experiments/04_run_experiments.py
```

Recommended meaning of each step:

| Step | Script | Purpose | Main Outputs |
| --- | --- | --- | --- |
| 1 | `experiments/01_preprocess.py` | Build the DAG and concept-level training sessions | `data/processed/graph.pkl`, `sessions.pkl`, `train_sessions.pkl` |
| 2 | `experiments/02_train_oracle.py` | Train and sanity-check the monotonic Oracle | `data/processed/oracle_ckpt.pt` |
| 3 | `experiments/03_smoke_test.py` | Test ZPD action logic, DP behavior, and Oracle integration | Console verification |
| 4 | `experiments/04_run_experiments.py` | Compare Ariadne and baseline strategies over sampled targets | `results/trajectories.json`, `results/metrics.csv` |

## Configuration

Most runtime settings live in `configs/config.yaml`.

Important sections:

- `data`: raw input paths and processed artifact directory.
- `mastery`: parameters for mastery proxies.
- `oracle`: hidden dimension, dropout, learning rate, epochs, batch size, MC samples, and model path.
- `planner`: failure penalty and risk sensitivity.
- `experiments`: target sampling and result-output settings.

Before running experiments on a new dataset, verify that the raw data paths, item-to-node mapping, and DAG edge file match the expected schemas.

## Results and Analysis

The repository includes example result artifacts:

- `results/trajectories.json`: planned paths and per-target strategy outputs.
- `results/metrics.csv`: compact experiment metrics for analysis.
- `results/fig*.png` / `results/fig*.svg`: figures used for comparison and reporting.
- `notebooks/01_ablation_analysis.ipynb`: exploratory notebook for ablation analysis.

Treat generated results as experiment artifacts, not as immutable source data. If rerunning the full pipeline, confirm whether overwriting result files is intended.

## Documents

`documents/` contains reference papers related to Bayesian Knowledge Tracing, Deep Knowledge Tracing, POMDPs, reinforcement learning for instructional sequencing, stochastic shortest path planning, LAO*, and spaced repetition. The root-level `Introduction.md` and `ecs32a_dag_interactive.html` provide additional project and DAG context.

These documents provide research context for the implementation and should help RAs connect code decisions to the broader literature.

## Development Notes for Research Assistants

- Keep `local/` out of README documentation and shared repo assumptions.
- Preserve the separation between `data_engine`, `oracle_core`, and `planner_engine`.
- When changing the Oracle, rerun the monotonic sanity checks in `experiments/02_train_oracle.py`.
- When changing planner logic, rerun `experiments/03_smoke_test.py`; the trap test is designed to catch accidental greedy behavior.
- When adding new experiments, prefer a new script under `experiments/` and record outputs under `results/`.
- Avoid committing private raw data unless the dataset is explicitly approved for repository storage.

## Current Status

Project Ariadne is an active research prototype. Some files are experiment artifacts, and some APIs may change as the paper direction evolves. For reproducibility, document any new assumptions in the relevant experiment script or in `documents/`.
