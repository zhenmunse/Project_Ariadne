# Project Ariadne

[English](README.md) | [中文](README_zh.md) | [ONBOARDING](ONBOARDING.md)

Project Ariadne 是一个面向先修知识图谱的自适应学习路径规划研究原型。本仓库把数据预处理、带单调性约束的神经网络成功率 Oracle，以及基于 DAG 的动态规划 Planner 组合在一起，用于研究如何为学习者选择概念学习路径。

本仓库目前是 private repo，主要供内部科研使用。这份 README 面向 Research Assistant，帮助你理解代码结构、复现实验流程，并在不破坏现有模块边界的情况下扩展实现。新加入的 RA 请先阅读 [ONBOARDING.md](ONBOARDING.md)，其中包含岗位任务、deadline、协作规范和第一周 deliverables。

## 研究目标

本项目研究的问题是：当课程概念之间存在先修约束时，如何根据学习者当前状态和目标学习概念，规划一条低成本的个性化学习路径。Ariadne 会估计候选概念的成功概率和不确定性，然后在先修 DAG 上搜索合适的学习路径。

当前原型重点包括：

- 从原始交互日志构建概念级 session。
- 将课程概念表示为有向无环先修图。
- 训练基于图结构的 Oracle，用于预测概念成功概率。
- 让 Oracle 对先修掌握状态满足单调性约束。
- 比较长视野 DAG 规划与 myopic、no-prior 等 baseline。

## 仓库结构

```text
Project_Ariadne/
├── .gitignore
├── ONBOARDING.md
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
│   ├── Introduction.md
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
├── LICENSE
├── README.md
├── README_zh.md
└── requirements.txt
```

## Onboarding

[ONBOARDING.md](ONBOARDING.md) 是给 RA 看的 onboarding guide。它包含项目概览、环境配置、阅读顺序、沟通规则，以及 Data & Pipeline、ML Experiments、Planning & Theory、Literature/Writing 等岗位的具体 deliverables。

## 核心模块

### `src/data_engine`

这个模块负责准备后续 pipeline 使用的图结构和学习 session 数据。

- `graph_builder.py` 读取 item-to-concept 映射和先修边，检查图是否为 DAG，构建 NetworkX 图和适合张量计算的图表示，并保存 `graph.pkl`。
- `preprocessor.py` 将原始交互日志转换为概念级 session 和训练样本。它会过滤不在图中的 item，序列化同一时间戳的并发事件，将连续的同概念交互聚合为 session，并生成 Oracle 训练所需的 learner-history 样本。

原始日志需要包含 `user_id`、`item_id`、`is_correct` 和 `timestamp` 四列。如果真实原始文件不存在，`experiments/01_preprocess.py` 可以生成 toy data，用于完整 smoke run。

### `src/oracle_core`

这个模块用于估计学习者在目标概念上的成功概率。

- `dataset.py` 将 `(user_history, target_node, label)` 样本转换为张量，包括节点级 learned flag 和 score feature。
- `model.py` 定义 `MonotonicOracle`，这是一个基于 GCN 的模型，并带有硬性的单调性设计：在其他输入相同的情况下，如果学习者掌握了更多先修概念，预测成功概率不应下降。

Oracle 暴露 `predict_mc(...)` 接口，通过 MC Dropout 返回：

- `P_succ`：估计成功概率。
- `sigma2`：不确定性估计。
- `T_base`：Planner 使用的基础时间或成本常数。

### `src/planner_engine`

这个模块把 Oracle 的预测转换为学习路径。

- `zpd_utils.py` 根据先修约束找出当前可学习的下一步 action。
- `solver.py` 实现 `DAGPlanner`，即在 mastered-node 状态空间上使用 memoized dynamic programming 的规划器。
- `baselines.py` 实现 ablation baseline，包括一步贪心 planner 和只基于频率的 Oracle。

Planner 当前最小化的成本函数是：

```text
Cost = T_base + (1 - P_succ) * T_penalty + lambda_risk * sigma2
```

其中 `lambda_risk` 控制规划时对不确定性的惩罚强度。

## 运行流程

先安装依赖：

```bash
pip install -r requirements.txt
```

然后在仓库根目录依次运行：

```bash
python experiments/01_preprocess.py
python experiments/02_train_oracle.py
python experiments/03_smoke_test.py
python experiments/04_run_experiments.py
```

每一步的含义如下：

| Step | Script | Purpose | Main Outputs |
| --- | --- | --- | --- |
| 1 | `experiments/01_preprocess.py` | 构建 DAG 和概念级训练 session | `data/processed/graph.pkl`, `sessions.pkl`, `train_sessions.pkl` |
| 2 | `experiments/02_train_oracle.py` | 训练并 sanity-check 单调 Oracle | `data/processed/oracle_ckpt.pt` |
| 3 | `experiments/03_smoke_test.py` | 测试 ZPD action 逻辑、DP 行为和 Oracle 集成 | 控制台验证输出 |
| 4 | `experiments/04_run_experiments.py` | 在采样 target 上比较 Ariadne 和 baseline 策略 | `results/trajectories.json`, `results/metrics.csv` |

## 配置

大部分运行参数都在 `configs/config.yaml` 中。

重要配置区块包括：

- `data`：原始输入路径和 processed artifact 目录。
- `mastery`：mastery proxy 相关参数。
- `oracle`：hidden dimension、dropout、learning rate、epochs、batch size、MC samples 和模型路径。
- `planner`：失败惩罚和风险敏感度。
- `experiments`：target 采样和结果输出设置。

在新数据集上运行实验前，请确认 raw data 路径、item-to-node 映射和 DAG edge 文件都符合预期 schema。

## 结果与分析

仓库中包含一些示例结果文件：

- `results/trajectories.json`：规划路径和每个 target 上的策略输出。
- `results/metrics.csv`：用于分析的紧凑实验指标。
- `results/fig*.png` / `results/fig*.svg`：用于比较和报告的图。
- `notebooks/01_ablation_analysis.ipynb`：用于 ablation analysis 的探索性 notebook。

这些结果应被视为实验 artifact，而不是不可变的源数据。如果重新运行完整 pipeline，请先确认是否希望覆盖已有结果文件。

## 文档资料

`documents/` 包含参考论文，主题包括 Bayesian Knowledge Tracing、Deep Knowledge Tracing、POMDP、instructional sequencing 中的 reinforcement learning、stochastic shortest path planning、LAO* 和 spaced repetition。`documents/Introduction.md` 提供额外的项目动机和研究背景。

## 开发注意事项

- 不要把 `local/` 纳入 README 文档或共享仓库假设。
- 保持 `data_engine`、`oracle_core` 和 `planner_engine` 三个模块的职责边界。
- 修改 Oracle 后，请重新运行 `experiments/02_train_oracle.py` 中的 monotonic sanity checks。
- 修改 Planner 逻辑后，请重新运行 `experiments/03_smoke_test.py`；其中 trap test 用来发现意外退化成 greedy behavior 的问题。
- 新增实验时，优先在 `experiments/` 下添加新脚本，并把输出记录到 `results/`。
- 除非数据集已明确允许放入仓库，否则不要提交私人 raw data。

## 当前状态

Project Ariadne 仍是活跃研究原型。部分文件是实验 artifact，部分 API 可能会随着论文方向调整而变化。为了保证可复现性，请把新增假设记录在相关实验脚本或 `documents/` 中。
