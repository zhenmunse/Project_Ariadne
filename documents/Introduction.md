# Introduction

**Title:** Navigating the Zone of Proximal Development: Probabilistic Curriculum Sequencing via Robust Hierarchical Stochastic Shortest Paths

Zonglin Han

个性化学习（Personalized Learning）的核心挑战在于解决“探索”与“掌握”之间的权衡：既要巩固先修知识（Prerequisites），又要引导学生向新的知识边界拓展。尽管基于深度学习的知识追踪模型（如 DKT, SAINT+）在预测学生答题表现上取得了显著的精度提升，但现有的推荐系统往往陷入“短视”的误区：它们主要优化下一题的预测准确率（Next-item Accuracy），而非学生掌握复杂概念所需的**长期效率（Long-term Efficiency）**。更关键的是，这些黑盒模型缺乏对教育学理论的本质回应——特别是 Vygotsky 的“最近发展区”（Zone of Proximal Development, ZPD）理论——导致生成的推荐序列往往缺乏可解释性，甚至在冷启动场景下产生具有误导性的路径。

为了弥合高维深度模型与经典教育理论之间的鸿沟，我们提出了 **Project Ariadne**。这是一个模型驱动的（Model-based）教育优化框架，将课程排序问题形式化为知识图谱上的 **Robust Hierarchical Stochastic Shortest Path (RH-SSP)** 问题。与传统的强化学习方法不同，Ariadne 不依赖海量的试错交互，而是通过以下五个维度的创新，构建了一个兼具数学严谨性与教育可解释性的通用框架：

**1. 理论建模：分层随机最短路径 (Hierarchical SSP)** 为了解决大规模知识图谱（Knowledge Graph, KG）带来的状态空间爆炸（State Space Explosion）问题，我们提出了一种**双层规划架构**。

- **抽象层 (Abstract Layer)**：在粗粒度的“学习目标”（Learning Outcome, LO）层级进行全局路径规划，利用 SSP 算法计算通往最终掌握状态的最小期望时间（Minimum Expected Time）。
- **细化层 (Grounding Layer)**：在微观技能（Micro-skill）层面，通过局部策略评估具体的动作代价。 我们创新性地将 **ZPD** 理论转化为图上的**动作约束（Action Masking）**：在任意状态下，规划器仅被允许访问那些“前置知识已满足但尚未掌握”的边界节点。这种约束不仅大幅剪枝了搜索空间，更从根本上保证了推荐路径符合认知发展规律。

**2. 估计机制：结构化神经符号 Oracle (Structured Neuro-symbolic Oracle)** SSP 的核心在于转移概率（Transition Probability）的准确估计。我们拒绝使用无约束的端到端黑盒模型，而是设计了一个**结构化的神经符号 Oracle**。该 Oracle 利用图神经网络（GNN）聚合知识点的邻域信息（Embedding），但其输出层被严格限制在一个**单调函数族（Monotonic Function Family）**内。具体而言，我们强制模型遵循“难度差（Difficulty Gap）越大，成功率越低”的物理约束。这种 **Inductive Bias** 使得模型在极度稀疏的数据下也能泛化出合理的概率分布，避免了纯数据驱动模型在小样本下拟合出反常识的噪音。

**3. 迁移学习：基于课程对齐的冷启动策略 (Alignment-based Transfer)** 针对教育场景中常见的新课程“冷启动”难题，Ariadne 复用了我们此前的研究成果 *Quantifying Curriculum*。通过跨系统的知识点对齐（Curriculum Alignment），我们将成熟课程中学习到的“概念类型-难度曲线”元知识（Meta-knowledge），迁移至新课程的图结构中。这使得 Oracle 在没有任何交互数据的情况下，依然能基于图拓扑和对齐关系，推断出初始的转移代价，从而实现**零样本（Zero-shot）**或**少样本（Few-shot）**的高效初始化。

**4. 鲁棒规划：风险敏感的目标函数 (Risk-Sensitive Planning)** 鉴于 Oracle 的估计必然存在误差，传统的 SSP 可能会选择一条“平均时间最短但方差极大”的高风险路径。Ariadne 引入了**鲁棒优化（Robust Optimization）**机制，不再仅仅优化期望值，而是优化 **CVaR（Conditional Value at Risk）** 或 **最坏情况下的期望时间**。我们将 Oracle 的输出视为一个置信区间分布，规划器被训练为在不确定性下寻找最稳健的策略，从而避免学生陷入因模型预测失误而导致的“学习陷阱”。

**5. 评估体系：双重鲁棒离线策略评估 (Doubly Robust OPE)** 为了克服观测数据中的选择偏差（Selection Bias），我们建立了一套严格的离线评估流程。不同于简单的模拟实验，Ariadne 采用 **Doubly Robust (DR)** 方法进行反事实推断（Counterfactual Inference）。结合倾向性得分（Propensity Score）与我们的 Oracle 模型，DR 评估器能在仅有历史日志数据的情况下，无偏地估计出新策略在真实教学环境中的潜在收益，从而证实 Ariadne 在提升学习效率上的显著优势。

综上所述，Project Ariadne 不仅是一个高效的课程推荐算法，更是一种将认知科学原理（ZPD）、运筹优化（SSP）与现代深度学习（GNN）有机融合的方法论范式。它证明了：通过合理的结构化约束与鲁棒设计，人工智能可以在不牺牲可解释性的前提下，精确导航人类复杂的认知发展路径。

# Project Ariadne: Technical Architecture Brief

**Version:** Internal Draft v1.0

**Core Hypothesis:** Educational sequencing can be modeled as a risk-sensitive Stochastic Shortest Path (SSP) problem on a Knowledge Graph, where transition costs are estimated by a structured neuro-symbolic oracle.

------

## Part 1: Core Mechanics (两条主线)

### 主线 A：Hierarchical SSP with ZPD Constraints (The Planner)

**定义：** 一个在知识图谱上寻找“最小期望时间路径”的规划器。为了解决状态爆炸问题，我们采用分层架构和基于规则的动作剪枝。

1. **分层规划 (Hierarchy):**
   - **Abstract Layer (LO-Level):** 在“学习目标”（Learning Outcome）构成的图上进行全局规划。状态 $S_{abs}$ 是 LO 的掌握向量。
   - **Grounding Layer (Micro-Skill Level):** 每个 LO 内部包含若干微技能或题型。局部规划器计算完成该 LO 的期望代价（Expected Cost），并将其作为边权传递给 Abstract Layer。
   - **机制：** 这是一个嵌套的 Bellman Update 过程，上层规划依赖下层的 Cost Estimation。
2. **ZPD 动作剪枝 (Action Masking):**
   - 我们将 Vygotsky 的 ZPD 理论形式化为一种**启发式搜索约束**。
   - **规则：** 在任意时间步 $t$，Action Set $A_t$ 仅包含满足以下条件的节点 $v$：
     - $v$ 尚未被掌握 ($S_v = 0$)。
     - $v$ 的所有强前置依赖（Prerequisites）已被掌握 ($S_{pre(v)} = 1$)。
   - **作用：** 强制规划器在“可达前沿”（Frontier）上进行探索，物理上排除了无效的跳跃式学习，从而大幅压缩搜索空间。

### 主线 B：Structured Neuro-symbolic Oracle (The Estimator)

**定义：** 一个用于估计状态转移概率 $P(\text{success}|s,a)$ 和时间消耗 $Cost(s,a)$ 的预测模块。不同于纯黑盒模型，我们强制注入了结构化归纳偏置。

1. **图结构参数共享 (Graph-based Parameter Sharing):**
   - 利用 GNN (e.g., GAT/GCN) 在知识图谱上进行消息传递，生成每个知识点 $i$ 的 Embedding $z_i$。
   - **优势：** 相似或相邻的知识点共享参数特征，缓解单点数据稀疏问题。
2. **单调性约束 (Monotonic Inductive Bias):**
   - 我们显式定义预测函数 $f$ 为广义 IRT 形式的变体，而非自由拟合的 MLP。
   - **公式逻辑：** $P(\text{success}) = \sigma( \text{Ability}(u) - \text{Difficulty}(z_i) + \text{PrereqStrength} )$。
   - **约束：** 强制模型输出随“难度-能力差”单调下降。这保证了即使在训练数据极少时，模型也不会学习出“越难的题越容易做对”的过拟合噪声。
3. **对齐初始化 (Alignment-based Initialization):**
   - 利用 *Quantifying Curriculum* 的成果，将源课程（Source Course）训练好的 Oracle 参数，通过 LO 对齐关系迁移到目标课程（Target Course）。
   - **作用：** 为新课程的 Oracle 提供具有物理意义的参数初始化（Prior），而非随机初始化。

------

## Part 2: System Safeguards (三个保障)

### 保障 1：Risk-Sensitive Planning (应对模型不确定性)

**问题：** Oracle 的预测不是真理，而是带有误差的估计。盲目相信点估计（Point Estimate）会导致规划器选择“平均看起来很快，但一旦失败代价巨大”的脆弱路径。

**方案：**

- **Distributional Output:** Oracle 输出的不只是预测均值 $\mu$，还有置信度/方差 $\sigma^2$（或完整的分布参数）。
- **Robust Objective:** 规划器的目标函数从 $\min \mathbb{E}[T]$ 修改为 $\min (\mathbb{E}[T] + \lambda \cdot \text{Var}[T])$ 或优化 CVaR（Conditional Value at Risk）。
- **效果：** 规划器会倾向于选择那些“不仅期望时间短，而且预测置信度高”的稳健路径。

### 保障 2：Doubly Robust Off-Policy Evaluation (应对离线评估偏差)

**问题：** 我们没有在线 A/B Test 的权限，只能使用历史日志（Log Data）。直接使用 Log Data 评估新策略存在严重的选择偏差（Selection Bias）。

**方案：**

- 采用 **Doubly Robust (DR)** 估计器进行反事实推断。
  - 组件 A：**Direct Method (DM)**，即我们的 Oracle 模型，预测反事实结果。
  - 组件 B：**Inverse Propensity Weighting (IPW)**，估计历史 logging policy 选择该动作的概率。
- **声明：** 我们不声称能做到“完美无偏”，但 DR 方法能在“A 或 B 只要有一个准”的条件下，提供比简单的 Direct Simulation 更可信的下界估计。这是目前学术界公认的“尽力而为”的最佳实践。

### 保障 3：Cold Start Strategy (应对极寒启动)

**问题：** 在没有任何交互数据的新节点/新学生上，系统如何运行？

**方案：**

- **纯图拓扑推断：** 当 $N_{samples}=0$ 时，Oracle 完全退化为基于图结构（Graph Topology）和迁移参数（Transfer Priors）的推断。
- **动态校准：** 随着交互数据 $k=1, 2, 5...$ 的增加，Oracle 的权重逐渐从 Prior 向 Data Likelihood 倾斜（Bayesian Update 思想）。这保证了系统在 $t=0$ 时即可运行，且随着使用迅速变强。

------

## Part 3: Next Immediate Actions (下一步行动)

基于此技术简报，我们的工作重心不再是发散思维，而是**收敛验证**：

1. **数据准备 (Data)**: 获取 Junyi Academy 或类似数据集，构建 explicit Knowledge Graph (DAG)。
2. **Oracle 原型 (Prototype)**: 训练一个简单的 GNN+Monotonic Layer 模型，验证其在少样本下的预测 Loss 是否优于纯 MLP 和 DKT。
3. **Planner 冒烟测试 (Smoke Test)**: 在一个包含 ~20 个节点的子图上跑通 Risk-Sensitive SSP，观察生成的路径是否直观上合理（Sanity Check）。