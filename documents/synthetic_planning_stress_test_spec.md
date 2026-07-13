# Project Ariadne — Synthetic Planning Stress Test Spec

## 0. Objective

新增 controlled synthetic / semi-synthetic experiments，用于回答：

1. **When does long-horizon planning matter?**
2. 在 correctly specified learner model 下，LAO* 是否能避开 myopic transfer traps？
3. state-dependence strength 增强时，Greedy regret 是否系统性上升？
4. analytic certified bound \(p_t^*(v)\) 是否能减少 LAO* state expansions，同时保持 optimal value 不变？
5. synthetic oracle 是否能作为 Section 3 general interface \(p(v,s)\) 的独立实例，而不修改 formulation 或 solver？

主实验叙事：

> Hard prerequisites determine feasibility, while soft cross-concept transfer determines the value of ordering among feasible actions.

---

# Task S1 — 建立 synthetic experiment module

## To-do

新增目录：

```text
experiments/synthetic/
    config.py
    graph_factory.py
    transfer_factory.py
    oracle.py
    calibration.py
    run_dependence_sweep.py
    run_trap_family.py
    run_bound_ablation.py
    aggregate_results.py
```

新增测试：

```text
tests/test_synthetic_oracle.py
tests/test_synthetic_calibration.py
tests/test_synthetic_planning.py
```

新增输出目录：

```text
results/synthetic/
    dependence_sweep/
    trap_family/
    bound_ablation/
    figures/
```

## Deliverables

- synthetic experiment package 可独立运行；
- 所有 config、seed、graph、transfer weights、oracle parameters 可序列化；
- 每个 run 保存完整 provenance hash。

---

# Task S2 — 分离 feasibility graph \(G\) 与 transfer graph \(H\)

## Technical definition

### Hard prerequisite graph

\[
G=(V,E_G)
\]

用途：

- 定义 legal actions；
- 定义 prerequisite closure；
- 决定 reachable mastery states；
- 不直接定义 success probability。

Legal action：

\[
\mathcal A(s)=
\{v\notin s:\operatorname{pred}_G(v)\subseteq s\}.
\]

### Soft transfer graph

\[
H=(V,E_H,W)
\]

其中：

\[
w_{uv}\ge 0.
\]

用途：

- 定义 mastered concept \(u\) 对 target concept \(v\) 的 positive transfer；
- 不改变 action legality；
- 允许 sibling / parallel-branch / related-concept transfer；
- 不限制为 \(u\prec_G v\)。

禁止：

- negative weights；
- 将 \(H\) 的 edges 自动加入 \(G\)；
- 仅使用 ancestor-only transfer；
- 修改 Section 3 formulation。

## Transfer edge policy

至少支持：

```text
ancestor_transfer
sibling_transfer
cross_branch_transfer
mixed_transfer
```

主实验优先使用：

```text
mixed_transfer
```

约束：

- 所有 transfer weights nonnegative；
- transfer sparsity 固定或可控；
- transfer graph 生成过程 deterministic under seed；
- 不允许 self-loop \(w_{vv}\)。

## Deliverables

- `graph_factory.py`；
- `transfer_factory.py`；
- G/H 分离的数据结构；
- graph / transfer artifact JSON；
- unit tests 验证：
  - \(G\) 为 DAG；
  - \(H\) 不要求为 prerequisite subgraph；
  - \(w_{uv}\ge 0\)；
  - no self-transfer；
  - fixed seed 可复现。

---

# Task S3 — 实现 monotone synthetic oracle

## Oracle definition

\[
p_\beta(v,s)
=
\sigma\left(
\alpha_v(\beta)
+
\beta\sum_{u\in s}w_{uv}
\right),
\]

其中：

\[
\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\beta\ge 0,
\qquad
w_{uv}\ge 0.
\]

## Required interface

```python
class SyntheticTransferOracle:
    def success_prob(self, v: int, state: frozenset[int]) -> float:
        ...

    def base_cost(self, v: int) -> float:
        return 60.0

    def best_case_success_prob(self, v: int) -> float:
        ...
```

## Requirements

- deterministic；
- same \((v,s)\) always returns identical value；
- monotonicity：

\[
s\subseteq s'
\implies
p_\beta(v,s)\le p_\beta(v,s');
\]

- output strictly in \((0,1]\)；
- uniform nominal cost：

\[
T_v=60.
\]

## Tests

随机采样 valid state pairs \(s\subseteq s'\)，断言：

```text
p(v, s) <= p(v, s')
```

测试：

```text
same query -> same probability
query-order independence
new oracle object -> same probability
beta = 0 -> state-independent probability
```

## Deliverables

- `oracle.py`；
- monotonicity test；
- determinism test；
- serialized oracle config。

---

# Task S4 — 实现 marginal-difficulty calibration

## Goal

扫描 \(\beta\) 时保持整体 problem difficulty 基本不变，避免混合：

```text
state-dependence strength
+
overall success-probability increase
```

## Reference concept difficulty

为每个 concept 固定：

\[
q_v\in(0,1).
\]

可选来源：

```text
fixed synthetic distribution
FrozenMonotonicOracle marginal probabilities
ECS32A concept-level empirical prior
```

主实验需冻结一种来源。

## Reference transfer mass

对固定 Random Frontier policy \(\pi_{\mathrm{RF}}\)，估计：

\[
m_v^{\mathrm{ref}}
=
\mathbb E_{\sigma\sim\pi_{\mathrm{RF}}}
\left[
\sum_{u\in s_v(\sigma)}w_{uv}
\right].
\]

实现方式：

- 每个 graph instance 固定 Random Frontier seeds；
- 每个 concept 统计被学习前的 transfer mass；
- Monte Carlo sample count 写入 config；
- calibration 与 evaluation 使用不同 random seeds。

## Node-level calibration

\[
\alpha_v(\beta)
=
\operatorname{logit}(q_v)
-
\beta m_v^{\mathrm{ref}}
+
c_\beta.
\]

## Global calibration

通过 one-dimensional root finding 求 \(c_\beta\)，满足：

\[
\mathbb E_{\sigma\sim\pi_{\mathrm{RF}}}
[C_\beta(\sigma)]
=
C_{\mathrm{reference}}.
\]

推荐：

```text
solver: bisection or Brent
absolute tolerance: 1e-10
maximum iterations: 200
```

## Required diagnostics

每个 \(\beta\) 输出：

```text
beta
c_beta
reference_cost
calibrated_random_frontier_cost
absolute_calibration_error
mean_probability_at_reference_state
min_probability
max_probability
```

## Acceptance criteria

\[
\left|
C^{\mathrm{RF}}_\beta-C_{\mathrm{reference}}
\right|
\le 10^{-6}
\]

或 relative error：

\[
\le 10^{-8}.
\]

## Deliverables

- `calibration.py`；
- calibration artifact CSV/JSON；
- root-finding regression test；
- beta sweep 中 Random Frontier reference cost 基本恒定。

---

# Task S5 — 构建 graph families

## A. Layered synthetic DAGs

至少包含：

```text
4 layers x 4 nodes
4 layers x 5 nodes
4 layers x 6 nodes
```

每个 size 至少生成：

```text
5 graph seeds
```

控制参数：

```text
layer_count
nodes_per_layer
edge_density
minimum_indegree
maximum_indegree
graph_seed
```

要求：

- connected to target；
- unique target sink；
- closure contains all generated nodes；
- nontrivial frontier states；
- DAG width / depth 写入 metadata。

## B. ECS32A semi-synthetic topology

使用 frozen ECS32A target closures 的 \(G\)，替换 learner oracle 为 synthetic transfer oracle。

至少覆盖：

```text
10 frozen target closures
```

可选正文展示：

```text
representative medium / large closure
or target-equal aggregate across all 10 closures
```

## C. Transfer graph generation

对 layered DAG 和 ECS32A closure 分别生成 \(H\)。

推荐：

```text
transfer density: {0.1, 0.2, 0.3}
weight distribution: nonnegative bounded distribution
```

权重建议：

\[
w_{uv}\sim\operatorname{Uniform}(0,w_{\max})
\]

或 normalized Gamma / LogNormal 后截断。

要求：

- 固定 expected incoming transfer mass；
- 不随 graph size 无控制增长；
- 对不同 topology 进行 degree normalization。

## Deliverables

- graph instances；
- transfer instances；
- instance manifest；
- graph statistics CSV；
- visualization 可选，非必须。

---

# Task S6 — \(\beta=0\) regression test

## Required property

当：

\[
\beta=0
\]

时：

\[
p(v,s)=\sigma(\alpha_v)
\]

与 state 无关。

由于每个 node 必须恰好学习一次：

\[
J_{\mathrm{Greedy}}
=
J_{\mathrm{LAO^*}}
=
J_{\mathrm{DP}}.
\]

并且：

\[
R_{\mathrm{Greedy}}
=
R_{\mathrm{LAO^*}}
=
0.
\]

## Important restriction

不要求：

```text
Greedy sequence == LAO* sequence
```

原因：

- \(\alpha_v\) 可异质；
- Greedy 按 immediate cost 选择；
- LAO*/DP 在 global value ties 下按 deterministic tie-break；
- total cost equality 不保证 sequence equality。

## Tests

对所有 graph families 和 seeds：

```text
abs(greedy_cost - dp_cost) <= 1e-9
abs(lao_cost - dp_cost) <= 1e-9
greedy_regret == 0 within tolerance
lao_regret == 0 within tolerance
```

## Deliverables

- `test_beta_zero_equivalence`；
- full regression report；
- no sequence-equality assertion。

---

# Task S7 — Controlled dependence-strength sweep

## Beta grid

初始 grid：

\[
\beta\in
\{0,\ 0.25,\ 0.5,\ 1,\ 2,\ 4\}.
\]

如 probability saturation 过强，调整为：

\[
\{0,\ 0.1,\ 0.25,\ 0.5,\ 1,\ 2\}.
\]

最终 grid 必须在看主结果前冻结。

## Conditions

每个 instance / beta 跑：

```text
Exact DP
LAO* with analytic bound
Greedy
Random Frontier
```

Random Frontier：

```text
100 runs per instance-beta
```

## Primary metric

Greedy model-based normalized regret：

\[
R_{\mathrm{Greedy}}(\beta)
=
\frac{J_{\mathrm{Greedy}}(\beta)-J^*(\beta)}
{J^*(\beta)}.
\]

## Secondary metrics

```text
LAO* optimality gap
Random Frontier mean regret
Random Frontier regret std
expanded states
iterations
runtime
optimal sequence cost
calibration error
```

## Aggregation

先按：

```text
graph instance
beta
```

聚合，再按 topology 等权。

不得按 node count 或 run count 直接 pool。

## Expected regression behavior

```text
beta = 0 -> Greedy regret = 0
beta increases -> mean Greedy regret nondecreasing in aggregate
LAO* regret = 0
Random Frontier regret generally increases with beta
```

不强制每个 individual seed 单调，但 aggregate trend 应报告。

## Deliverables

```text
results/synthetic/dependence_sweep/raw_runs.jsonl
results/synthetic/dependence_sweep/per_instance.csv
results/synthetic/dependence_sweep/per_topology.csv
results/synthetic/dependence_sweep/summary.json
```

---

# Task S8 — Trap family

## Hard prerequisite structure

构造最小解释型 family：

```text
a and b initially legal

a -> c1
a -> c2
...
a -> ck

b -> target
c1 -> target
c2 -> target
...
ck -> target
```

其中：

- \(a\) 是 beneficiaries \(c_i\) 的 hard prerequisite；
- \(b\) 是 final target 的 prerequisite；
- \(b\) 不是 \(c_i\) 的 hard prerequisite；
- 学完 \(a\) 后，\(b,c_1,\dots,c_k\) 同时 legal。

## Soft transfer

\[
w_{b,c_i}=\tau,
\qquad i=1,\dots,k.
\]

其他 transfer weights 可设为 0。

## Base probabilities

设：

\[
p_b=p_a-\delta
\]

或直接冻结：

```text
p_b
q = p(c_i | b not mastered)
```

当 \(b\in s\)：

\[
p(c_i\mid b\in s)
=
\sigma(\operatorname{logit}(q)+\tau).
\]

## Parameter grid

建议：

```text
delta: 10-20 evenly spaced values
tau: 10-20 evenly spaced values
k: {2, 4, 8, 16}
```

实际范围需避免 probability saturation。

## Primary result

Greedy regret landscape：

\[
R_{\mathrm{Greedy}}(\delta,\tau,k).
\]

主文固定一个代表性 \(k\) 画 heatmap。

Supplementary 报其余 \(k\)。

## Boundary annotation

标注 Greedy immediate-choice boundary，例如：

\[
p_b=q.
\]

图名使用：

```text
Greedy-regret landscape
```

禁止使用：

```text
phase transition
```

除非给出正式相变定义。

## Required checks

```text
LAO* cost == Exact DP cost
Greedy chooses beneficiary before b in trap region
LAO* chooses b before beneficiaries in look-ahead region
regret increases with tau and k after trap activation
```

## Deliverables

```text
results/synthetic/trap_family/raw_runs.csv
results/synthetic/trap_family/landscape.csv
results/synthetic/trap_family/summary.json
results/synthetic/figures/trap_heatmap.pdf
results/synthetic/figures/trap_heatmap.png
```

---

# Task S9 — Certified heuristic bound

## Maximum feasible pre-action state

对 target closure \(V_t\) 和 action \(v\)：

\[
s^{\max}_{t,v}
=
V_t
\setminus
\left(
\{v\}
\cup
\operatorname{Desc}_{G_t}(v)
\right).
\]

## Analytic certified upper bound

\[
p_t^*(v)
=
\sigma\left(
\alpha_v(\beta)
+
\beta
\sum_{u\in s^{\max}_{t,v}}
w_{uv}
\right).
\]

由于：

\[
w_{uv}\ge0,
\]

对所有 feasible action states：

\[
p(v,s)\le p_t^*(v).
\]

## Bound conditions

比较：

```text
Trivial bound: p_bar(v) = 1
Analytic bound: p_bar(v) = p_t*(v)
```

## Metrics

```text
optimal value
value equality
expanded states
generated states
Bellman revisions
iterations
runtime
initial heuristic value
heuristic / optimal value ratio
expansion reduction factor
```

## Required assertions

```text
analytic-bound LAO* value == trivial-bound LAO* value
both == Exact DP value
analytic bound never violates upper-bound contract
analytic heuristic is at least as large as trivial heuristic
```

理论上：

\[
h_{\mathrm{analytic}}(s)
\ge
h_{\mathrm{trivial}}(s).
\]

## Deliverables

```text
results/synthetic/bound_ablation/raw_runs.csv
results/synthetic/bound_ablation/summary.csv
results/synthetic/bound_ablation/summary.json
```

正文只引用 aggregate expansion reduction。

完整结果进入 supplementary。

---

# Task S10 — Main figure

## Figure layout

一张 two-panel figure。

### Panel A

```text
Greedy-regret landscape on sibling-transfer trap
```

- x-axis: \(\delta\)
- y-axis: \(\tau\)
- color: Greedy normalized regret
- fixed representative \(k\)
- annotate immediate-choice boundary

### Panel B

```text
Dependence strength vs Greedy regret
```

- x-axis: \(\beta\)
- y-axis: mean Greedy normalized regret
- one line per topology
- include ECS32A semi-synthetic topology
- uncertainty band across graph / transfer seeds

## Plot requirements

- no bar chart；
- target / graph-instance equal weighting；
- show \(\beta=0\) point；
- LAO* reference line 可选，固定在 0；
- y-axis scientific notation if necessary；
- caption 必须说明 calibration keeps Random Frontier reference cost fixed。

## Deliverables

```text
results/synthetic/figures/main_synthetic_figure.pdf
results/synthetic/figures/main_synthetic_figure.png
results/synthetic/figures/main_synthetic_figure_data.csv
```

---

# Task S11 — Aggregation and freeze

## Aggregated outputs

```text
results/synthetic/final/
    all_runs.jsonl
    dependence_summary.csv
    trap_summary.csv
    bound_summary.csv
    graph_manifest.json
    oracle_manifest.json
    calibration_manifest.json
    synthetic_freeze_manifest.json
```

## Freeze manifest

记录：

```text
repository commit SHA
config hash
graph instance hashes
transfer graph hashes
oracle parameter hashes
beta grid
trap parameter grid
calibration seeds
evaluation seeds
solver version/hash
tie-break rule
DP/LAO*/Greedy implementation hashes
figure-generation script hash
output artifact hashes
UTC generation timestamp
```

## Acceptance criteria

- 所有 runs deterministic；
- 所有 LAO* values 与 Exact DP 一致；
- \(\beta=0\) 时 Greedy regret 为 0；
- calibration error 达标；
- analytic bound 无 contract violation；
- main figure 可由 frozen CSV 自动重建；
- 不手工修改 figure data；
- no post-hoc parameter-grid changes。

---

# Task S12 — Paper integration deliverables

## Section 6 main-text addition

目标篇幅：

```text
0.20-0.30 page
```

必须包含：

1. G/H separation；
2. calibrated dependence sweep；
3. trap mechanism；
4. LAO* exactness；
5. ECS32A near-myopic regime connection；
6. oracle mismatch connection。

## Locked wording

```text
Hard prerequisites determine feasibility, while soft cross-concept
transfer determines the value of ordering among feasible actions.
```

```text
The synthetic oracle family instantiates the general learner-model
interface of Section 3 with an explicit transfer structure, without
modifying the formulation or planner.
```

## Core interpretation

```text
Under a correctly specified learner model, look-ahead avoids predictable
transfer traps and becomes increasingly valuable as state dependence
strengthens. Under a mismatched learner model, exact planning can instead
amplify model error.
```

可作为 discussion closing：

```text
Planning is a lever; the learner model is its fulcrum.
```

## Supplementary content

放入：

- full beta sweep tables；
- all topology variants；
- all trap \(k\) values；
- certified-bound proof；
- bound-ablation tables；
- calibration details；
- graph-generation specification；
- regression tests；
- all provenance hashes。

## Deliverables

```text
documents/synthetic_experiment_report.md
documents/synthetic_experiment_freeze.md
paper Section 6 synthetic paragraph
supplementary synthetic section
final two-panel figure
```

---

# Final Completion Checklist

- [ ] G and H are separate.
- [ ] \(H\) contains only nonnegative transfer weights.
- [ ] Synthetic oracle is monotone in mastery state.
- [ ] Section 3 formulation is unchanged.
- [ ] Marginal difficulty is calibrated across \(\beta\).
- [ ] Random Frontier reference cost remains fixed within tolerance.
- [ ] \(\beta=0\) gives zero Greedy and LAO* regret.
- [ ] No sequence-equality assertion is used at \(\beta=0\).
- [ ] LAO* matches Exact DP on every synthetic instance.
- [ ] Trap family produces an interpretable Greedy-regret landscape.
- [ ] Dependence sweep shows when planning becomes useful.
- [ ] ECS32A semi-synthetic topology is included.
- [ ] Analytic \(p_t^*(v)\) bound is validated.
- [ ] Trivial-bound and analytic-bound LAO* values are identical.
- [ ] Expansion reduction is reported in supplementary.
- [ ] Main figure contains trap landscape and beta-regret sweep.
- [ ] All raw runs, configs, hashes, and figures are frozen.
- [ ] Section 6 addition stays within approximately 0.25 page.
