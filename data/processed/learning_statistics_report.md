# Learning Statistics Report

## 1. 本次做了什么

本次新增并运行 `experiments/05_generate_learning_statistics.py`，把题目级 cleaned interactions 通过 `data/question_concept_mapping_template.csv` 映射到 concept-level。脚本现在只输出这个 Markdown 总结报告，不再把中间结果拆成一堆 CSV。

脚本默认输入：

- `data/anonymized_submissions_ECS32A_sq26.csv`
- `data/question_concept_mapping_template.csv`
- `data/processed/cleaned_interactions.csv`
- `data/ecs32a_concepts_required_full_v1.csv`
- `data/ecs32a_dag_edges_required_full_v1.csv`
- `data/ecs32a_teaching_order_required_full_v1.csv`

重新生成命令：

```bash
python experiments/05_generate_learning_statistics.py
```

## 2. 一句话结论

当前 `mapping` 表内部没有发现结构性冲突：没有重复 `item_id`、没有一题多 concept、没有 concept-label 冲突。但是 cleaned 数据里确实有 3 个真实提交过的 `item_id` 不在 mapping 中：`9596878`、`9597496`、`9643536`。这不是 cleaned 文件坏了，而是 mapping 没有告诉脚本这 3 道题属于哪个 concept。

这 3 道题合计对应 `457` 条 cleaned interaction。它们会进入 cleaned 总量统计，但不会进入 concept-level、student-concept、mastery、struggle、DAG bottleneck 等所有依赖 `concept_id` 的统计。

## 3. 核心统计结果

### 3.1 Mapping 质量

- Mapping 行数：`141`
- unique mapped questions：`141`
- covered concepts：`34`
- 只有 1 道题的 concept 数：`9`
- DAG/catalog 中 0 道题覆盖的 concept 数：`27`
- duplicate item_id count：`0`
- item_id mapped to multiple concepts count：`0`
- concept_id with multiple question_label count：`0`
- mapping 中有但 cleaned 从未出现的题：`0`
- cleaned 中有但 mapping 不存在的题：`3`（9596878; 9597496; 9643536）

### 3.2 Cleaned Interactions 总体

- 总 interaction 数：`71,350`
- 学生数：`296`
- cleaned 中 unique questions：`144`
- 总体正确率：`64.73%`
- 时间范围：`2026-04-02T20:10:47+00:00` 到 `2026-06-11T19:51:19+00:00`
- 覆盖天数：`69.99` 天
- 每个学生提交次数：mean `241.05`，median `208`，min `1`，max `1,288`
- 成功 join 到 concept-level 的 interaction：`70,893`
- 未能 join 的 cleaned interaction：`457`

### 3.3 Concept-Level 概览

- concept-level 汇总中的 concept 总数：`61`
- 有实际 interaction 的 concept 数：`34`
- teaching order vs actual first-seen order Spearman：`0.937`

最困难 concept（按 success_rate 从低到高，至少有 interaction）：

| concept_id | question_label | attempts | unique_students | success_rate | mastery_rate |
| --- | --- | --- | --- | --- | --- |
| 57 | dictionaries counting join filter | 68 | 68 | 23.53% | 23.53% |
| 23 | robust input loop patterns | 702 | 227 | 39.46% | 76.21% |
| 47 | list methods append pop sort | 988 | 273 | 48.08% | 44.32% |
| 38 | robust input with try except | 310 | 252 | 49.03% | 50.00% |
| 44 | file reading line iteration | 1,534 | 267 | 51.43% | 51.69% |
| 13 | if else conditional blocks | 2,283 | 275 | 51.82% | 65.82% |
| 10 | advanced operators and string ops | 1,268 | 291 | 52.13% | 78.35% |
| 16 | guardian pattern and demorgan | 7,311 | 282 | 52.52% | 58.87% |
| 43 | file handles write close | 1,484 | 268 | 54.11% | 60.45% |
| 3 | string literals and print | 271 | 271 | 55.72% | 55.72% |

Top bottleneck concept（按 bottleneck_score 排序）：

| concept_id | question_label | out_degree | mastery_rate | struggle_index | bottleneck_score |
| --- | --- | --- | --- | --- | --- |
| 13 | if else conditional blocks | 7 | 65.82% | 0.509 | 3.563 |
| 17 | while loop basics | 7 | 74.02% | 0.452 | 3.166 |
| 52 | parameters return values side effects | 4 | 45.20% | 0.567 | 2.266 |
| 29 | error categories syntax runtime semantic | 3 | 73.91% | 0.542 | 1.625 |
| 40 | strings indexing slicing | 5 | 77.15% | 0.280 | 1.402 |
| 16 | guardian pattern and demorgan | 2 | 58.87% | 0.665 | 1.329 |
| 8 | type conversion | 3 | 70.79% | 0.439 | 1.317 |
| 12 | boolean values and comparisons | 3 | 68.09% | 0.437 | 1.310 |
| 6 | variables assignment and naming | 6 | 81.16% | 0.208 | 1.246 |
| 3 | string literals and print | 4 | 55.72% | 0.310 | 1.240 |

### 3.4 Assessment-Level 概览

| assessment_title | mapped_questions | mapped_concepts | attempts | unique_students | success_rate |
| --- | --- | --- | --- | --- | --- |
| Practice Quiz 1 | 6 | 4 | 5,379 | 274 | 63.21% |
| Practice Quiz 2 | 16 | 5 | 15,707 | 253 | 61.83% |
| Practice Quiz 3 | 11 | 4 | 10,991 | 245 | 65.39% |
| Practice Quiz 4 | 11 | 5 | 7,827 | 255 | 63.61% |
| Practice Quiz 5 | 15 | 7 | 9,062 | 249 | 60.54% |
| Practice Quiz 6 | 13 | 6 | 7,186 | 253 | 74.67% |
| Quiz 1 | 7 | 5 | 1,903 | 274 | 72.10% |
| Quiz 2 | 8 | 4 | 2,135 | 267 | 73.16% |
| Quiz 3 | 8 | 4 | 2,150 | 269 | 69.72% |
| Quiz 4 | 8 | 5 | 2,112 | 264 | 67.61% |
| Quiz 5 | 7 | 7 | 1,862 | 267 | 72.61% |
| Quiz 6 | 7 | 3 | 1,846 | 268 | 58.99% |
| Review Quiz 1 | 6 | 5 | 1,437 | 241 | 67.36% |
| Review Quiz 2 (Final taken on June 3) | 10 | 9 | 695 | 72 | 45.90% |
| Review Quiz 3 (Final taken on June 11) | 8 | 5 | 601 | 77 | 44.59% |

### 3.5 Unmapped Cleaned Questions

#### 这到底是什么意思

这里说的 unmapped，不是指这些提交记录无效，也不是指 cleaned_interactions 坏了。意思是：这些 `item_id` 在学生提交记录里真实存在，但 `question_concept_mapping_template.csv` 没有给它们分配 `concept_id`。所以脚本不知道它们属于哪个知识点，只能把它们排除在 concept-level、student-concept、bottleneck、DAG violation 等统计之外。

#### 证据链

我检查了三层文件，结果如下：

- `cleaned_interactions.csv` 里有这 3 个 item_id，说明它们是清洗后保留下来的有效提交。
- `question_concept_mapping_template.csv` 里没有这 3 个 item_id。
- `question_concept_mapping_final.csv` 里也没有这 3 个 item_id。
- template 是从 final mapping 生成的，所以根因不是统计脚本 join 错了，而是 final mapping 源头没有收这 3 道题。

影响可以这样理解：

- 如果这些题本来就不属于学习路径，例如只是额外 review/final 检查题，那么排除它们是合理的。
- 如果这些题其实应该属于某个 concept，那么现在的 concept 统计会少算这些练习记录，对应 concept 的 attempts、success_rate、mastery_rate、struggle_index 都会偏差。
- 报告里的 question-level 统计只覆盖 mapping 里的 141 道题，因此这 3 个 unmapped item 不进入 question-level 结果。
- 报告里的 cleaned 总量统计仍包含它们，因为那部分统计的是 cleaned 全量数据。

| item_id | attempts | unique_students | success_rate | first_timestamp | last_timestamp |
| --- | --- | --- | --- | --- | --- |
| 9596878 | 314 | 171 | 63.06% | 2026-04-02T20:10:47+00:00 | 2026-06-11T07:12:58+00:00 |
| 9643536 | 73 | 73 | 0.00% | 2026-06-11T17:31:17+00:00 | 2026-06-11T19:51:19+00:00 |
| 9597496 | 70 | 70 | 0.00% | 2026-06-03T16:56:15+00:00 | 2026-06-03T18:52:19+00:00 |

按 assessment 汇总后，缺口集中在这些地方：

| assessment_title | unmapped_item_ids | completed_rows |
| --- | --- | --- |
| Practice Quiz 1 | 9596878 | 221 |
| Practice Quiz 2 | 9596878 | 93 |
| Review Quiz 2 (Final taken on June 3) | 9597496 | 70 |
| Review Quiz 3 (Final taken on June 11) | 9643536 | 73 |

回查 raw 数据后，这些 item_id 的 assessment 来源如下：

| item_id | assessment_title | raw_rows | completed_rows | raw_unique_students | raw_completed_success_rate |
| --- | --- | --- | --- | --- | --- |
| 9596878 | Practice Quiz 1 | 410 | 221 | 199 | 47.51% |
| 9596878 | Practice Quiz 2 | 169 | 93 | 118 | 100.00% |
| 9597496 | Review Quiz 2 (Final taken on June 3) | 102 | 70 | 70 | 0.00% |
| 9643536 | Review Quiz 3 (Final taken on June 11) | 84 | 73 | 73 | 0.00% |

特别注意：`9596878` 在 raw 数据里同时出现在 Practice Quiz 1 和 Practice Quiz 2；但 cleaned_interactions 只有 `item_id`，没有 `assessment_title`，所以 cleaned 层面只能把它们合并成同一道 item 的 314 条记录。

#### 为什么可能会这样

基于当前仓库文件，能确认的是：这 3 个 ID 没有进入 `question_concept_mapping_final.csv`。不能确认的是它们的具体题面，因为 raw CSV 只有 `question_id`、assessment、时间、正确性等列，没有题目标题或题面内容。

最可能的原因有三类：

1. `question_concept_mapping_final.csv` 是人工/半人工整理的教学题映射，整理时漏掉了每个相关 assessment 中额外出现的一道题。
2. `9596878` 被 PrairieLearn 在 Practice Quiz 1 和 Practice Quiz 2 里复用；如果 mapping 是按某个静态题目清单整理的，这种跨 assessment 复用题很容易漏掉。
3. `9597496` 和 `9643536` 出现在 final/review assessment 中，而且正确率都是 0；它们可能是额外 review/final 题、占位/特殊题，或者没有被纳入 learning path 的题。当前文件不足以判断它们应该映射到哪个 concept。

因此，不应该直接给它们硬填 concept。正确做法是回到 PrairieLearn/课程题目元数据中查这三个 `question_id` 的题面或 question directory，再决定是否补进 mapping。

### 3.6 路径顺序

teaching order 与实际首次出现顺序差距最大的 concept：

| concept_id | question_label | teaching_order | avg_actual_order | order_gap | students_seen |
| --- | --- | --- | --- | --- | --- |
| 52 | parameters return values side effects | 53 | 26.655 | -26.345 | 281 |
| 56 | dictionary iteration query patterns | 57 | 31.175 | -25.825 | 275 |
| 47 | list methods append pop sort | 48 | 22.447 | -25.553 | 273 |
| 57 | dictionaries counting join filter | 58 | 32.485 | -25.515 | 68 |
| 55 | dictionaries basics key value lookup | 56 | 30.928 | -25.072 | 250 |
| 42 | string methods parsing validation | 43 | 22.228 | -20.772 | 272 |
| 36 | nested for loops | 37 | 16.493 | -20.507 | 284 |
| 43 | file handles write close | 44 | 23.910 | -20.090 | 268 |
| 44 | file reading line iteration | 45 | 24.940 | -20.060 | 267 |
| 38 | robust input with try except | 39 | 19.940 | -19.060 | 252 |

## 4. 逻辑校验与已知假设

- 脚本会校验 mapping 必须是一题一行；如果出现重复 `item_id` 或一个 `item_id` 对多个 `concept_id`，会直接报错停止。
- `is_correct` 必须能转换为 0/1；`timestamp` 必须能解析为时间，否则脚本会停止。
- concept-level 统计只使用成功 join 到 mapping 的 interactions；未 join 的题已集中写在本报告的 unmapped 小节。
- `mastered` 的当前定义是：学生在该 concept 的最后一次尝试 `is_correct = 1`。
- `first_try_success_rate` / `last_try_success_rate` 是按 `user_id + concept_id` 聚合后的首次/最后一次尝试均值。
- `struggle_index = 0.4 * (1 - first_try_success_rate) + 0.3 * normalized(avg_attempts_per_student) + 0.3 * (1 - last_try_success_rate)`。
- DAG violation 的当前定义包括两类：先接触 target 再接触 prereq；或者接触 target 但从未接触 prereq。
- 路径顺序使用 UTC 解析后的 timestamp 排序；如果同一学生多个 concept 首次出现时间完全相同，用 `concept_id` 做稳定 tie-break。
- daily attempts 使用原始 timestamp 字符串中的日期，保留课程本地日期视角；first/last timestamp 输出为 UTC ISO 时间。
- Practice vs Quiz 只识别标题形如 `Practice Quiz N` 和 `Quiz N` 的 assessment。
- DAG/catalog 中没有题目覆盖的 concept 会保留在 concept summary 中，但 attempts/rates 保持为空，避免误读为 0% 或 100%。

## 5. 结论

当前可以确认的是：mapping 表内部没有发现结构性冲突，也就是没有 duplicate item_id、没有一题多 concept、没有 concept-label 冲突。不能说“数据完全没有任何问题”，因为仍有 3 个 cleaned item_id 没有 concept 映射。

这 3 个 unmapped item 的含义很具体：学生确实提交过这些题，cleaned_interactions 也保留了这些有效提交；但是 mapping 表没有告诉脚本这些题属于哪个 concept。因此它们能进入 cleaned 总量统计，却不能进入 concept、student-concept、DAG bottleneck 等依赖 concept_id 的统计。

下一步判断标准也很明确：如果 `9596878`、`9597496`、`9643536` 是课程学习路径里应该建模的题，就应该补到 mapping 表并重新跑脚本；如果它们只是额外 review/final 或不用于路径规划的题，那保持排除是合理的，但报告里已经明确记录了它们没有进入 concept-level 统计。
