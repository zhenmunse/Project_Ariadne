# Junyi Topology Diagnostic Report
- Date / 数据来源与版本 / 下载渠道: 2026-07-15 / Junyi Academy Math Practicing Log (to Jan. 2015), PSLC 2015 版 / [USTC mirror](https://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/) (`junyi.rar`)
- 原始 exercises 数 / 清洗后 nodes / edges (raw & reduced): 837 rows / 835 nodes / 979 raw valid edges, 978 cycle-cleaned DAG edges, 894 transitive-reduction edges
- Dangling prerequisite 边数 / 自环数 / 删除的环边数（见 removed_edges.csv）: 0 / 2 / 1

## 全图
- Poset width: 301. 计算口径为清洗后 DAG 先做 `nx.transitive_reduction`（835 nodes, 894 edges），再对该 reduction 的 transitive closure 做最大二分匹配，取 `n - matching_size`。
- Sink 数量: 292

## Closures（附 junyi_closures.csv）
- T-sink: closure size min/median/max = 1 / 78 / 212; width min/median/max = 1 / 8 / 18; ideal count min/median/max = 2 / 96,608 / >1e7。
- T-vol: closure size min/median/max = 1 / 18.5 / 77; width min/median/max = 1 / 4 / 9; ideal count min/median/max = 2 / 1,598 / 386,405。
- Ideal count 触发 guard 的 closure 数量: 68（全部来自 T-sink；T-vol 为 0）。

## mΔ/J* proxy（附 junyi_mdelta.csv）
- 全部 closure 的 `mdelta_med_ratio`: min / median / max = 0.002854 / 0.056357 / 0.124874（即 0.2854% / 5.6357% / 12.4874%；基于 241 个有可用 Δ 的 closure，另 71 个为 NA）。
- coverage 中位数: 0.783974。这里 coverage 是 restricted closure `V_t_obs` 内 Δ 非 NA 的比例。
- 可观测性: `obs_fraction` min/median/max = 0 / 0.851351 / 1；58 个 closure 的 `obs_fraction < 0.7`，见附录 A。
- Restricted-closure 口径: `V_t_obs = {v in V_t: first-attempt success rate exists and > 0}`；`m`、`J_star_proxy`、两个 mΔ ratio 均仅在 `V_t_obs` 上计算，`m_total` 保留原 closure 大小。不平滑、不插值、不删除 closure 行。
- ECS32A 对照: width ≤ 4, ideals ≤ 47, mΔ/J* < 0.2% (frozen oracle)

## 异常与备注
- Exercise table 有两个重复 name：`matrix_mul_two`、`matrix_app_fruit_oil`；各有 1 行重复。按 name 去重时保留最早 `creation_date`，若日期相同则保留文件序第一行；行数由 837 变为 835。
- Prerequisite name 匹配率为 100%。发现并丢弃 2 个自环；没有 dangling prerequisite。
- Cycle 清洗只删除 1 条边：`simplifying_radicals -> adding_and_subtracting_radicals`。删除数量不异常；清洗后 DAG 为 978 edges，transitive reduction 为 894 edges。
- Problem log 实际列名为 `user_id`, `exercise`, `correct`, `time_done`；共读取 25,925,992 rows，`time_done` 为 numeric，无法解析比例为 0。
- 835 个 closure exercise 中，113 个没有 first-attempt success rate，12 个 first-attempt success rate 为 0；二者均从每个 closure 的 `V_t_obs` 排除，并由 `m_total` 与 `obs_fraction` 显式反映。
- 12 个 zero-rate exercise 的 first-attempt 支持度均低于 30；附录 B 同时报告 first-attempt 数和总 attempt 数，供人工检查，不据此做平滑或插值。
- 本报告中的 Δ 是基于首次尝试与 direct-prerequisite mastery proxy 的 diagnostic proxy，不是训练 oracle 的 Δ。

## 附录 A：obs_fraction < 0.7 的 closures

| target | m | m_total | obs_fraction |
|---|---:|---:|---:|
| `adding_decimals_0.5_new` | 0 | 1 | 0.000 |
| `adding_decimals_1` | 0 | 1 | 0.000 |
| `adding_decimals_2_new` | 0 | 1 | 0.000 |
| `angle types` | 0 | 1 | 0.000 |
| `angle-types` | 0 | 1 | 0.000 |
| `application_of_linear_equations_2` | 0 | 1 | 0.000 |
| `attributs_of_shapes` | 1 | 5 | 0.200 |
| `cc_and_cubic_meter_conversion` | 0 | 1 | 0.000 |
| `cell_bctest_cap` | 0 | 1 | 0.000 |
| `classification_bctest_cap` | 0 | 1 | 0.000 |
| `common_factors_of_a_polynomial_1` | 0 | 1 | 0.000 |
| `comparing_objects` | 0 | 1 | 0.000 |
| `comparing_size` | 0 | 2 | 0.000 |
| `compound_numbers_world_problems` | 0 | 1 | 0.000 |
| `consistency_bctest_cap` | 0 | 1 | 0.000 |
| `cube_rectangular_and_regular_tetrahedron` | 0 | 1 | 0.000 |
| `cube_volume_calculation` | 0 | 1 | 0.000 |
| `cuboid_volume_calculation` | 0 | 1 | 0.000 |
| `difference_between_solid_and_plane_graphics` | 0 | 1 | 0.000 |
| `distributive_property_with_variables` | 0 | 1 | 0.000 |
| `ecology_bctest_cap` | 0 | 1 | 0.000 |
| `environment_bctest_cap` | 0 | 1 | 0.000 |
| `evolution_bctest_cap` | 0 | 1 | 0.000 |
| `factoring_difference_of_squares_4` | 0 | 1 | 0.000 |
| `factoring_polynomials_3` | 0 | 1 | 0.000 |
| `fraction_and_per_swap_divide` | 0 | 1 | 0.000 |
| `fractions_and_percentage_of_exchange` | 0 | 1 | 0.000 |
| `fractions_cut_and_copy` | 0 | 1 | 0.000 |
| `g_and_kg_word_problems` | 0 | 1 | 0.000 |
| `inheritance_bctest_cap` | 0 | 1 | 0.000 |
| `meters_and_centimeters_conversion` | 0 | 1 | 0.000 |
| `multiplying_fractions_and_whole_numbers_word_problems` | 0 | 1 | 0.000 |
| `number_sense_length_l3` | 0 | 3 | 0.000 |
| `number_sense_weight_L1` | 0 | 1 | 0.000 |
| `number_sense_weight_l3` | 0 | 3 | 0.000 |
| `nutrition_bctest_cap` | 0 | 1 | 0.000 |
| `order_by_length` | 0 | 1 | 0.000 |
| `ordering_objects` | 0 | 1 | 0.000 |
| `radius_angle_by_percentage` | 0 | 1 | 0.000 |
| `recording_life _problem_by_expressions` | 0 | 1 | 0.000 |
| `reproduction` | 0 | 1 | 0.000 |
| `reproduction_bctest_cap` | 0 | 1 | 0.000 |
| `scale_see_compare_see` | 0 | 1 | 0.000 |
| `seq-arithmetic-sequences-l2` | 0 | 6 | 0.000 |
| `seq-understaning-arithmetic-sequence` | 0 | 1 | 0.000 |
| `seq-understaning-arithmetic-sequence-1` | 0 | 1 | 0.000 |
| `seq-understaning-arithmetic-sequence-2` | 0 | 1 | 0.000 |
| `seq_sequence_exam_3` | 0 | 5 | 0.000 |
| `seq_sequence_pattern_2` | 0 | 4 | 0.000 |
| `short_divide_gcf_and_lcm` | 0 | 1 | 0.000 |
| `side_length_cube_and_relationship_with_volume_cuboid` | 0 | 1 | 0.000 |
| `simple_calculation_of_composite_volume` | 0 | 1 | 0.000 |
| `simplifying_rational_expressions_1` | 0 | 1 | 0.000 |
| `simplifying_rational_expressions_2` | 0 | 1 | 0.000 |
| `solving_problems_with_line_plots_1` | 0 | 2 | 0.000 |
| `tenfold_and_hundred_fold` | 0 | 1 | 0.000 |
| `transportation_bctest_cap` | 0 | 1 | 0.000 |
| `understand_percentage` | 0 | 1 | 0.000 |

## 附录 B：first-attempt success rate = 0 的 exercises

| exercise | first attempts | total attempts |
|---|---:|---:|
| `application_of_linear_equations_2` | 1 | 41 |
| `common_factor_advance` | 3 | 34 |
| `completing_the_square_1_new` | 2 | 7 |
| `distributive_property_with_variables` | 2 | 5 |
| `dividing_decimals_int_to_int_0.5` | 3 | 8 |
| `dividing_decimals_int_to_int_0.7` | 3 | 7 |
| `matrix_app_fruit_oil` | 1 | 3 |
| `matrix_mul_two` | 4 | 8 |
| `multiplying_fractions_and_whole_numbers_word_problems` | 2 | 2 |
| `skip_counting_by_10s` | 1 | 1 |
| `subtraction_within_20` | 1 | 8 |
| `variables_word_problems_2` | 1 | 63 |
