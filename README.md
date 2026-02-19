# Project Ariadne v0.2

面向知识图谱的自适应学习路径规划研究原型。

## 运行顺序

| Step | 脚本 | 功能 |
|------|------|------|
| 1 | `python -m src.preprocess` | 原始日志 → 概念级 session + mastery 标签 |
| 2 | `python -m src.train_oracle` | 训练 Oracle（概念成功概率预测） |
| 3 | `python -m src.planner` | DP Planner 在 DAG 上求最优学习路径 |

## 配置

所有超参数集中在 `configs/config.yaml`。

## 目录结构

```
configs/          # 配置文件
src/              # 源代码
data/processed/   # 预处理输出
experiments/      # 模型权重与实验日志
```
