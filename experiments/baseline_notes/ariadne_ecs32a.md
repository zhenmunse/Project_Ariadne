# MonotonicOracle on ECS32A

## Summary

The Ariadne `MonotonicOracle` was trained successfully on the cleaned ECS32A
dataset and evaluated on a student-level validation split.

| metric | value |
|---|---:|
| final training MSE | 0.121643 |
| validation MSE | 0.123673 |
| validation RMSE | 0.351672 |
| validation MAE | 0.303330 |
| validation AUC | 0.774873 |
| validation accuracy | 0.798618 |

Training and validation MSE are close, so this run shows no obvious overfitting.

## Files

| file | purpose |
|---|---|
| `experiments/09_prepare_oracle_data.py` | Builds the graph, sessions, and student-level splits. |
| `experiments/02_train_oracle.py` | Trains `MonotonicOracle` and saves the checkpoint. |
| `experiments/10_evaluate_oracle.py` | Evaluates the checkpoint on validation students. |
| `data/processed/oracle_ckpt.pt` | Trained Oracle checkpoint. |

Generated pickle files are excluded from Git.

## Data Preparation

Inputs:

```text
data/processed/cleaned_interactions.csv
data/question_concept_mapping_final.csv
data/ecs32a_dag_required_full_v1.json
```

The preprocessing step:

1. maps each `item_id` to its Ariadne `concept_id`,
2. drops interactions without a teaching-concept mapping,
3. sorts interactions by student and timestamp,
4. aggregates consecutive interactions on the same concept into sessions,
5. builds `(history, target_node, label)` Oracle samples,
6. splits students into train, validation, and test sets with `seed=42`.

Data summary:

| item | value |
|---|---:|
| cleaned interactions | 71,350 |
| retained mapped interactions | 70,893 |
| retained students | 294 |
| mapped items | 141 |
| observed concepts | 34 |
| DAG nodes | 61 |
| DAG edges | 134 |

Student-level split:

| split | students | samples |
|---|---:|---:|
| train | 236 | 25,089 |
| validation | 29 | 3,103 |
| test | 29 | 3,367 |

The full 61-node DAG is retained even though only 34 concepts have mapped
questions. The JSON DAG is used as the authoritative source because the edge
CSV omits the `2 -> 4` edge.

## Oracle Input

Each sample contains:

```text
history: list[(node_id, average_score)]
target_node: node_id
label: target session average_score in [0, 1]
```

The resulting `OracleDataset` tensors have these shapes:

```text
x:      [batch, 61, 2]
target: [batch]
mask:   [batch, 61]
y:      [batch]
```

## Training

| setting | value |
|---|---:|
| hidden dimension | 64 |
| dropout | 0.3 |
| learning rate | 0.001 |
| epochs | 50 |
| batch size | 128 |
| optimizer | Adam |
| loss | MSE |
| device | CUDA |
| model parameters | 8,387 |

Training MSE decreased from `0.221722` in epoch 1 to `0.121643` in epoch
50. All 50 monotonicity checks passed.

The final MC Dropout sanity check reported:

```text
mean probability: 0.745212
variance:         0.00582313
base time:        60.0
```

## Validation

Regression metrics use all 3,103 validation sessions. AUC and accuracy use the
2,170 sessions whose average-score labels are exactly `0` or `1`.

| metric | value |
|---|---:|
| MSE | 0.123673 |
| RMSE | 0.351672 |
| MAE | 0.303330 |
| AUC | 0.774873 |
| accuracy | 0.798618 |

The validation result confirms that the trained Oracle generalizes to held-out
students under the current split. The test split remains unused and is not
required for this pipeline verification; it should be reserved for a final
model comparison if hyperparameters or model selection are later based on the
validation result.

## Reproduction

From the repository root:

```powershell
python experiments/09_prepare_oracle_data.py
python experiments/02_train_oracle.py
python experiments/10_evaluate_oracle.py
```

## Limitations

- The training script saves the final epoch rather than the best validation
  checkpoint.
- Training time was not recorded in this run.
- Only 34 of the 61 DAG concepts have observed question interactions.
- AUC and accuracy cover binary session labels only; continuous session labels
  are evaluated with MSE, RMSE, and MAE.
