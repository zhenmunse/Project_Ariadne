# DKT Baseline on ECS32A

## Reproduction Summary

The ECS32A DKT baseline is reproduced successfully:

```text
DKT + ECS32A real data
validation AUC: 0.7304367898081858
validation accuracy: 0.7106138745633006
best epoch: 29
training AUC: 0.7434304869518926
training time: 11.3780624 seconds
```

## Goal

Week 3 Tongan Task 1: convert the cleaned ECS32A Ariadne interactions into pyKT format, train a DKT baseline with the Week 1 pyKT settings, and report:

- training AUC
- validation AUC
- best epoch
- training time

Deliverables:

| file | purpose |
|---|---|
| `experiments/05_convert_to_pykt.py` | Ariadne -> pyKT conversion script. |
| `experiments/baseline_notes/dkt_ecs32a.md` | This reproduction note. |

Generated pyKT files live in the sibling pyKT checkout:

```text
../pykt-toolkit/data/output/train_valid_sequences.csv
../pykt-toolkit/data/output/test_sequences.csv
```

## Input Data

Interaction source:

```text
data/processed/cleaned_interactions.csv
```

| column | meaning | pyKT use |
|---|---|---|
| `user_id` | anonymized student ID | groups rows into one student sequence |
| `item_id` | PrairieLearn question/template ID | `questions` |
| `is_correct` | completed attempt correctness, `0/1` | `responses` |
| `timestamp` | submission time | sequence ordering only |

Concept mapping source:

```text
data/question_concept_mapping_template.csv
```

| column | meaning | pyKT use |
|---|---|---|
| `item_id` | PrairieLearn question/template ID | join key |
| `concept_id` | Ariadne DAG concept/node ID | `concepts` |

Rows without an `item_id -> concept_id` mapping are dropped because concept-level DKT needs a concept ID for every interaction.

## Conversion

Run from the Ariadne repo root:

```powershell
../pykt-toolkit/.venv/Scripts/python.exe experiments/05_convert_to_pykt.py
```

The converter:

1. reads cleaned interactions and the question-concept mapping,
2. joins on `item_id`,
3. drops unmapped rows,
4. validates `is_correct` is binary,
5. sorts by `user_id`, timestamp, and `item_id`,
6. writes pyKT's temporary 6-line raw sequence format,
7. calls pyKT's existing splitter,
8. keeps only the two DKT files needed by the training path.

Temporary raw pyKT format:

```text
uid,seq_len
questions
concepts
responses
timestamps
usetimes
```

For this baseline, timestamps and usetimes are not model inputs. The temporary file uses `NA` for those two lines, and the final CSVs omit them entirely. This avoids pyKT trying to parse ISO timestamps as integers.

Final retained columns:

```text
train_valid_sequences.csv:
fold,uid,questions,concepts,responses,selectmasks,is_repeat

test_sequences.csv:
fold,uid,questions,concepts,responses,selectmasks,is_repeat,cidxs
```

`cidxs` is kept because the installed pyKT test loader expects it.

## Data Checks

Converter summary:

```text
raw_rows=71350 kept_rows=70893 dropped_unmapped=457
users=294 items=141 concepts=34
missing_items=[9596878, 9597496, 9643536]
```

pyKT split summary:

| item | count |
|---|---:|
| student sequences after `min_seq_len=3` | 293 |
| train+valid student sequences | 235 |
| test student sequences | 58 |
| train+valid prediction points | 58,042 |
| fold 0 validation prediction points | 12,022 |
| test prediction points | 12,846 |

## Training

Run from `../pykt-toolkit/examples`:

```powershell
../.venv/Scripts/python.exe wandb_dkt_train.py --dataset_name ecs32a_ariadne --model_name dkt --emb_type qid --use_wandb 0 --add_uuid 0
```

Config entry:

```text
../pykt-toolkit/configs/data_config.json
dataset key: ecs32a_ariadne
dpath: ../data/output
train_valid_file: train_valid_sequences.csv
test_file: test_sequences.csv
```

Hyperparameters:

| parameter | value |
|---|---:|
| model | `dkt` |
| emb_type | `qid` |
| fold | `0` |
| seed | `42` |
| dropout | `0.2` |
| emb_size | `200` |
| learning_rate | `0.001` |
| batch_size | `256` |
| maxlen | `200` |

## Results

pyKT final line:

```text
fold modelname embtype testauc testacc window_testauc window_testacc validauc validacc best_epoch
0    dkt       qid     -1      -1      -1             -1             0.7304367898081858 0.7106138745633006 29
```

Required metrics:

| metric | value |
|---|---:|
| training AUC | 0.7434304869518926 |
| validation AUC | 0.7304367898081858 |
| best epoch | 29 |
| training time | 11.3780624 seconds |

Additional metrics:

| metric | value |
|---|---:|
| training accuracy | 0.707133149195493 |
| validation accuracy | 0.7106138745633006 |
| final epoch train loss | 0.5395650429701963 |

## Notes

- The Week 3 requirement is satisfied by training AUC, validation AUC, best epoch, and training time.
- No hyperparameter tuning was done; this is a reproduction baseline using the Week 1 DKT setup.
