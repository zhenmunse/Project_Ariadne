# DKT Baseline Reproduction

Week 1 status: done. Reproduced pyKT DKT on `assist2015`; Ariadne data was not used. Latest rerun: June 21, 2026.

## Environment
- Source: local clone of `https://github.com/pykt-team/pykt-toolkit`
- pyKT commit: `f766468f1d3083f737e5fde22a17a08de0d52552`
- pyKT package version: `0.0.38`
- Python: `3.11.14` from `pykt-toolkit/.venv`
- PyTorch: `2.11.0+cu128`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU

## Commands
Run from the pyKT clone root:
```powershell
.\.venv\Scripts\Activate.ps1
cd tests
$env:PYTHONPATH = (Resolve-Path ..).Path
python ..\examples\data_preprocess.py --dataset_name assist2015
$start = Get-Date
python ..\examples\wandb_dkt_train.py --use_wandb 0 --add_uuid 0
$end = Get-Date
"Training time: $($end - $start)"
```

## Configuration
- Dataset/model: `assist2015` / `dkt` / `qid`
- Fold/seed: `0` / `42`
- Hyperparameters: dropout `0.2`, emb size `200`, lr `0.001`, batch `256`, max epochs `200`, optimizer `adam`, seq len `200`

## Results

| Training AUC | Training Acc | Validation AUC | Validation Acc | Best Checkpoint Epoch | Stopped After | Training Time |
|---|---|---|---|---|---|---|
| `0.7458764078876597` | `0.7592618665465228` | `0.7317426552047004` | `0.7565037129315128` | `4` | `14` epochs | `00:01:16.5397754` |

- Training loss: `0.5057690814771449` at best epoch; `0.47876838230309204` final
- Best checkpoint: `tests\saved_model\assist2015_dkt_qid_saved_model_42_0_0.2_200_0.001_0_0\qid_model.ckpt` from epoch `4`
- Training summary: `0 dkt qid -1 -1 -1 -1 0.7317426552047004 0.7565037129315128 4`

## Deviations And Issues
- `wandb_dkt_train.py` reports training loss and validation AUC/accuracy, but not training AUC directly.
- Training AUC was computed after training by loading the best checkpoint and evaluating it on the train split.
- `data_preprocess.py` calls Unix `rm`; on Windows it prints `rm is not recognized`, but preprocessing still completes.
- `PYTHONPATH` was set to the clone root so Python imports local pyKT source.
- Weights & Biases was disabled with `--use_wandb 0`.
