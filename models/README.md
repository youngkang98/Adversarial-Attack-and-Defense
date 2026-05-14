# models/

Baseline classifier training scripts. Train these first before running any attack or defense.

## Scripts

| Script | Original name | Description |
|---|---|---|
| `train_isic_baseline.py` | `TrainModel.py` | Trains a baseline ResNet on ISIC2019 |
| `train_isic_densenet.py` | `New_UAP_CGPT.py` | Trains DenseNet-201 on ISIC2019 |
| `train_oct_baseline.py` | `train_OCT.py` | Trains a baseline model on OCT2017 |

## Usage

```bash
python models/train_isic_baseline.py
python models/train_isic_densenet.py
python models/train_oct_baseline.py
```

Expected output: model checkpoint saved to the path configured inside each script.
Set `PYTHONPATH=.` from repo root so `shared/` imports resolve correctly.
