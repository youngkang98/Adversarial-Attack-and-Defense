# shared/

Shared Python modules imported by scripts across multiple pipeline stages.

Set `PYTHONPATH=.` from repo root before importing.

## Modules

| Module | Original name | What it provides |
|---|---|---|
| `dataloader.py` | `dataloader.py` | `ISICDataset`, `OCTDataset`, `CXRAYDataset` — PyTorch Dataset classes for each medical imaging dataset |
| `model_architectures.py` | `PreActBottleNeck.py` | `PreActResNet18/34/50/101/152` and `PreActBottleneck` — 6 model architecture definitions |
| `evaluation_utils.py` | `Utils.py` | `plot_confusion_matrix()`, `classification_report` helpers |

## Import pattern

```python
from shared.dataloader import ISICDataset
from shared.model_architectures import PreActResNet18
from shared.evaluation_utils import plot_confusion_matrix
```

## Used by

- `attacks/uap/generate_uap_isic_*.py` — imports `dataloader`, `model_architectures`
- `defenses/adversarial_training/adv_train_awp.py` — imports `dataloader`, `model_architectures`
- `evaluation/test_uap_isic*.py` — imports `dataloader`, `evaluation_utils`
- `models/train_isic_baseline.py` — imports `dataloader`
