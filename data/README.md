# data/splits/

Dataset split CSV and TXT files for reproducible train/test partitioning.

## Naming convention

```
{DATASET}_{split}[_{classes}][_{type}].{ext}

DATASET   — ISIC2019 | OCT2017 | CXRAY
split     — train | test
classes   — 012 (3-class) | 02 (2-class) | 0124 (4-class)
type      — Adversarial (adversarial examples split)
ext       — .csv | .txt
```

## Subdirectories

| Dir | Files | Description |
|---|---|---|
| `isic2019/` | 18 files | ISIC2019 skin lesion train/test/adversarial splits |
| `oct2017/` | 3 files | OCT2017 retinal train/test + adversarial train split |
| `cxray/` | 2 files | CheXpert chest X-ray train/test splits |

## Regenerating splits

Use `data_tools/generate_csv_from_folder.py` to regenerate CSV splits from a dataset folder,
and `data_tools/separate_images_by_class.py` to sort images into per-class subdirectories.
