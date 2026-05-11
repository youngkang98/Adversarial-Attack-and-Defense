# Adversarial Attack and Defense

Research repository for adversarial attacks and defenses on medical imaging models (ISIC2019 skin lesion, OCT2017 retinal, CheXpert chest X-ray, COVID-Net).

## Datasets

| Dataset | Task | Split files |
|---|---|---|
| ISIC2019 | 8-class skin lesion classification | `data/splits/isic2019/` |
| OCT2017 | 4-class retinal OCT classification | `data/splits/oct2017/` |
| CheXpert (CXRAY) | Chest X-ray classification | `data/splits/cxray/` |
| COVID-Net | COVID-19 detection | `attacks/covid_uap/` |

Raw dataset images live in `ISIC2019/` and `OCT/` at repo root — these are not moved.

## Pipeline

```
models/          ← 1. Train baseline classifiers
attacks/         ← 2. Generate adversarial examples (UAP, backdoor, COVID-UAP)
defenses/        ← 3. Apply adversarial training defenses
evaluation/      ← 4. Evaluate attack/defense effectiveness
```

## Quickstart

Run all scripts from the **repo root** with `PYTHONPATH` set so `shared/` resolves:

```bash
# Linux / macOS / Git Bash
export PYTHONPATH=.

# Windows PowerShell
$env:PYTHONPATH = "."
```

Example — generate ISIC 3-class UAP:
```bash
python attacks/uap/generate_uap_isic_3class.py
```

## Directory Structure

```
├── shared/                     # Shared modules (dataloader, model architectures, eval utils)
├── data/splits/                # CSV/TXT dataset split files
├── models/                     # Baseline model training scripts
├── attacks/
│   ├── uap/                    # Universal Adversarial Perturbation scripts
│   ├── backdoor/               # WaNet and backdoor-toolbox
│   └── covid_uap/              # COVID-Net UAP pipeline
├── defenses/adversarial_training/
├── evaluation/                 # Test scripts for all attack/defense combinations
├── data_tools/                 # CSV generation and image sorting utilities
├── results/                    # Experiment outputs (isic, oct2017, cxray, covid19)
├── third_party/art_/           # IBM Adversarial Robustness Toolbox fork
└── archive/                    # Backup scripts and loose artifacts
```

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this repository, please cite the relevant attack/defense papers referenced in each subdirectory's README.
