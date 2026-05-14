# Adversarial Attack and Defense

Research repository for adversarial attacks and defenses on medical imaging models (ISIC2019 skin lesion, OCT2017 retinal, CheXpert chest X-ray, COVID-Net).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Environment Setup](#environment-setup)
3. [Data Placement](#data-placement)
4. [Experiment Workflows](#experiment-workflows)
   - [CXRay — Train Clean Model + Run UAP](#cxray--train-clean-model--run-uap)
   - [ISIC2019 — Train Clean Model + Run UAP](#isic2019--train-clean-model--run-uap)
   - [OCT2017 — Train Clean Model + Run UAP](#oct2017--train-clean-model--run-uap)
   - [WaNet Backdoor Attack](#wanet-backdoor-attack)
5. [Key Design Notes](#key-design-notes)
6. [Datasets](#datasets)

---

## Repository Structure

```
Adversarial-Attack-and-Defense/
├── shared/                          # Shared modules imported by all scripts
│   ├── dataloader.py                # Dataset classes: ISICDataset, DatasetSeprateByClass, etc.
│   ├── evaluation_utils.py          # Plotting, classification report utilities
│   └── model_architectures.py       # PreActResNet definitions (architecture reference)
│
├── data/
│   ├── cxray/                       # CXRay images (place here — see Data Placement)
│   │   ├── train/NORMAL/ ...
│   │   └── test/NORMAL/ ...
│   ├── oct2017/                     # OCT2017 images (place here)
│   └── splits/                      # CSV/TXT dataset split files
│       ├── isic2019/                # ISIC2019_train.csv, ISIC2019_test.csv, ...
│       ├── oct2017/                 # OCT2017-train.csv, OCT2017-test.csv
│       └── cxray/                   # CXRAY-train.csv, CXRAY-test.csv
│
├── models/
│   └── train_baseline.py            # ← Unified clean model training (all datasets)
│
├── checkpoints/                     # Model weights — NOT committed to git
│   ├── cxray/                       # cxray_clean_baseline.pth.tar
│   ├── isic2019/                    # isic2019_clean_baseline.pth.tar
│   └── oct2017/                     # oct_clean_baseline.pth.tar
│
├── attacks/
│   ├── uap/                         # Universal Adversarial Perturbation scripts
│   │   ├── generate_uap_cxray.py
│   │   ├── generate_uap_isic_3class.py
│   │   ├── generate_uap_isic_8class.py
│   │   └── generate_uap_oct2017.py
│   ├── backdoor/
│   │   ├── wanet/                   # WaNet backdoor attack — always trains poisoned model
│   │   └── backdoor_toolbox/
│   └── covid_uap/                   # COVID-Net UAP pipeline
│
├── defenses/adversarial_training/   # AWP, OAAT, FBF, OCT adversarial training
├── evaluation/                      # Test scripts for all attack/defense combinations
├── results/                         # Experiment outputs (isic/, oct2017/, cxray/, covid19/)
├── data_tools/                      # CSV generation and image sorting utilities
├── third_party/art_/                # IBM Adversarial Robustness Toolbox fork
└── archive/                         # Backup scripts and loose artifacts
```

---

## Environment Setup

Anaconda is recommended — it handles PyTorch GPU and CUDA compatibility better than plain pip.

```bash
conda create -n adversarial-ml python=3.8
conda activate adversarial-ml

# PyTorch with GPU — replace cu118 with your CUDA version (check with: nvidia-smi)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Remaining dependencies
pip install tensorflow
pip install adversarial-robustness-toolbox
pip install kornia
pip install tensorboard tensorboardX
pip install scikit-learn matplotlib seaborn pillow tqdm pandas opencv-python
```

All scripts must be run from the **repo root** with `PYTHONPATH` set:

```bash
# Linux / macOS / Git Bash
export PYTHONPATH=.

# Windows PowerShell
$env:PYTHONPATH = "."
```

---

## Data Placement

### CXRay (Chest X-Ray)

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset and place it as:

```
data/cxray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### ISIC2019

Images live in `ISIC2019/` at repo root (already in place — do not move).

### OCT2017

Download [OCT2017](https://www.kaggle.com/paultimothymooney/kermany2018) and place as:

```
data/oct2017/
├── train/
│   ├── CNV/ DME/ DRUSEN/ NORMAL/
└── test/
    ├── CNV/ DME/ DRUSEN/ NORMAL/
```

---

## Experiment Workflows

### CXRay — Train Clean Model + Run UAP

**Step 1 — Train clean baseline classifier**

```bash
PYTHONPATH=. python models/train_baseline.py --dataset CXRAY
```

Optional overrides:
```bash
PYTHONPATH=. python models/train_baseline.py \
  --dataset CXRAY \
  --n_epochs 100 \
  --batch_size 16 \
  --lr 1e-3 \
  --device cuda
```

Output checkpoint: `checkpoints/cxray/cxray_clean_baseline.pth.tar`

The checkpoint uses key `netC` (compatible with all UAP scripts in this repo).

**Step 2 — Generate Universal Adversarial Perturbation**

```bash
PYTHONPATH=. python attacks/uap/generate_uap_cxray.py
```

This script:
- Loads the clean model from `checkpoints/cxray/cxray_clean_baseline.pth.tar`
- Selects a subset of training images to craft the UAP (10% by default, controlled by `adv_percentages`)
- Generates the perturbation using IBM ART's `UniversalPerturbation` with FGSM as the base attacker
- Evaluates attack success rate on the test set

Output: `results/cxray/` — confusion matrices, classification reports, evaluation results, and the raw noise array (`Noise/Noise_<N>.npy`)

**Configurable parameters** (edit at the top of `generate_uap_cxray.py`):

| Variable | Default | Description |
|---|---|---|
| `eps` | `[0.04]` | Max perturbation magnitude (L-inf norm) |
| `attack_eps` | `[0.0024]` | FGSM step size for UAP crafting |
| `adv_percentages` | `[0.1]` | Fraction of training images used to craft UAP |
| `targeted_attack` | `False` | Set `True` for targeted UAP, `False` for untargeted |
| `target_class` | `4` | Target class index (only used when `targeted_attack=True`) |

---

### ISIC2019 — Train Clean Model + Run UAP

**Step 1 — Train (8-class)**

```bash
PYTHONPATH=. python models/train_baseline.py --dataset ISIC2019
```

**Step 1 — Train (3-class variant, classes 0/1/2)**

```bash
PYTHONPATH=. python models/train_baseline.py \
  --dataset ISIC2019 \
  --num_classes 3 \
  --csv_train data/splits/isic2019/ISIC2019_train_012.csv \
  --csv_test  data/splits/isic2019/ISIC2019_test_012.csv
```

**Step 2 — Generate UAP**

```bash
PYTHONPATH=. python attacks/uap/generate_uap_isic_3class.py   # 3-class
PYTHONPATH=. python attacks/uap/generate_uap_isic_8class.py   # 8-class
```

---

### OCT2017 — Train Clean Model + Run UAP

**Step 1 — Train**

```bash
PYTHONPATH=. python models/train_baseline.py --dataset OCT
```

**Step 2 — Generate UAP**

```bash
PYTHONPATH=. python attacks/uap/generate_uap_oct2017.py
```

---

### WaNet Backdoor Attack

`attacks/backdoor/wanet/train.py` **always trains a backdoor-poisoned model** (this is intentional — for clean model training, use `models/train_baseline.py` instead).

Run from inside the wanet directory:

```bash
cd attacks/backdoor/wanet
python train.py --dataset CXRAY      # or ISIC2019, OCT
```

Supported `--dataset` values: `CXRAY`, `ISIC2019`, `OCT`, `COVID-19`, `Echo`, `cifar10`, `gtsrb`, `celeba`

Evaluate backdoor effectiveness:

```bash
cd attacks/backdoor/wanet
python eval.py
```

---

## Key Design Notes

**Why `models/train_baseline.py` instead of `attacks/backdoor/wanet/train.py`?**

WaNet's `train.py` is an attack script — it always trains a poisoned model. It should not be repurposed for clean training. `train_baseline.py` is the single, clear entry point for training a clean ResNet-50 classifier across all datasets.

**Checkpoint format**

All clean model checkpoints use the key `netC` for the model state dict and `optimizerC` for the optimizer. This is consistent with WaNet's checkpoint format so all UAP and evaluation scripts can load from either source.

```python
# Load in any UAP script:
checkpoint = torch.load('checkpoints/cxray/cxray_clean_baseline.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['netC'])
optimizer.load_state_dict(checkpoint['optimizerC'])
```

**PYTHONPATH requirement**

All scripts import from `shared/` as a package. Running from the repo root with `PYTHONPATH=.` ensures `from shared.dataloader import ...` resolves correctly.

---

## Datasets

| Dataset | Classes | Split files | Image location |
|---|---|---|---|
| ISIC2019 | 8 (or 3) | `data/splits/isic2019/` | `ISIC2019/` (repo root) |
| OCT2017 | 4 | `data/splits/oct2017/` | `data/oct2017/` |
| CXRay | 2 | `data/splits/cxray/` | `data/cxray/` |
| COVID-Net | 3 | `attacks/covid_uap/` | user-provided |
