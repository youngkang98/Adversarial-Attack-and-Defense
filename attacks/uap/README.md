# attacks/uap/

Universal Adversarial Perturbation (UAP) generation scripts.
A UAP is a single image-agnostic perturbation `δ` (||δ||_p ≤ ε) that causes misclassification on most inputs.

## Dataset → Script mapping

| Script | Original name | Dataset | Notes |
|---|---|---|---|
| `generate_uap_generic.py` | `UAP.py` | Generic | Base UAP implementation |
| `generate_uap_isic_3class.py` | `ISIC_UAP_3class.py` | ISIC2019 (3-class) | Classes 0,1,2 |
| `generate_uap_isic_8class.py` | `ISIC_UAP_8class.py` | ISIC2019 (8-class) | All 8 classes |
| `generate_uap_oct2017.py` | `OCT2017_UAP.py` | OCT2017 | Retinal OCT |
| `generate_uap_cxray.py` | `CXRAY_UAP.py` | CheXpert | Chest X-ray |
| `generate_uap_tensorflow.py` | `New_TF_UAP.py` | — | TensorFlow-based UAP |

## Usage

```bash
# From repo root
python attacks/uap/generate_uap_isic_3class.py
python attacks/uap/generate_uap_oct2017.py
```
