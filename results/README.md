# results/

Experiment output files (classification reports, confusion matrix images, evaluation logs).

## Subdirectories

| Dir | Original name | Contents |
|---|---|---|
| `isic/` | `Result/` | ISIC2019 3-class results for adversarial training variants (advtrain, madrytrain, w_advtrain) |
| `oct2017/` | `OCT2017_result/` | OCT2017 UAP attack and post-defense evaluation results |
| `cxray/` | `CXRAY_result/` | CheXpert UAP attack and post-defense evaluation results |
| `covid19/` | `COVID19_Result/` | COVID-Net UAP attack and post-defense evaluation results |

## File naming convention

```
{metric}_{n_samples}_att{att_size}_eps{epsilon}.{ext}

metric      — classification_report | confusion_matrix | evaluation_results | psnr
n_samples   — number of test samples used
att_size    — perturbation amplitude
epsilon     — perturbation budget
```

> **Note:** `OCT/` at repo root contains 250 adversarial example images — left in place pending
> confirmation that no script reads from `./OCT/` as input data.
