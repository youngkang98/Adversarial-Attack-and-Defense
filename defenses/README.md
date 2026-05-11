# defenses/

Adversarial defense implementations. All defenses here are based on **adversarial training** —
augmenting the training set with adversarial examples so the model learns to be robust.

## adversarial_training/

| Script | Original name | Method | Dataset |
|---|---|---|---|
| `adv_train_awp.py` | `adversarial_training_awp.py` | AWP (Adversarial Weight Perturbation) | ISIC2019 |
| `adv_train_oaat.py` | `adversarial_training_oaat.py` | OAAT (One-vs-All Adversarial Training) | ISIC2019 |
| `adv_train_fbf_cifar10.py` | `FBF_adv_train_CIFAR10.py` | FBF (Free Fast FGSM) | CIFAR-10 |
| `adv_train_oct2017.py` | `OCT2017_advtrain.py` | Standard adversarial training | OCT2017 |

### Method summary

- **AWP** — perturbs model weights during training to flatten the loss landscape
- **OAAT** — trains separate robust classifiers per class in a one-vs-all fashion
- **FBF** — fast free adversarial training using a single backward pass

## Usage

```bash
python defenses/adversarial_training/adv_train_awp.py
python defenses/adversarial_training/adv_train_oaat.py
```
