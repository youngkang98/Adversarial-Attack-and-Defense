# evaluation/

Test scripts that measure attack effectiveness and defense robustness.

## Script → Attack/Defense mapping

| Script | Original name | Evaluates |
|---|---|---|
| `test_uap_isic.py` | `Test_UAP_ISIC.py` | UAP attack on ISIC2019 baseline model |
| `test_uap_oct.py` | `Test_UAP_OCT.py` | UAP attack on OCT2017 baseline model |
| `test_uap_generic.py` | `test_UAP.py` | Generic UAP evaluation |
| `test_uap_tensorflow.py` | `test_TF_UAP.py` | TensorFlow UAP evaluation |
| `test_uap_isic_postdefense.py` | `Test_UAP_ISIC_training_defense.py` | UAP on ISIC2019 **after** adversarial training |
| `test_uap_covid_postdefense.py` | `Test_UAP_COVID-Net_training_defense.py` | UAP on COVID-Net **after** adversarial training |
| `test_cifar_advtrain.py` | `Test_CIFAR_Advtrain.py` | Robustness of FBF-trained CIFAR-10 model |
| `test_isic2018_defense.py` | `ISIC2018_testing_defense.py` | Defense evaluation on ISIC2018 |
| `test_attention_isic.py` | `Test_Att_ISIC.py` | Attention-based model evaluation on ISIC2019 |

## Usage

```bash
python evaluation/test_uap_isic.py
python evaluation/test_uap_isic_postdefense.py
```
