# third_party/

Third-party libraries included directly in the repository.

## art_/

A fork of the [IBM Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

Used by `attacks/covid_uap/covid_eval_framework.py` to wrap COVID-Net as an ART classifier
and run standardised attack/evaluation pipelines.

> ⚠️ **Do not modify files inside `art_/`** — this is a vendored dependency.
> If you need a newer ART version, replace the entire directory or install via pip:
> ```bash
> pip install adversarial-robustness-toolbox
> ```
> and update import paths accordingly.

Upstream repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox
