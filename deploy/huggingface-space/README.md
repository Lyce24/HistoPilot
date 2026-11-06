---
title: HistoPilot
emoji: 🔬
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
short_description: Read-only demo of a pathology MIL workspace on public data
license: mit
---

# HistoPilot demo

[HistoPilot](https://github.com/Lyce24/HistoPilot) is a research workspace for computational pathology. It takes a slide table to cross-validated, compared and applied multiple instance learning (MIL) models, and keeps every step frozen, versioned and reproducible.

This Space is read-only and serves two studies:

- **TCGA-NSCLC public study**, a real study in the full application. Five MIL aggregators were trained on 946 TCGA patients for LUAD versus LUSC and applied to 451 CPTAC patients as an external cohort. Browse the datasets, splits and features, the experiment's results and controlled comparison, and each external run's intervals, recalibration, clinical utility and dataset shift. Download a predictor's model card. [Benchmark write-up](https://github.com/Lyce24/HistoPilot/blob/main/docs/benchmarks.md).
- **BLCA demo**, a synthetic walkthrough of every stage, including attention maps.

Nothing you open is saved and no compute runs. Slide images, features and model weights are not included. To work with your own slides, install HistoPilot on your workstation; see the [quick start](https://github.com/Lyce24/HistoPilot#quick-start).

The real study uses UNI2-h patch features from MahmoodLab's [feature release](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features) ([UNI](https://www.nature.com/articles/s41591-024-02857-3), CC-BY-NC-ND 4.0), for non-commercial research. This Space shows results derived from them, never the features.
