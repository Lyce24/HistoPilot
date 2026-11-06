# Benchmarks on public data

HistoPilot run end to end on a public task: LUAD versus LUSC, trained by cross-validation on TCGA-NSCLC and applied once to CPTAC-NSCLC, an external cohort from other hospitals and scanners. Every number below is HistoPilot's own output. Each one was then recomputed with scikit-learn by [`recompute.py`](../examples/benchmarks/tcga-nsclc/recompute.py), which reads only the saved prediction files and the label table, and every AUROC matched to four decimals. The [public demo](deployment.md#packaged-projects) serves this study, read-only, in the full application.

## Study

| | |
| --- | --- |
| Task | Non-small cell lung cancer subtype per patient: LUSC (positive class) versus LUAD |
| Development | TCGA-NSCLC: 946 patients, one diagnostic slide each (468 LUAD, 478 LUSC). Five patient-level stratified folds; within each fold, 15% of the training patients are held out for early stopping and checkpoint choice. One training seed (42). |
| External cohort | CPTAC-NSCLC: 451 patients, one slide each (239 LUAD, 212 LUSC). Each model's five fold models are averaged as a fold ensemble and applied once, without reading labels, then scored by the service against the cohort's frozen labels. Two slides whose released features use 512-pixel patches were left out. |
| Features | UNI2-h (ViT-H/14) patch features, 256 px at 20×, from MahmoodLab's [feature release](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features) (CC-BY-NC-ND 4.0), stored as float16. They are not redistributed. |
| Models | ABMIL (the reference arm), CLAM-SB, TransMIL, DSMIL and mean pooling, with HistoPilot's default recipes: AdamW at 3 × 10⁻⁴, weight decay 10⁻⁴, up to 40 epochs with patience 8, bags of up to 4,096 patches. |
| Intervals | 95% patient bootstrap, 2,000 resamples. |

## Results

| Model | TCGA, out of fold: AUROC | CPTAC, external: AUROC | CPTAC balanced accuracy | Peak GPU memory | Minutes per fold |
| --- | --- | --- | ---: | ---: | ---: |
| ABMIL (reference) | 0.973 (0.963–0.982) | 0.964 (0.948–0.977) | 0.896 | 777 MiB | 9.8 |
| CLAM-SB | 0.976 (0.966–0.985) | 0.966 (0.952–0.979) | 0.901 | 719 MiB | 13.0 |
| TransMIL | 0.965 (0.954–0.976) | 0.971 (0.956–0.984) | 0.932 | 2,378 MiB | 22.4 |
| DSMIL | 0.979 (0.968–0.987) | 0.972 (0.958–0.983) | 0.917 | 559 MiB | 8.5 |
| Mean pooling | 0.966 (0.955–0.976) | 0.949 (0.930–0.965) | 0.881 | 726 MiB | 8.0 |

Peak GPU memory is the largest allocation of any fold, on one RTX A5000; minutes per fold are the median over five folds. All 25 folds took 5.2 hours, run one at a time. The intervals in this table are recomputed by `recompute.py`; the app draws its own bootstrap resamples, so its intervals can differ in the third decimal.

- **Every attention-based aggregator transfers.** CPTAC AUROCs fall within 0.01 of their cross-validated values, and TransMIL's rises. Mean pooling, the attention-free baseline, loses the most (0.966 to 0.949).
- **The arms are close.** Paired with ABMIL on the same 946 patients' out-of-fold predictions, every arm is within 0.01 AUROC. Only DSMIL's paired interval excludes zero (0.006 higher than ABMIL), and after Holm adjustment for the four contrasts no difference is significant.

| Arm | ABMIL − arm, out-of-fold AUROC | Paired 95% interval | p | Holm-adjusted p |
| --- | ---: | --- | ---: | ---: |
| CLAM-SB | −0.003 | −0.009 to 0.002 | 0.25 | 0.29 |
| TransMIL | +0.007 | −0.002 to 0.019 | 0.11 | 0.29 |
| DSMIL | −0.006 | −0.011 to −0.001 | 0.03 | 0.10 |
| Mean pooling | +0.007 | −0.001 to 0.015 | 0.10 | 0.29 |

These are the app's numbers (Results → Controlled comparison). The paired intervals resample patients with the same draws for both models and hold each trained model fixed, so with one training seed, how much an arm would move if it were retrained is unmeasured. Rerunning with more training seeds measures it.

## Dataset shift

The [dataset shift](methods.md#dataset-shift) analysis compares the 451 CPTAC slides with the 946 TCGA development slides in UNI2-h feature space. The predictors share one training batch, so all five runs share the analysis. Its CPU task read the features of all 1,397 slides in 2 to 3 minutes.

- **Domain AUROC 0.999 (0.998–1.000).** A linear model on mean slide features names nearly every slide's source.
- **52% of CPTAC slides (236 of 451) lie beyond the development range**, where about 1% of development slides do by construction. The median CPTAC slide's distance to its nearest TCGA slides is 0.50; the median among TCGA slides is 0.23.

Performance by atypicality: patients ordered by the mean percentile of their slides' distance to the development data, and cut into thirds.

| Model | Least atypical third | Middle third | Most atypical third | Within the range | Beyond the range |
| --- | --- | --- | --- | --- | --- |
| ABMIL | 0.976 (0.955–0.992) | 0.982 (0.964–0.996) | 0.934 (0.884–0.971) | 0.975 (0.953–0.991) | 0.952 (0.920–0.976) |
| CLAM-SB | 0.975 (0.948–0.993) | 0.984 (0.967–0.997) | 0.929 (0.878–0.967) | 0.975 (0.952–0.991) | 0.951 (0.920–0.975) |
| TransMIL | 0.986 (0.969–0.997) | 0.986 (0.967–0.999) | 0.928 (0.871–0.970) | 0.982 (0.962–0.996) | 0.950 (0.915–0.978) |
| DSMIL | 0.969 (0.931–0.993) | 0.985 (0.968–0.998) | 0.945 (0.902–0.977) | 0.969 (0.942–0.991) | 0.964 (0.939–0.984) |
| Mean pooling | 0.956 (0.917–0.986) | 0.965 (0.939–0.987) | 0.918 (0.868–0.959) | 0.949 (0.914–0.976) | 0.940 (0.907–0.968) |

The thirds hold 151, 149 and 151 patients; 215 patients lie within the range and 236 beyond it. Intervals are from `recompute.py`; the service reports these groups without them.

**Reading it.** CPTAC is a different domain in feature space, yet the subtype signal carries over: overall AUROC stays at 0.95–0.97. The cost of the shift concentrates where it should. Every model scores lowest on the most atypical third, 0.918–0.945 against 0.956–0.986 on the other two thirds. With about 150 patients per third the intervals overlap, so this is a pattern across five models rather than a significant drop for any one. It is also where a reader of this cohort's errors should start.

## Reproduce it

[`examples/benchmarks/tcga-nsclc`](../examples/benchmarks/tcga-nsclc/) has the label table, the spec file of every stage, the feature conversion script and `recompute.py`, with the commands in order. You need access to the gated feature release; nothing else is private.

## Limits

- One training seed per arm, so seed-to-seed variation is unmeasured; see above.
- TCGA and CPTAC both come from research consortia. Neither cohort is a consecutive clinical series, and neither result is a clinical performance claim.
- Features come from one public release, as released. A different extraction (magnification, tissue detection, stain handling) moves these numbers, and the dataset shift analysis is the place to see how far.
