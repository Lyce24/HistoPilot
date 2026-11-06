# TCGA-NSCLC to CPTAC-NSCLC: rerun the public benchmark

This folder rebuilds the study behind [benchmarks](../../../docs/benchmarks.md) from spec files: LUAD versus LUSC per patient, trained by five-fold cross-validation on TCGA-NSCLC and applied to CPTAC-NSCLC as an external cohort, on UNI2-h patch features.

| File | What it is |
| --- | --- |
| `nsclc_dataset.csv` | One row per slide: `Slide_ID`, `Patient_ID`, `Subtype` (LUAD or LUSC), `Cohort` (TCGA or CPTAC) and `Feature_Path`, relative to the feature release. 946 TCGA and 453 CPTAC slides, all public identifiers. |
| `convert_uni2h_release.py` | Writes the release's `(1, N, 1536)` float32 features as `[N, 1536]` float16 HDF5 files with their coordinates, the layout HistoPilot attaches. |
| `specs/` | One spec per stage, in order. Paths point at `/data/tcga-nsclc/`; edit them if your files live elsewhere. |
| `recompute.py` | Recomputes every number of the benchmark from the project folder and the label table, with scikit-learn instead of HistoPilot, and checks them against the stored values. |

## 1. Get the features

The features are MahmoodLab's [UNI2-h feature release](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features), gated on Hugging Face under CC-BY-NC-ND 4.0: non-commercial research only, with attribution. **They may not be redistributed**, so they are not in this repository and the public demo carries none. Request access, then download the TCGA LUAD and LUSC archives and the CPTAC LUAD and LSCC archives, and convert them:

```bash
python convert_uni2h_release.py nsclc_dataset.csv /path/to/UNI2-h-features /data/tcga-nsclc/features
cp nsclc_dataset.csv /data/tcga-nsclc/
```

## 2. Build the study

Start HistoPilot with the folder as a data root (`bash serve.sh --data-root /data/tcga-nsclc`), create a project, select it with `histopilot use PROJECT`, then from this folder:

```bash
histopilot dataset import --from specs/1-dataset.yaml --tag nsclc-v1
histopilot targets create --from specs/2-targets.yaml --tag tcga-train-cptac-test
histopilot features attach --from specs/3-features-tcga.yaml --tag uni2h-tcga
histopilot features attach --from specs/3-features-cptac.yaml --tag uni2h-cptac
histopilot bundle create --from specs/4-bundle-tcga.yaml --tag bundle-tcga
histopilot bundle create --from specs/4-bundle-cptac.yaml --tag bundle-cptac
histopilot experiment create --from specs/5-experiment.yaml
histopilot experiment freeze "MIL aggregators from scratch"
histopilot experiment start "MIL aggregators from scratch" --wait
histopilot cohort create --from specs/6-cohort-cptac.yaml --tag cptac-external
histopilot apply template --experiment "MIL aggregators from scratch" --cohort @cptac-external -o apply.yaml
histopilot apply run --from apply.yaml --wait
```

Each command previews its change and asks first. Training is five folds of five models, 25 runs of 8 to 22 minutes each on one RTX A5000 (5.2 hours, one run at a time). Even mean pooling takes 8 minutes a fold, because each fold reads every slide's features from its own HDF5 file; a [feature pack](../../../docs/user-guide.md#2-slide-features) reads them from one memory-mapped file instead.

The experiment trains ABMIL, CLAM-SB, TransMIL, DSMIL and mean pooling with HistoPilot's default recipes, as a controlled comparison with ABMIL as the reference, over the same patient-level folds and one training seed (42). Within each fold, early stopping and the saved checkpoint follow validation AUROC on 15% of the training patients, held out from fitting; the predictions on the five test folds are the out-of-fold development estimate, and every arm is reported. Each model's five fold models are averaged as a fold ensemble and applied once, label-blind, to CPTAC. The cohort leaves out two CPTAC slides whose released features use 512-pixel level-0 patches, a different geometry from the rest.

## 3. Check dataset shift and recompute

```bash
histopilot run list --cohort @cptac-external
histopilot run shift-analyze RUN          # once: runs share the analysis
histopilot run shift-performance RUN
python recompute.py /path/to/the/project/folder nsclc_dataset.csv
```

`recompute.py` exits nonzero if any recomputed AUROC differs from the value HistoPilot stored.

## 4. More seeds, more architectures

With one training seed, the controlled comparison's paired intervals resample patients and hold each trained model fixed, so they cannot say how much an arm would move if it were retrained. To measure that, and to bring in the rest of HistoPilot's aggregators, write a comparison design with every arm against ABMIL, then copy the inputs from `specs/5-experiment.yaml` and set `trainingSeeds: [42, 43, 44]`:

```bash
histopilot experiment template --preset baseline \
  --compare-model clam_sb --compare-model clam_mb --compare-model transmil \
  --compare-model dsmil --compare-model dtfd --compare-model acmil --compare-model nnmil \
  --compare-model oceanpath_mamba2 --compare-model mean_pool --compare-model max_pool \
  -o design.yaml
histopilot experiment create --from design.yaml
```

That is 11 arms × 3 seeds × 5 folds = 165 runs: about a day and a half on one RTX A5000 if the new arms train as fast as the five measured here, 12.5 minutes a run on average. A [feature pack](../../../docs/user-guide.md#2-slide-features) of the TCGA features shortens it.
