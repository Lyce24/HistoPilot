# Clinical insights

The fourth roadmap stage contains **Clinical utility** and **Model interpretation**. The suggested path is experiments → predictors → model evaluation → clinical utility → model interpretation. Each saved report or attention study retains its source IDs and content identities. Interpretation also accepts any compatible slide, so a completed clinical analysis is optional.

## Clinical utility

Select a completed model evaluation, choose the analysis unit and positive class, and review the analysis before saving. The module reads checksummed predictions from the completed evaluation. It does not retrain the model, fit a calibrator, optimize a test-set threshold, or modify the frozen predictor.

The default unit and threshold come from the evaluation. Slide and patient results remain separate: patient analysis uses the saved patient aggregation and requires verifiable patient membership. Unlabeled rows are counted and excluded from outcome-dependent statistics. Multiclass discrimination and decision curves use the selected class versus all other classes; the multiclass Brier score is reported separately with its convention stated.

The report includes:

- Brier score, prevalence-based reference score, Brier skill score and log loss.
- ROC/AUC and precision–recall/average precision, with tied prediction scores handled together.
- Calibration bins with counts, observed outcome rates, mean predicted probabilities, expected and maximum calibration error, and observed/expected outcomes.
- Confusion counts, sensitivity, specificity, predictive values, accuracy, balanced accuracy, F1 and likelihood ratios at the chosen threshold, plus operating curves across a selected range.
- Decision-curve net benefit compared with treating everyone and treating no one; standardized net benefit and net interventions avoided per 100 relative to treating everyone.
- Clinical impact curves showing the numbers classified high risk, true positives, false positives and missed positives per 100 observations.

A descriptive threshold override is labeled explicitly. Clinical threshold ranges must reflect the intended decision and relative consequences of false-positive and false-negative decisions. Net benefit is `TP/N − FP/N × threshold/(1−threshold)`; it is not an observed treatment effect. Decision curves describe the selected cohort and its prevalence. Outcome sampling and repeated patient observations can limit transportability and uncertainty interpretation.

Undefined statistics appear as unavailable with explanations. Wilson intervals apply to sensitivity, specificity and predictive values only where the analysis units support independent patient observations; slide rows with repeated or unknown patients do not receive independence-based intervals. The report does not provide bootstrap intervals, calibration-slope fitting or external clinical validation. Save separate analyses to compare choices without overwriting evidence. JSON and CSV exports preserve the analysis and its source lineage. ROC/PR display and CSV points are bounded to 2,001 points, with sampling recorded explicitly; AUC and average precision still use every prediction. The focused decision-curve view clips values below −0.10 with a visible explanation; switch to the full range or inspect the numeric table/export for those values.

## Model interpretation

Interpretation uses separate pages for selection, computation, results and a focused slide viewer:

1. **Choose model and shared features.** Select an existing ABMIL refit or ensemble predictor, a frozen feature bundle, and its original features or a verified pack. The slide folder is inherited from that bundle's dataset; interpretation does not ask for the folder again. External compatible bundles use their own datasets. Older datasets can reuse a single shared parent folder recorded by their slides; an ambiguous or inaccessible location directs users back to Datasets.
2. **Choose slides.** The gallery recursively lists slide images with lazy thumbnails. Search matches names and relative paths across the entire dataset folder before pagination. Click thumbnails or checkboxes to select up to 128 slides across searches and pages, then choose **Continue with … selected**. Selection alone starts no computation. Missing features or coordinates and ambiguous names remain visible with an explanation.
3. **Prepare attention.** Each selected slide shows its own status. Already completed studies are skipped, queued or running jobs are reused, and missing attention is computed. Changing only CPU or memory reservations does not invalidate completed or active matching attention. Every selected job is tracked, including slides outside the visible area. Failed slides remain actionable without discarding successful slides. Results open only when all selected slides have verified completed attention. A lost response retains the exact request for a safe retry, including after reload.
4. **Attention results.** This page contains only the selected slides. Open one card to enter **Attention {slide}**, with one focused attention viewer and a **Back to results** action.

Back navigation preserves the model, feature representation, search and selection. The current selection and batch references survive reload within the browser session. Returning to selection does not automatically send users forward again. Changing model, bundle, representation or evidence links clears the selection and its result context; an unresolved request must first be recovered so its accepted jobs are not lost. Scientific inputs and geometry remain frozen when retrying failed jobs; execution resource reservations can be adjusted.

Native and registered legacy OceanPath packs supply their own indexed `features.bin` and `coords.bin` rows. The worker reads the selected slide's exact offset and complete patch bag directly from the pack, without reopening original HDF5 files. Bundle selection still uses the application's source freshness checks. Original feature mode reads the bundle's verified per-slide HDF5 features and coordinates. Features must match the predictor's encoder, dimensions and dtype; a different compatible bundle can contain external slides. Selecting slides does not extract new features.

Coordinate arrays must be integer `N × 2` XY origins in level-0 slide pixels, with the same patch order as finite `N × D` feature vectors. Patch width and height come from extraction or verified pack metadata; supply both explicitly under the optional footprint controls only when metadata is missing. Conflicting geometry, incompatible encoders, mismatched rows, out-of-bounds origins, inaccessible paths and changed inputs block inference. No approximate filename matching or automatic row truncation is used. Existing manually prepared studies remain available in **Open a saved attention study**, and their original API workflow remains supported.

Execution uses the isolated training environment and durable compute jobs, retaining the complete slide bag and exact frozen checkpoints. Increase the resource reservation when a full bag will not fit; jobs continue after leaving the page. A refit uses one checkpoint. An ensemble computes every frozen member and averages normalized per-patch attention and probabilities; individual member maps remain available.

ABMIL attention is **class-independent pooling attention**, normalized within each slide. It describes relative model weighting of patches, not tumor probability, class-specific evidence or a causal explanation. The display distinguishes visual color scaling from raw attention, which remains available in exported artifacts. Open a slide from the results page, then use the member selector, opacity and pan/zoom controls to inspect the overlay. Viewport changes preserve the original slide coordinate system and do not renormalize the underlying scientific attention. Overlapping patches display the strongest attention, matching patch inspection. The viewer displays at most 100,000 patches per viewport and explicitly labels partial overlays; zoom in to see every patch in a smaller region. Full-map downloads retain all patches.

The detailed attention viewer also offers **Top 10** or **Top 20** patches, ranked by normalized attention across the **entire slide**, independently of the current viewport, heatmap percentile filter or display limit. Numbered boxes mark their exact locations. The same numbers identify original-color patch cards; select a box or card to inspect its crop, coordinates and raw weight. The selected ensemble mean or member determines the ranking. Equal weights are ordered by original patch index, and overlapping patches remain individually selectable through their cards. Slides with fewer patches show only the available patches.

Patch crops use authenticated output coordinates and the frozen level-0 footprint to read the original slide. Crops preserve aspect ratio and clip at slide boundaries; they show the source image region, which may differ from the extractor's resized model input. Crop requests remain bounded and use the same slide-change checks as the viewer. Patch ranking does not rerun inference or change attention normalization. The slide occupies most of the desktop workspace beside a compact, scrollable patch rail. The highest-ranked patch is selected immediately. **Hide patches** gives the slide the full width, and **Expand view** opens a larger workspace with Escape to return. At narrower content widths the patch rail moves below the slide. On mobile, selection brings the crop into view; **Center selected patch** returns to its slide location. If ranking or crop verification fails, stale evidence is hidden and an explicit retry is available.

**Fit patch coverage** frames the bounding box of every extracted patch, including low-attention patches, with a small margin. This is the initial view when coverage metadata arrives before the user moves the viewer. **Fit slide** restores the complete slide, and manual pan/zoom is preserved when delayed metadata arrives. Coverage uses exact level-0 footprints clipped to the slide edges.

Batch overview cards show up to 5,000 patches and label partial previews explicitly. Open the detailed viewer and zoom to inspect all patches within smaller regions. Folder scans are bounded to 20,000 entries, 10,000 slides, 24 nested levels and five seconds; exceeding a limit asks for a smaller folder instead of returning a silently incomplete inventory.

Older JSON-only maps support ranked patches through a 64 MiB viewer limit, matching the historical writer limit. A compact cache retains only authenticated top-patch rows, and legacy parsing is serialized to bound memory. Larger JSON-only maps require current array artifacts for viewing; their original JSON export remains available.

New outputs include checksummed numeric arrays for bounded, memory-mapped viewport reads, alongside full JSON maps. Studies accept at most 128 slides, 2 million patches per slide and 8 GiB of feature data per slide. A synchronous batch has a 45-second preparation budget; already accepted slides are retained and remaining slides receive an explicit limit result. Submit fewer slides if preparation exceeds the budget. These limits are checked before publication where possible, and full-bag resource requirements are included in the preview.

The viewer reads bounded image regions using OpenSlide, with a small-raster fallback through Pillow. It does not decode a whole WSI at base resolution. Install the optional viewing dependencies in the service environment with `uv sync --extra imaging` (or `pip install 'histopilot[imaging]'`). Standard compute inference still uses the separately configured training interpreter. Authentication applies to image requests and artifact downloads as well as metadata APIs.

Saved clinical analyses protect their evaluation and predictor references during cleanup. Attention studies protect their predictor, selected feature bundle and source dependencies, and any linked clinical/evaluation records. Active attention jobs must stop before cleanup. Archiving or moving records to Trash preserves source slides, feature files, outputs and logs.

The user starts or restarts the HistoPilot server. Development and verification do not launch it automatically.

## References

- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Reporting and interpreting decision curve analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6261531/)
- [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html)
- [OpenSlide Python API](https://openslide.org/api/python/)
