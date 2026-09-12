# Offline HistoPilot interface verification

Actual React page/components and application CSS were built into a file:// HTML harness. Scientific/query responses were synthetic; no project service was started or contacted, and no training was launched.

Passed interactions:

- Targets opens on development-data selection; only one main setup step is visible.
- Typed protocol name, selected diagnosis and explicit positive class, navigated Back/Continue: values persisted.
- Typed split seeds `42, 123`; the review summary displayed both seeds and five-fold design.
- Loaded a mock saved protocol, previewed assignments, and reloaded it: returned to data selection and cleared the previous preview.
- Cleared Maximum epochs with actual keyboard Ctrl+A/Backspace: text remained blank, prior numeric state was not submitted, validation blocked.
- Typed 20 epochs, patience 7, and scientific-notation learning rate `3e-4` successfully.
- Set minimum epochs to 25 with maximum 20, collapsed Advanced settings, checked settings: submission blocked, details reopened, invalid control received focus. Correcting the value allowed validation.
- Whole-bag mode stores null and displays every-patch instructions; switching back restores a custom sampled limit of 1024.
- Data mock inspection progressed to ID mapping; Slide_ID and Patient_ID were inferred and optional columns/patient-table sections remained collapsed.
- Feature acquisition progressed through mock coverage inspection to step 3, with optional packing and mandatory validation visible.
- Model setup automatically selected the one compatible protocol/bundle pair. Check inputs & continue opened Batches without requiring a separately saved experiment input record.
- Actual batch editor with Concurrent runs=6, one selected GPU and Runs per GPU=1 displayed Configured limit: 1 concurrent run.
- Corrected feature-library CSS scope verified: desktop has sidebar only; 600px mobile has dropdown only. No horizontal overflow.
- Browser errors: none. Blocked/attempted network requests: zero.

Issue found and fixed:

The first feature-selector simplification unintentionally affected frozen-bundle library navigation. LocalFeatures.css now scopes the single-column selector to bundle preparation; the frozen library keeps its responsive sidebar/dropdown layout.

Screenshots:

- targets-data.png
- targets-review.png
- model-batch.png
- training.png (isolated shared controls)

Artifacts are temporary verification fixtures under /tmp/histopilot-guided-ui-check. This is an interaction/visual check; it does not verify real file contents or backend training behavior.
