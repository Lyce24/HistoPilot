# Manifest placement verdict

**Recommendation: a Manifest page is not a required main workflow stage.** Keep scientific manifests as authoritative stored evidence, and make their raw contents an optional **Advanced → Provenance** view with a download action. The main workflow should present the scientific decisions, readable input summaries, validation findings, and results that users need to act on.

Reviewed against the current canonical checkout on 2026-09-17. This is a source-based product verdict; it changes no interface, scientific records, or server state.

## What the current interface actually does

The main roadmap already contains eight scientific modules and **no Manifest module**: Datasets, Slide features, Targets & splits, Experiments, Test cohorts, Evaluate models, Clinical utility, and Model interpretation. Experiment planning likewise uses **Inputs → Batches → Review & submit**, followed by Runs and Results. Neither requires opening raw JSON. See the [roadmap definitions](../web/src/lib/roadmap.ts#L29) and [experiment navigation](../web/src/components/ExperimentNavigation.tsx#L4).

Most local workflows already place technical evidence behind optional disclosures:

| Current location | Current presentation | Recommended treatment |
| --- | --- | --- |
| [Datasets → Saved dataset details](../web/src/pages/LocalDataset.tsx#L400) | **View saved provenance** expands IDs, source mapping, hashes, and artifact metadata. | Keep dataset counts, source names, mapping, and identity warnings readable; retain raw provenance as optional evidence. |
| [Targets & splits → saved protocol](../web/src/pages/LocalProtocol.tsx#L622) | Target/split summaries, findings, partitions, and preflight precede **View saved configuration** JSON. | Preserve summaries and actionable findings; standardize the optional disclosure label. |
| [Slide features → source detail](../web/src/pages/LocalFeatures.tsx#L369) and [bundle detail](../web/src/components/FeatureBundleLibrary.tsx#L84) | **Configuration provenance** and **Frozen bundle evidence** expose raw records. The bundle already renders its JSON only when expanded. | Keep encoder, representation, dimensions, coverage, precision, included packs, and verification status visible; retain optional raw evidence and adopt lazy rendering consistently. |
| [Experiments → detail](../web/src/pages/LocalExperiments.tsx#L335) | **Exact input history & configuration snapshots** is collapsed below the workflow. Batch settings and [predictor snapshots](../web/src/components/ExperimentPredictors.tsx#L82) have similar disclosures. | Keep the normal Inputs/Batches/Review/Runs/Results flow. Group full snapshots in Advanced; preserve readable input and predictor lineage. |
| [Evaluate models → result](../web/src/pages/LocalModelEvaluation.tsx#L195) | Prediction downloads and next actions precede **Saved inputs, settings and results** JSON and a record export. | Keep scientific result downloads readily available; put the complete machine-readable record in Provenance. |
| [Saved attention study](../web/src/components/InterpretationStudy.tsx#L14) | The model/evaluation chain, job state, and viewer precede **Slide files and frozen provenance**. | Preserve model, slide, coordinate/feature compatibility, and attention limitations beside the viewer; leave full manifests optional. |
| [Legacy synthetic Provenance page](../web/src/pages/Results.tsx#L618) | A prominent **Example manifest** JSON panel sits beside a readable lineage chain. | This is the clearest candidate for demotion: retain the chain and synthetic labels, collapse the JSON, and keep its example download. |

The legacy example must not be mistaken for the current BLCA workflow. [Application routing](../web/src/App.tsx#L103) sends BLCA modules into their self-contained walkthrough; the [demo generator](../histopilot/application/blca_demo.py#L1051) supplies an empty `exampleManifests` list. Local sidebar tools also differ from the legacy synthetic Provenance navigation ([sidebar selection](../web/src/App.tsx#L249)).

## What belongs in the main view

Users should be able to answer these questions without reading JSON:

- **What am I using?** Named dataset/protocol/bundle versions, cohort membership and counts, target/class order, prediction unit, encoder and feature representation.
- **What will happen?** Model and material training settings, fold/seed/configuration counts, ensemble/refit policy, selected-checkpoint epoch budget, compute request, and next action.
- **Can I trust these inputs?** Coverage, verified or stale evidence, patient-identity limitations, development/test overlap, incompatible settings, and the action needed to resolve each blocker.
- **Where did this result come from?** Clickable links to its experiment, frozen inputs, predictor, evaluation, and relevant slide or checkpoint evidence; scientific outputs and readable status.

Long hashes, full membership arrays, internal field names, complete environment/code snapshots, execution receipts, and full file inventories belong in optional Provenance. Relevant source-folder selection, missing-source errors, and recovery instructions stay visible when they affect the current task. Moving raw evidence must never hide a validation failure or turn “not checked” into “verified.”

## Why manifests must remain in the backend

The current scientific store hashes manifest contents together with artifact metadata to establish content identity ([`_hash_content`](../histopilot/storage/scientific.py#L259)). Dataset verification checks the saved `manifest.json` against its expected record and rejects changed or missing evidence ([dataset integrity checks](../histopilot/storage/scientific.py#L1195)). Training, predictor, evaluation, and archive code use those frozen identities and records; displaying them as a main page is not part of that contract.

The useful user action is **Review & freeze/submit** against a readable summary. The service creates, validates, and retains the corresponding scientific records. A second required “Manifest” stop adds no independent scientific decision when it repeats those settings in JSON.

## Implementation and compatibility implications

Treat a future change as presentation work: standardize optional Provenance disclosures, add scoped manifest downloads where absent, and lazy-render large raw records. Do not add a Manifest stage or require opening a disclosure before progressing.

Preserve API fields, frozen schemas, IDs, content hashes, artifact names, worker plans, export payloads, and resume checks. No data migration is needed for moving existing displays. Keep existing `#provenance` access working for its supported legacy/local destinations; do not replace historical evidence with current defaults. Any downloaded full record should retain its existing synthetic or execution-status labeling.

Acceptance should verify that the normal BLCA/local workflow remains usable without opening JSON, every existing validation blocker and lineage link is still reachable, and a record's downloaded evidence still matches its saved source. Browser review should check disclosure accessibility and large-record behavior. No UI implementation or browser verification was performed for this verdict.
