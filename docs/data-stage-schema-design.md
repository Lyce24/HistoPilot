# Data-stage design: standard identifiers, typed attributes, and cohort exploration

Status: proposed design. This defines the next importer contract; it is not implemented by the current source-folder picker. It refines the [Bladder priority review](bladder-priority-review.md).

**Recommendation:** standardize the canonical data model and downloadable templates, while accepting existing CSV/XLSX headers through a reviewed mapping. Reserve a small set of identity/link fields. Keep clinical and pathology attributes extensible, describe them in a persisted data dictionary, and choose targets/model inputs separately for each experiment. Support both combined and separate tables, normalizing them into the same internal representation.

## 1. What should be fixed

Use `Slide_ID` and `Patient_ID` exactly in HistoPilot templates and canonical CSV/XLSX exports. These are user-facing interchange names; application objects may use `slide_id` and `patient_id`. Original source headers and cell values remain in the source snapshot and mapping provenance.

| Canonical field | Meaning and constraint | Required when |
| --- | --- | --- |
| `Slide_ID` | Stable string identifier for one slide; unique in the canonical Slides table | Every included slide record needs a resolved ID before dataset freeze. Discovery may suggest the filename stem, subject to review. |
| `Patient_ID` | Stable string identifier for a person; many slides can link to one patient | Required and unambiguous for every selected slide before patient-grouped execution. Required and unique in a supplied Patients table. |
| `Slide_Path` | Reference to the slide file on an allowed server source | Optional column. A reviewed root/filename mapping can resolve it instead; pixels may be unavailable for an existing-feature workflow. |
| `Case_ID` | Optional case grouping whose meaning is documented | Only if the study has a defined case entity. It is not a substitute for patient identity. |
| `Specimen_ID` | Optional tissue specimen identity | Only when available and explicitly mapped. Do not equate case and specimen automatically. |
| `Block_ID` | Optional tissue block identity | Only when available; its parent relationships must be consistent. |

`Patient_ID` is a required field in the canonical slide schema and template, but it can contain unresolved values during import. Represent an unresolved link internally as `null` with an identity status, never as one shared fake patient such as `UNKNOWN`. This distinction allows the current bladder inventory to be imported without inventing patients. Dataset freeze records unresolved identity; patient-safe execution remains blocked. The user can add a verified crosswalk later and create a new dataset version.

The simplest valid hierarchy is Patient → Slide. Specimen, block and case metadata enrich it when known; users should not invent intermediate entities just to fill a template. Existing domain records that require a specimen will need to represent optional hierarchy explicitly during importer implementation.

Other column names are not universally mandatory. Do not require generic `Label`, `Age`, `Gender`, `Grade`, `Fold`, or `Site` columns. A disease preset can suggest names and meanings, but a different name does not make valid data unusable. In particular, `WHO_2022` and `WHO_1973` are attributes of a bladder dataset, not hardcoded global HistoPilot fields.

A prepared source header `De ID` can map to `Slide_ID` without rewriting the user's file. Case-insensitive/underscore aliases can suggest mappings; collisions or ambiguous matches require review. Mapping to `Patient_ID` requires confirmation of identity semantics, not just a matching spelling.

### Identifier and relationship rules

Treat IDs as text. Preserve leading zeros, case and meaningful punctuation; never truncate at the first dot or lowercase IDs silently. Flag whitespace/normalization changes for review. Excel formatting cannot recover zeros that were already lost in a numeric cell: the importer should surface that uncertainty, not invent the original identifier.

IDs are scoped to a project/dataset identity namespace. If two sources reuse an ID for different slides or people, require an explicit crosswalk/namespace resolution; do not merge by demographic similarity. Aliases to the same physical artifact must be audited separately from identifier equality.

The canonical Slides table has one row per slide and at most one patient link per slide. The canonical Patients table has one row per patient. Repeating `Patient_ID` across slides is expected; repeating `Slide_ID` in the slide master is an ambiguity. Multiple annotations or feature stores must not multiply slide-master rows.

## 2. Classify columns along separate axes

Avoid one dropdown containing “label / patient_features / slide_features.” Those choices combine different concepts. A patient attribute can be a target, a predictor, a filter or unused; the same is true of a slide attribute.

| Axis | Question answered | Suggested choices |
| --- | --- | --- |
| **Entity level** | Whose observation is this? | Patient, slide, or a defined case/specimen. Preserve unresolved source-row scope until meaning is established. |
| **Value type** | How should values be parsed and summarized? | Identifier, categorical, ordered categorical, integer, decimal, boolean, date, text. |
| **Meaning** | What does the attribute measure or describe? | Name, description, units, timing, accepted categories, missing-value policy, source provenance. |
| **Experiment use** | How is this field used in this experiment? | Target, explicitly selected tabular input, eligibility filter, stratification attribute, or unused. More than one nonconflicting use can apply. |
| **Presentation** | How should users inspect it? | Display label, suggested chart, category order, visibility and chart priority. |

At import, confirm entity level and value type, and allow a **target candidate** annotation. A candidate is not the active target. A versioned TargetDefinition later chooses the source attribute, prediction unit, class mapping/order, positive class where relevant, and unknown/conflict policy.

Likewise, importing an attribute never automatically feeds it to a model. Tabular predictors require an explicit experiment allowlist and a backend that supports them. Identity/path/split fields and the active target are excluded from predictors, together with its source fields, aliases and declared derived equivalents. This applies in both directions: selecting binary WHO2022 as the target also excludes its original low/high grade column. For the first image-only bladder MIL baseline, clinical columns are available for exploration/filtering, with no tabular predictors enabled by default. WHO1973 should not silently enter a WHO2022 prediction model simply because it is numeric.

Use **Patient attributes**, **Slide/case attributes**, and **Image feature sets** in the UI. Reserve “image feature set” for extracted embeddings with encoder, preprocessing, dimensions and artifact provenance. Do not place thousands of embedding dimensions into the clinical spreadsheet or generate a pie chart for each dimension.

### A data dictionary makes the schema reviewable

The import UI should generate and persist a dictionary; users should not have to author a schema file before importing. Advanced users can supply it as an `Attributes` sheet or companion CSV. Review imported definitions through the same UI.

| Example attribute | Entity | Type | Meaning | Default presentation |
| --- | --- | --- | --- | --- |
| `WHO_2022` | Slide, or case when confirmed | Categorical | Recorded grade; low/high; target candidate | Count bar, optional donut |
| `WHO_1973` | Slide, or case when confirmed | Ordered categorical | Recorded grade; order 1, 2, 3 | Ordered bars |
| `Age_At_Diagnosis` | Patient | Decimal | Years at a specified diagnosis | Histogram |
| `Age_At_Collection` | Case/specimen | Decimal | Years at specimen collection | Histogram at the recorded entity level |
| `Sex` | Patient | Categorical | Source-defined values and meaning | Bars; small-category donut optional |
| `Tissue_Area_mm2` | Slide | Decimal | Measured tissue area in mm², with measurement provenance | Histogram |
| `Comment` | Source slide/case row | Text | Original annotation | Searchable table, missingness count |

These are illustrative definitions. The actual bladder `Age` field is not yet known to mean age at diagnosis or collection; preserve it as recorded until clarified. Do not silently rename the existing `Gender` field to `Sex` or assume `Location` means institution.

Suggested dictionary columns are `Table`, `Column`, `Attribute_ID`, `Display_Name`, `Entity_Level`, `Data_Type`, `Description`, `Units`, `Time_Context`, `Categories`, `Category_Order`, `Missing_Values`, `Target_Candidate`, `Default_Chart`, and `Show_By_Default`. Only table/column binding, confirmed level, and type are needed to finish basic schema review; the UI fills suitable display defaults. Meaning required for a scientific use must be resolved before that use is ready.

`Attribute_ID` is assigned by HistoPilot and stays stable when a display label changes. An advanced template may provide a unique stable ID. Changes to types, category normalization or scientific meaning create a new schema/dataset revision; rearranging charts is a view preference. Freeze source-to-attribute mapping and the dictionary with the dataset.

For simple scalar columns, the dictionary is enough. Keep versioned target class mapping and split definitions in their own objects; a display dictionary is not a replacement for scientific contracts.

## 3. Support both combined and separate files

**Recommended default: one XLSX workbook with separate logical sheets.** This gives users one upload while preserving a clean data model. The equivalent CSV bundle is equally supported. A single combined table remains a convenient quick-start path, particularly for existing bladder files.

| Input layout | Best use | Import behavior |
| --- | --- | --- |
| `bladder.xlsx`: `Slides`, optional `Patients`, optional `Attributes` | Recommended editable template | Each sheet declares one entity/table role. A Patients sheet is unnecessary if it has no attributes beyond verified IDs already linked in Slides. |
| `slides.csv` + optional `patients.csv` + optional `attributes.csv` | Pipelines and version control | Same model as workbook sheets. Filenames suggest roles; the import UI confirms them. |
| One combined CSV or XLSX sheet | Small studies and existing exports | One row per slide. Dictionary/UI identifies patient versus slide attributes; importer validates and separates them internally. |
| Optional `Cases` / `Specimens` sheet | Multiple slides per case/specimen, case-owned labels or time-dependent measurements | Join by explicit keys and checked cardinality. Not required for the first two-table importer. |
| Separate feature index / split assignment table | Multiple feature stores or evaluation protocols | Register as typed scientific inputs; do not mix them into the slide master or treat them as ordinary predictor columns. |

Example of equivalent normalized tables, using fictional IDs and values:

`patients.csv` — one row per patient:

```csv
Patient_ID,Age_At_Diagnosis,Sex
P001,64,Female
P002,71,Male
```

`slides.csv` — one row per slide:

```csv
Slide_ID,Patient_ID,WHO_2022,WHO_1973
S001,P001,low,1
S002,P001,low,1
S003,P002,high,3
```

Equivalent combined table:

```csv
Slide_ID,Patient_ID,Age_At_Diagnosis,Sex,WHO_2022,WHO_1973
S001,P001,64,Female,low,1
S002,P001,64,Female,low,1
S003,P002,71,Male,high,3
```

The combined format repeats patient values. After approved missing-value/category normalization, all nonmissing values of a patient-owned attribute must agree across that patient's rows. A mismatch such as diagnosis ages 64 and 71 must produce a finding; never select the first value or majority automatically. If the attribute actually records age at different collections, move it to the appropriate case/specimen scope instead of overwriting valid observations.

Mixed missing and known patient values also need explicit handling. Offer a reviewed “use the unique known patient value” policy and record its source rows. If an additional Patients table disagrees with a combined table, do not silently prioritize one source. Detect many-to-many joins before materialization, and show unmatched table rows and unlinked slides separately.

If verified patient links are supplied in Slides without a Patients table, HistoPilot may derive a distinct patient-key table from those links. This creates storage rows from known identities; it does not infer new patient identities. Patients with no linked selected slides stay visible in source reconciliation and do not silently enter a slide cohort.

Keep uploads simple: one header row, rectangular sheets, explicit sheet selection and previews. Preserve the original file. Flag duplicate headers, merged header layouts, formula cells without dependable values, numeric IDs with possible information loss, and mixed-type columns. Do not evaluate arbitrary spreadsheet formulas. Do not infer that `0` means missing or that literal `NA` is missing in every column.

### Features and labels should not force a giant spreadsheet

A feature index can use `Feature_Set_ID`, `Slide_ID`, `Feature_Path`, and optionally `Coordinates_Path`, while its manifest carries representation/encoder/preprocessing details. The same slide can appear in multiple feature sets: uniqueness belongs to `(Feature_Set_ID, Slide_ID)`. Any multiple bags per slide require an explicitly different representation contract, not duplicated slide rows.

For P0, ordinary scalar labels remain normal named attributes in their source entity table. A separate labels table is optional and must name the join key and label unit. Multiple raters, time points or alternative annotations require an observation/annotation identity and an explicit resolution rule; never widen a join by accident or choose the last label read. Advanced repeated-annotation support can follow the basic importer.

Existing `k_fold` can be retained as a **split hint**. The split importer separately validates memberships and the declared interpretation of `-1` and `0..4`. No global `Label` or `Fold` header is required to browse a dataset, and selecting a target does not mutate the spreadsheet.

## 4. Apply the design to today's bladder files

| Current header/source | Proposed mapping | Review needed |
| --- | --- | --- |
| `De ID` | `Slide_ID` | Exact values already match the 138 TIFF stems. The user confirmed slide/case semantics. |
| No verified patient field | Canonical `Patient_ID` remains unresolved | Add a real patient crosswalk. Never copy `De ID` or `Case` to satisfy the field. |
| `Case` in raw workbook | Preserve as `Source_Case_ID` initially | Promote to a case relationship only after meaning and cardinality are established. |
| `WHO 2022` | `WHO_2022`, categorical attribute | Preserve low/high values and recorded slide/case scope; mark as a target candidate. |
| `Binary WHO 2022` | `WHO_2022_Binary`, categorical, not continuous numeric | Record `0=low`, `1=high`; validate agreement with original grade. Select one explicit target definition. |
| `WHO 1973` | `WHO_1973`, ordered categorical | Order 1, 2, 3; available for visualization, an explicit target, or the reviewed grade-2 holdout rule. |
| `k_fold` | Legacy split hint | Retain source values; run protocol and patient-leakage validation separately. |
| `Age`, `Gender`, `Race` | Preserve as source attributes | Patient ownership and age timing must be confirmed before patient-level summaries or inputs. |
| `Location` | Opaque source attribute | The raw workbook has only two populated rows; institution/site meaning is unverified. |
| `Comment` | Text attribute | Searchable table and missingness; no automatic target/input assignment. |

These mappings derive from the [local audit](evidence/bladder-audit-2026-09-09.json). Standardization must preserve the raw 144-row versus prepared 138-row reconciliation and six unmatched records. It must not silently convert every original row into an available slide.

If patient mapping arrives later as `Slide_ID,Patient_ID`, join and review it, then publish a child dataset version. Frozen prior targets/splits retain their original dataset links. Do not fill patient IDs into an old frozen dataset in place.

## 5. Attribute-driven exploration inspired by cBioPortal

cBioPortal separates patient and sample clinical tables, links them with required identifiers, and accepts custom attributes described by names, types and other metadata. This is the useful architectural precedent; a cBioPortal sample is not automatically equivalent to a histology slide. HistoPilot needs its own slide/case/specimen hierarchy. Its CSV/XLSX mapping UI should replace the need for users to hand-author cBioPortal's special metadata header rows. [cBioPortal clinical-data format](https://docs.cbioportal.org/file-formats/#clinical-data)

cBioPortal's Study View uses prioritized chart cards, selectable additional charts, and chart-based filtering for cohort exploration. HistoPilot should adopt that interaction pattern with pathology-specific attributes and explicit counting units. [Study View customization](https://docs.cbioportal.org/deployment/customization/studyview/), [cohort selection documentation](https://docs.cbioportal.org/user-guide/faq/#how-can-i-create-a-subset-or-sub-cohort-of-a-study)

### Proposed Dataset page

Provide **Summary**, **Records**, **Attributes**, and **Import / versions** views within Dataset. The import wizard still follows Sources → Mapping → Review → Freeze; exploration works on a complete preview or a selected frozen version and clearly labels which one is shown.

The Summary starts with slides, mapped patients, missing links, target coverage if a target is selected, slide availability and selected-feature coverage. Unknown patient totals stay unknown. Below, display a small initial set of useful chart cards with **Add chart**, a record table, removable filter chips, **Clear filters**, and **Save cohort**. Chart cards show units, counts, missingness and the current dataset version.

Suggested first bladder cards are WHO2022 distribution, ordered WHO1973 distribution, WHO2022 × WHO1973 contingency table/stacked bars, and source availability. Add patient identity coverage immediately; add slides per patient and patient attributes once identities/meaning are resolved. Patch-count and feature-coverage cards appear only after HistoPilot has inspected the selected feature set. The external audit is not app verification.

### Chart defaults

| Column kind | Default view | Notes |
| --- | --- | --- |
| Categorical, few values | Count bars; optional pie/donut for 2–5 categories | Show exact counts and percentages. Keep category colors stable through filtering. |
| Ordered categorical | Ordered bars | WHO1973 values 1/2/3 are categories, not a continuous measurement. |
| Numeric | Histogram with range selection | State units and missing count; boxplot for an explicit comparison. |
| Boolean | Two-category count bar | Missing/unknown remains a separate state. |
| Date | Time histogram when date meaning is defined | Do not infer collection dates from arbitrary strings. |
| Identifier, path, free text, high-cardinality field | Searchable table and completeness/uniqueness summary | No pie chart for every unique ID or comment. |
| Two categorical fields | Contingency table or stacked bars | Same counting unit or an explicitly defined aggregation required. |
| Data quality | Completeness/availability bars and a findings table | Keep missing, invalid, unavailable and excluded distinct. |

Bars should be the general categorical default; pies/donuts are optional for a small number of mutually exclusive categories. Do not auto-display a chart for every column. Prioritize confirmed labels, identity/coverage and useful attributes, while allowing users to choose other cards. Multivalued attributes need a later explicit representation; their percentages must not be forced into a single pie.

### Counts and filter semantics must be scientific, not just visual

Every chart declares its unit: **slides**, **cases/specimens**, or **unique patients**. Patient demographics count each represented patient once, regardless of slide count. A WHO2022 chart attached to slide rows counts slides. A patient-level grade distribution requires a patient-target resolution rule.

For example, P1 having one low and one high slide and P2 having one high slide gives **2/3 high slides**. It does not establish a high-grade patient percentage. Displaying “2/2 high patients” would silently impose an any-positive aggregation policy.

Use these default filter semantics:

1. OR selected values within one categorical filter; AND across different filters. Missingness has an explicit selectable category rather than disappearing from the denominator.
2. Evaluate slide predicates on the same slide record. A high primary slide and a low metastatic slide from one patient must not satisfy `high AND metastatic` through two separate slides.
3. Patient-attribute filters select linked slides from qualifying patients. A missing patient-attribute value concerns a linked patient with a null value; an unlinked slide belongs to a separate missing-identity category. Slide-attribute filters keep the matching slides, then patient charts count distinct mapped patients represented by that selection.
4. Adding all other slides of represented patients is a separate explicit action with a count preview. A filtered slide cohort and a whole-patient expansion are different memberships.
5. Show selected and total counts at each unit, plus valid/missing values. With unresolved links, report known mapped patients and unlinked slide count; the total patient count remains unknown.
6. Chart filtering creates an exploration selection. A mutable preview can save that selection against its import revision; it cannot publish a frozen analysis cohort. **Save cohort** requires a frozen dataset and any target definition required by the selected cohort type, then persists typed predicates, exact membership, dataset/target references and exclusion policy. Browsing never edits frozen splits or retrains a model.

P0 summaries are descriptive. Automated significance testing, survival curves, extensive pairwise comparisons and patient-label aggregation should follow explicit statistical/task definitions. A small number of reliable summaries is more useful than many charts with ambiguous denominators.

## 6. Import UI and release scope

The schema editor should show **source column → canonical name → entity level → type → definition/units → target candidate → chart**. Show sample values, uniqueness, missingness and parse failures next to each choice. Suggest assignments from the header/profile/preset, but ask the user to resolve ambiguous identity or clinical meaning. Allow bulk confirmation of unambiguous attributes; do not force users to classify every optional field before they can save an import draft.

Freeze a basic dataset only after slide identity, table structure and included-row mapping are unambiguous. Unresolved optional attributes can remain preserved as source-row attributes, visibly unavailable for uses requiring confirmed semantics. Missing patient links prevent patient-grouped execution, not source browsing. Unknown target values can remain in the raw dataset; the chosen target/cohort must explicitly resolve or exclude them before execution readiness.

The first release should support CSV/XLSX, a single combined sheet or Slides plus Patients tables, an optional dictionary generated by the UI, explicit identity mapping, typed scalar attributes, basic summaries, checked joins, frozen import history, and saved exploration/cohort state. Compute summary counts and the paginated record table on the server using the same dataset version and filter contract; browser sampling must not determine cohort statistics. Case/specimen relationships may be preserved as reviewed references initially; full repeated-event/rater tables, arbitrary joins and metadata-assisted models can follow.

Before shipping, test equivalent combined/split inputs for identical normalized records; conflicting repeated patient attributes; string IDs/leading zeros; missing patient links; duplicate slide identities; unknown numeric-looking labels; same-slide filter semantics; patient-versus-slide denominators; and patient crosswalk additions creating a child version. Use fictional multi-slide patients, not real clinical identifiers, in these fixtures.

**Next implementation:** agree on this canonical schema and dictionary, then build the file/sheet selection and mapping preview against the existing bladder CSV/XLSX. The dictionary drives the basic charts after parsing and identity review; chart design should not determine the scientific data model.
