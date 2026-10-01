# Command line

The `histopilot` command drives a running HistoPilot service from a terminal. It reads projects, experiments and results; prepares datasets, Targets & splits, features, experiments and Apply models runs from spec files; follows and operates the Task Center; and runs cleanup and study backups. It calls the same local API as the browser, so both always show the same records.

People get readable tables. Scripts and AI agents add `--json` and get one JSON document per command; the [CLI contract](cli-contract.md) defines what they can rely on. [AI agents](agents.md) covers giving an agent access.

## Before you start

1. Start the service with `histopilot serve` or `bash serve.sh`. The CLI calls `http://127.0.0.1:8787` unless `--url` or `HISTOPILOT_URL` names another port.
2. Find your project and make it the default for that service:

   ```text
   $ histopilot project list
   ID                                        NAME       AVAILABLE  UPDATED
   project-ca36a0fd28ca406ab8a2d2092aedc5dc  Study      yes        2026-09-30T06:58:46Z
   $ histopilot use project-ca36a0fd28ca406ab8a2d2092aedc5dc
   Default project for http://127.0.0.1:8787: Study
   ```

   `--project` or `HISTOPILOT_PROJECT` overrides the default for one command. `project create --name NAME --folder DIR` makes a new project in a new or empty folder; `project open DIR` loads an existing one.
3. `histopilot status` summarizes the service, its Task Center runner and the default project. `histopilot project roadmap` shows each stage as the roadmap page does, and the one to do next.

**On another machine.** The service listens on loopback only. Forward its port over SSH, for example `ssh -L 8787:127.0.0.1:8787 workstation`. If the local end of the forward uses a different port, also set `HISTOPILOT_HOST_HEADER=127.0.0.1:8787` (the service's own address), because the service refuses any other Host. A service started with `--login` also needs `HISTOPILOT_LOGIN`; see [AI agents](agents.md#sign-in).

**Shell completion.** `histopilot --install-completion` adds tab completion of commands and options to your shell; `--show-completion` prints the script instead.

## Reading records

Commands are grouped by what they act on: `histopilot <noun> <verb>`. `histopilot <noun> --help` lists the verbs.

| Noun | Reads | Records |
| --- | --- | --- |
| `project` | `list`, `show`, `roadmap`, `exposure` | Projects known to this service |
| `experiment` | `list`, `show`, `results`, `preview`, `batches list` | Experiments; `results` shows each configuration's seed-averaged AUROC with its interval, the seed ensemble, recall by class and the paired comparisons |
| `dataset` | `list`, `show`, `records`, `inspect` | Frozen datasets and their per-slide records; `inspect` reads a table before importing it |
| `targets` | `list`, `show` | Targets & splits versions |
| `features` | `list`, `show`, `validation` | Attached feature sources |
| `bundle`, `pack`, `extraction` | `list`, `show`; `extraction catalog` | Feature bundles, validation and packing jobs, TRIDENT extractions |
| `batch` | `list`, `show`, `results` | Development batches, the training runs of an experiment, with their experiment and state; `results` shows each seed group's out-of-fold metrics |
| `predictor`, `refit` | `list`, `show`; `predictor choices` | Frozen predictors and refit plans |
| `cohort`, `run`, `apply` | `list`, `show`; `run download` | Cohorts, runs of a predictor on a cohort, and batches of runs |
| `reference`, `analysis`, `interpretation` | `list`, `show`; `download` for analyses and interpretations | Reference standards, clinical utility analyses, attention studies |
| `draft`, `configuration` | `list`, `show` | Unfrozen drafts, and frozen configurations of any kind |

- Name a record by its ID. A version that carries a tag, such as a dataset or a Targets & splits version, can also be named `@tag`: `histopilot dataset show "@baseline v1"`. Tags are compared without regard to case.
- Name an experiment by its ID or by its name, compared without regard to case: `histopilot experiment results "Baseline study"`. A name that two experiments share is refused with their IDs (`EXPERIMENT_AMBIGUOUS`).
- Predictor and run names leave out the batch, so two models' fold ensembles of one configuration and seed share a name. `predictor list` and `run list` show each one's batch, and every predictor and run in `--json` carries `model`: its experiment, batch, architecture, method and a description such as "nnMIL · Configuration 1 · Fold ensemble · Train 42 / split 42", as the browser's Runs table describes it. `--experiment EXP` keeps one experiment's batches, predictors or runs, and `run list --cohort COHORT` the runs on one cohort.
- Lists show 50 items; `--limit` and `--offset` page through the rest. Kinds that keep archived and trashed records (predictors, refits, runs, Apply models batches, reference standards, analyses and attention studies) add them with `--include-inactive`.
- `show` prints the top-level fields; add `--json` to see every field.
- `download` saves a file and checks it against the SHA-256 the service recorded for it: `histopilot run download RUN slide-predictions.csv -o predictions.csv`. The service computes the scored tables of a run whose job never saw labels, so those files have no recorded hash to check against.

## Preparing a study from spec files

Each stage has a spec file: YAML or JSON with `kind`, `specVersion: 1` and the fields of the service's request. `histopilot NOUN template -o FILE` writes a complete one, with every field that changes the science written out; edit it, then create the record from it. `histopilot schema KIND` prints every field a kind may hold. A spec names other records by ID or `@tag`; each tag it reads comes back as a note naming the ID it stands for.

Creating a record previews it first and shows the findings, then asks, like every change (see [Changing things](#changing-things)).

| Stage | Commands |
| --- | --- |
| Datasets | `dataset inspect TABLE` shows its sheets and columns; `dataset template -o import.yaml`; `dataset import --from import.yaml --tag "baseline v1"` |
| Targets & splits | `targets template -o targets.yaml`; `targets infer targets.yaml --field FIELD -o targets.yaml --force` fills the target's classes from the field's training values, as the page does; set `positiveClass` yourself; `targets create --from targets.yaml --tag TAG` |
| Slide features | `extraction catalog`; `extraction template`, `extraction run --from FILE --wait`; then `features template --from-extraction JOB -o features.yaml` for its outputs, or `features template` for a folder you already have; `features attach --from FILE --tag TAG` |
| Bundles and packs | `bundle template`, `bundle create --from FILE --tag TAG`; `pack template`, `pack create --from FILE --wait` |
| Experiments | `experiment template --preset nnmil -o design.yaml`; `experiment create --from design.yaml`; `experiment update EXP --from design.yaml` to change the draft; `experiment preview EXP`; `experiment freeze EXP`; `experiment start EXP --wait` |
| Predictors | `predictor choices`; `predictor seed-ensemble --experiment EXP --batch B --candidate C --name NAME`; `refit launch`, `refit publish` |
| Cohorts | `cohort template --variant labeled` or `unlabeled`; `cohort create --from FILE --tag TAG` |
| Apply models | `apply template --experiment EXP -o apply.yaml`; `apply preview --from apply.yaml`; `apply run --from apply.yaml --wait` |
| Reference standards and clinical utility | `reference template`, `reference create --from FILE`; `analysis template`, `analysis create --from FILE` |
| Interpretation | `interpretation template`, `interpretation run --from FILE --wait` |

**Experiments.** A design file holds the experiment's name, notes, tags and predictor policy; its inputs (dataset, Targets & splits, feature bundle, loading policy and the version 4 training design); and its batches, each with an `id` and a complete recipe. `experiment export-design EXP` writes one back, and `experiment create` accepts it unchanged. Before the setup is frozen, `experiment update EXP --from FILE` saves an edited design over the same draft, `experiment batches add EXP --from FILE` adds another design file's batches (only its batches are read), and `experiment batches remove EXP PLAN` removes one.

**Controlled comparisons.** `experiment template --compare-model nnmil --compare-model mean_pool` writes a batch whose configurations are the preset's recipe, as the reference, and one arm per model. Each arm changes only the settings its model owns, so the comparison measures the model alone. `--clinical-field age:numeric` adds clinical inputs, and `--compare-input multimodal` or `clinical` adds arms that use them.

**Proposed values.** Where the browser proposes values from the project's records, the CLI proposes the same ones, and notes where each came from:

- `apply template --experiment EXP` chooses the method (seed ensembles when any is ready, otherwise each seed group's fold ensemble and refit), every ready predictor, the testing cohort the predictors' development reserved, and that cohort's inference settings. `--method`, `--cohort` and `--predictor` override them.
- `experiment apply-config EXP --batch B --candidate C` does what Results' "Apply this configuration" does. It writes an Apply models spec for the configuration's seed ensemble, or for its fold ensemble when the configuration has one seed group. When the seed ensemble is not built yet, it first builds it from the verified fold checkpoints; this asks for confirmation, and nothing trains.
- `run label-sources RUN` lists the labels a run can be scored against. Without `--reference`, a labeled run is scored against its cohort's labels, and an unlabeled run against the first reference standard that fits it; a note says which.

**Comparing runs.** `run summary RUN` says what a run predicted, labeled or not: class counts, confidence, how its members agreed, and its model. `run summary RUN --comparison OTHER` adds how often another run on the same cohort agrees with it, case by case: agreement, kappa, a weighted kappa for three or more classes (it reads the class order as an ordered scale) and the cross-table. That is how two models applied to one unlabeled cohort compare; `run cases RUN --comparison OTHER --outcome disagreement` lists the cases where they disagree, with the other run's call beside each (`run export-cases` takes `--comparison` too). `run compare LEFT RIGHT` is a different test: a paired patient-level comparison of two scored runs, unavailable for slide-level experiments and unlabeled cohorts.

**Exports.** `experiment export-results EXP -o folds.csv` writes the same file as the Results page's "Folds and seeds" download: every seed's out-of-fold result and every test fold. `run export-predictions` and `run export-cases` write a run's predictions and its case review.

**Tags.** `histopilot NOUN label ID --tag TAG --note NOTE` renames a dataset, Targets & splits, feature, cohort or other configuration version. Its contents and hashes never change.

## The Task Center

| Command | What it does |
| --- | --- |
| `tasks list` | Tasks of the default project; `--all-projects` for the whole machine, `--state live` for running and waiting work |
| `tasks show TASK` | Owner, state, waiting reason, failure explanation and the log's tail |
| `tasks log TASK` | The whole log; `--from BYTE` reads from an offset, `--follow` keeps printing until the task settles |
| `tasks wait TASK` | Waits until the task settles: exit 0 when it succeeds, 9 when it fails, is cancelled or needs attention |
| `tasks owners`, `tasks owner KEY` | Owners (experiments, batches, records) in queue order, with their allowed actions |
| `tasks hold`, `release`, `stop`, `move --to top` | Queue control for an owner |
| `tasks cancel`, `tasks retry` | One task, or with `--owner KEY` all of an owner's work |
| `tasks capacity` | Shows capacity; with options such as `--cpu-task-slots 4` or `--pause` it changes it for every project |

Records that run have their own actions too: `launch`, `resume` and `cancel` for batches, runs, refits and interpretations; `resume` and `cancel` for extractions; `cancel` for packs and Apply models batches; `refit publish`; `experiment resume-predictors` and `cancel-predictors`. `--wait` on a command that starts work waits for it the way `tasks wait` does.

A stopped runner leaves work queued. `tasks wait` then stops with exit 9 and says so, and `histopilot runner start` starts the runner again.

## Changing things

Commands that change records show what will happen first, and then ask:

```text
$ histopilot cleanup apply draft-448532976d38408f97c0d8fc5359a9b6 --action trash
Trash 1 record: draft:draft-448532976d38408f97c0d8fc5359a9b6.
Proceed? [y/N]:
```

- The default answer is no. `--yes` confirms without a prompt; it has no environment variable, so every command must say it.
- `--dry-run` shows the preview and changes nothing.
- A preview that has blocking findings stops the command, with or without `--yes`.
- `--preview-hash HASH` changes something only if a fresh preview still has that hash. A person can then confirm exactly the preview a colleague or an agent showed them.
- The CLI records each operation's ID before sending it, and keeps it until the command ends. Rerunning a command after a lost connection, a timeout or Ctrl-C, even one that came while `--wait` waited for the work, replays the same operation, and the service applies it once; a replayed `project create` or draft returns the record the first attempt made.
- Commands that freeze a spec file (`targets create`, `cohort create` and the like) save it as a draft first, to preview it, so `--dry-run` or a "no" leaves that draft. Rerunning the same command reuses it.

| Noun | Changes |
| --- | --- |
| `cleanup` | `list` shows every record's cleanup key; `preview` and `apply` archive, trash or restore records |
| `backup` | `export`, `verify` and `restore` study archives as Task Center tasks, with `--wait`; `list`, `show`, `cancel`, `retry` |
| `sources` | `list` registered folders; `add` a folder, or `relink` one that moved |

Reading and previews never ask. Saving a draft, which includes an experiment and its batch plans, is a preview too: nothing is frozen, queued or removed. Creating projects, registering or relinking source folders, study archives, Task Center capacity, AI exposure levels and agent tokens reach beyond one project's records. They are confirmed the same way, and an agent's token never reaches them.

## AI agents

| Command | What it does |
| --- | --- |
| `project exposure [--set LEVEL]` | Shows or changes what AI agents may see of a project: `none`, `metadata` or `full` |
| `token create --scope read --scope preview [--days 7]` | Creates a scoped token for one project; it is printed once |
| `token list`, `token revoke ID` | Lists tokens without their secrets; revokes one |
| `confirm list`, `confirm show ID` | Changes agents asked for, waiting for a person |
| `confirm approve ID`, `confirm decline ID` | Runs a request as you, or declines it |
| `agent serve --token-file FILE` | Serves the agent tools over MCP to a chat app, with a scoped token |
| `login url`, `login rotate` | The sign-in link of a service started with `--login` |

[AI agents](agents.md) explains exposure levels, tokens, approvals and how to keep an agent away from private data.

## Scripts and agents

With `--json`, standard output carries exactly one JSON document, and the command never prompts:

```text
$ histopilot experiment list --json
{"schemaVersion": 1, "ok": true, "data": [{"id": "draft-56e4…", "name": "Baseline ABMIL", "runState": null, …}], "warnings": [], "page": {"offset": 0, "limit": 50, "hasMore": false}}
```

- Anything that runs carries `runState`: `queued`, `running`, `succeeded`, `failed`, `needs-attention` or `cancelled`.
- A failure returns `{"ok": false, "error": {"code", "kind", "message", "status", "findings"}}`, and the exit code tells the caller what to do next. See the [exit codes](cli-contract.md#exit-codes) and every service code in [error codes](error-codes.md).
- A change without `--yes` exits 7 and returns the preview in `data.preview`, ready to show a person. So does an agent's change that waits for a person's approval, with the request in `data.request`.
- `histopilot api GET /projects` sends a raw request to any route under `/api/v1`; changes are confirmed as above.
- `histopilot schema KIND` prints the JSON Schema of a spec kind, such as `targets` or `cohort`.
- `HISTOPILOT_TOKEN` makes every command use a scoped agent token instead of the service's session.

## Older commands

`serve`, `doctor`, `runner`, `verify-study`, `restore-study` and `verify-feature-pack` work as before. The flat commands `train-batch`, `training-status`, `pack-features` and `feature-jobs` still print their raw JSON, and a note on standard error names the noun-verb command that replaces each.
