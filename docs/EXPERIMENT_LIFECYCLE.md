# Experiments: planning, tracking and results

The Experiments workspace owns development training and the predictors it produces. The roadmap connects **Experiments → Evaluate models**, with **Test cohorts** as the other evaluation prerequisite. There is no separate Build predictors stage.

## User flow

1. **Create experiment:** enter a name, optional tags and notes, then choose **Create & open inputs**. Select an existing experiment under **Start from template** to copy its saved inputs, batch recipes and predictor choices into a new editable plan. Runs, checkpoints and results belong to the original experiment.
2. **Planning:** choose and save inputs. Add, edit, duplicate or remove batch plans. Batch settings separate configuration mode, training defaults, optimization, architecture, precision and resources. Built-in templates provide an ABMIL baseline, a five-epoch quick check and a learning-rate comparison. Choose **Skip**, **Refit**, **Ensemble** or **Both** for the experiment; refit choices require an epoch percentile. Runs and results are locked.
3. **Review & submit:** save or discard pending edits, review the complete experiment, then choose **Freeze & submit experiment**. The backend checks every saved batch and the training runtime before accepting submission. Submission permanently locks inputs, batch settings and predictor choices and starts the frozen batches.
4. **Running:** inspect overall batch progress, then select a batch and run. The run table shows status, epoch progress and latest losses. Selected runs show saved epoch curves, available resource samples and optional detailed evidence. Predictor jobs show waiting/running/ready states, refit epoch budgets and current losses. Cancellation and resuming unfinished work preserve the same frozen configuration.
5. **Finished:** when submitted batches and planned predictor work are completed or cancelled, inputs, batches and runs are read-only. Results show complete out-of-fold groups with AUROC, accuracy and their seed pair, plus the ready predictor library and an **Evaluate predictors** handoff. Incomplete groups never receive a fabricated final score. With Skip, completion depends only on cross-validation. Create a new experiment from a template to change settings or repeat a finished experiment.

Failed, interrupted, unlaunched or unknown work remains in Running with its actual status so recovery stays available. “Finished” does not imply every run succeeded: cancelled batches and incomplete result groups remain explicit. Experiment names, notes and tags remain manageable independently of the scientific configuration.

## Predictor choices and counts

| Choice | Output per configuration / training-seed / split-seed group |
| --- | --- |
| Skip | No predictor; cross-validation results only. |
| Refit | One fresh model trained on all development data. |
| Ensemble | One predictor averaging probabilities from the group's fold checkpoints. |
| Both | One refit and one ensemble. |

For **3 training seeds × 15 configurations × 5 folds**, using one split seed, training produces **225 fold runs**. Both produces **90 predictors: 45 ensembles + 45 refits**. Fold count does not multiply predictor count. Additional split seeds create additional independent groups. Identical custom configuration rows are deduplicated, as in backend expansion.

Refit uses P50, P75, P90, P100 or a custom percentile of the folds' best-checkpoint epochs, with linear interpolation and rounding up. It trains on the complete development cohort for that fixed budget, without early stopping or test-cohort access. The resulting predictor records the source epochs, percentile and actual epoch budget.

The coordinator waits for the entire source batch to complete before publishing its predictors, because publication makes the source checkpoint evidence immutable. Refits run one at a time per experiment and share the existing CPU/RAM/GPU admission mechanism with CV and other compute jobs. A completed ensemble can be evaluated while other refits are still running. Evaluate models filters ready outputs by experiment and method; its review pins the exact selected predictor IDs, so new arrivals never enter an already reviewed evaluation.

## Persistence and recovery

- Editable recipes are stored as `batchPlans` on the experiment, with revision checks. Changing experiment inputs updates the inherited inputs of editable recipes. A copied experiment has independent plan ownership.
- Submission stores a durable receipt before launching workers. It pins publication intent, exact batch IDs, the submission operation ID and per-batch launch identities. Repeating the same submission recovers its existing work without creating another experiment or another set of runs.
- A submission also pins its worker code and Python dependency environment. Later batches cannot silently use changed code or dependencies after a partial failure. Restore the original environment to retry, or copy the experiment into a new plan. Shared worker leases coordinate GPU, CPU and RAM admission across batches.
- Predictor policy is pinned in the submission receipt. Submit starts a persistent experiment coordinator from the archived code; status polling never starts work. Each predictor publication, refit launch and refit publication has a stable operation identity. The coordinator survives browser and API-server disconnections. Interrupted predictor creation has its own resume/cancel controls; a cancelled or finished experiment never reopens its settings.
- Old experiments without a submitted predictor policy do not acquire automatic jobs. Their existing predictors remain visible, and historical refit plans retain their recovery route. Copying an old experiment creates a new editable plan. Choosing Skip cannot be bypassed through older manual build APIs after submission.
- Preflight rejection leaves the experiment editable. A publication or launch failure after acceptance keeps the plan locked and exposes a retry using the same submission identity. Changing a locked plan requires a copy.
- Locks apply to typed experiment updates, older draft routes, adding frozen batches, and individual batch lifecycle changes. Archiving or restoring an experiment does not reopen its configuration. Existing executions also establish a lock for experiments created before this lifecycle was introduced.
- Frozen historical scientific evidence is preserved. Earlier unlaunched frozen batches remain immutable and are included when their owning experiment is submitted. Earlier draft recipes can be loaded into editable plans; they are not silently launched.
- The selected-run history endpoint reads bounded, validated optional metadata. Missing or malformed history produces an explanation without hiding execution state or saved results. Curves use one-based epochs and show at most the latest 2,000 epochs; complete history remains on disk.

## Tracking conventions

The workspace draws on W&B's [run table and chart workspace](https://docs.wandb.ai/models/track/workspaces), [individual run metrics and system measurements](https://docs.wandb.ai/models/runs/view-logged-runs), and [explicit run states](https://docs.wandb.ai/models/runs/run-states). It uses HistoPilot's local records and workers; no W&B service integration is required.

Validation curves describe checkpoint selection. Development assessment/OOF results describe held-out development folds and remain separate. Comparing those scores for model selection does not turn them into independent test estimates. GPU utilization describes the measured device, including other programs, and recorded samples are labeled with their timestamps.

## Checking this branch

The frontend must be rebuilt and bundled after source changes. The user starts or restarts the HistoPilot server; development verification does not start it or submit real training. Use a disposable project when trying the submission flow.

See the [current predictor integration review and screenshots](dev-review/experiment-predictors/README.md) and the [earlier lifecycle verification](dev-review/experiments/README.md) for results and reproduction commands. Coverage includes experiment lifecycle/API regressions, predictor orchestration/recovery, batch planning and tracking tests, and offline browser fixtures with invented records and mocked submission/evaluation.
