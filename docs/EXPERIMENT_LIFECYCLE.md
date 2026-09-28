# Experimental Setup and Experiments

**Experimental Setup** owns the scientific design. **Experiments** starts frozen setups and tracks their execution. Targets & splits supplies fixed training/testing membership; only the training members enter fold design, validation, model selection and refitting. The reserved testing members remain separate for evaluation.

## User flow

1. In **Experimental Setup**, create a named setup, or copy a prior setup into a new editable record.
2. In **Inputs**, select the dataset, frozen target/split version and feature bundle. Configure K-fold training, early-stop validation, split seeds and loading policy. Other fold strategies are disabled in the current setup editor. **Check & continue to hyperparameters** checks compatibility and training coverage, then saves the shared inputs.
3. In **Hyperparameters**, add model batches, parameter grids or explicit recipe rows and training seeds. Choose **Skip**, **Refit**, **Ensemble** or **Both** for each batch; refit choices require an epoch percentile. Save all batch edits. Batches carry no compute settings: the Task Center decides parallelism, device and threads when the work runs. A batch saved earlier with compute settings shows them as legacy, and saving it again removes them.
4. In **Review & freeze**, freeze the setup. The frozen record pins inputs, training design, recipes and predictor choices. Freezing does not start training. Copy it to change scientific settings.
5. Continue to **Experiments**, open the ready setup and choose **Start experiment**. Submission checks current features and runtime availability, then queues the frozen work as [Task Center](TASK_CENTER_DESIGN.md) tasks. Queue order, hold, cancel, logs and resources are managed in the Task Center; the experiment itself offers one **Resume** when its work needs attention.
6. Inspect **Runs** and **Results**. **Runs** starts with one status line for the experiment (training and predictor progress, queue position, failures, time left) that links to the experiment in the Task Center, followed by the run table and curves. **Results** opens on the metrics and lists ready predictors. Completed out-of-fold groups expose development metrics, and ready predictors can be handed to **Evaluate models**. Incomplete groups do not receive a fabricated score. Test evaluation uses the reserved testing cohort or a separate independent cohort.

A frozen setup and a completed experiment are different artifacts. Preparation can finish before a GPU is available. Failed or interrupted executions retain their actual state and recovery actions. Historical experiments and protocol designs remain accessible; they are not rewritten into the new pipeline.

## Predictor choices and counts

Predictor choices belong to each batch. An experiment can mix a Skip batch, an ensemble batch, a P50 refit batch, and a Both/P75 batch. The submission review lists every batch’s choice and the total expected outputs.

| Choice | Output per configuration / training-seed / split-seed group |
| --- | --- |
| Skip | No predictor; cross-validation results only. |
| Refit | One fresh model trained on all development data. |
| Ensemble | One predictor averaging probabilities from the group's fold checkpoints. |
| Both | One refit and one ensemble. |

For **3 training seeds × 15 configurations × 5 folds**, using one split seed, training produces **225 fold runs**. Both produces **90 predictors: 45 ensembles + 45 refits**. Fold count does not multiply predictor count. Additional split seeds create additional independent groups. Identical custom configuration rows are deduplicated, as in backend expansion.

Refit uses P50, P75, P90, P100 or a custom percentile of the folds' best-checkpoint epochs, with linear interpolation and rounding up. It trains on the complete development cohort for that fixed budget, without early stopping or test-cohort access. The resulting predictor records the source epochs, percentile and actual epoch budget.

The coordinator waits for the entire source batch to complete before publishing its predictors, because publication makes the source checkpoint evidence immutable. Each ready refit is launched as its own Task Center task and admitted like other GPU work, so several refits of one experiment can run at once when capacity allows. Experiments submitted before 2026-09-27 keep their pinned coordinator, which launches one refit at a time. A completed ensemble can be evaluated while other refits are still running. Evaluate models starts with explicit selection of one or more experiments, then ensemble/refit methods and a test cohort. Review pins the exact available predictor IDs; new arrivals never enter an already reviewed evaluation. Individual predictor selection remains available under advanced controls. Results compare ensemble and refit on matching source groups within the same cohort and scoring unit; unmatched or undefined scores do not produce a paired comparison.

## Persistence and recovery

- Editable recipes are stored as `batchPlans` on the experiment, with revision checks. Changing experiment inputs updates the inherited inputs of editable recipes. A copied experiment has independent plan ownership.
- Submission stores a durable receipt before launching workers. It pins publication intent, exact batch IDs, the submission operation ID and per-batch launch identities. Repeating the same submission recovers its existing work without creating another experiment or another set of runs.
- A submission also pins its worker code and Python dependency environment. Later batches cannot silently use changed code or dependencies after a partial failure. Restore the original environment to retry, or copy the experiment into a new plan. The Task Center admits every task against GPU slots, VRAM, RAM and CPU across batches, experiments and projects, and writes a lease for each task so workers started in tmux before the Task Center still see its usage.
- Each batch’s predictor policy is pinned in the submission receipt as `predictorPolicies[batchId]`. Coordinator version 2 stores that map and each refit item’s own percentile; historical version 1 coordinators retain their original experiment-wide policy and archive. Submit starts a persistent experiment coordinator from the archived code; status polling never starts work. Each predictor publication, refit launch and refit publication has a stable operation identity. The coordinator survives browser and API-server disconnections. Interrupted predictor creation resumes through the experiment's **Resume** (or the owner's Retry in the Task Center); experiments submitted before the Task Center keep their own resume/cancel controls. A cancelled or finished experiment never reopens its settings.
- Old experiments without a submitted predictor policy do not acquire automatic jobs. Their existing predictors remain visible, and historical refit plans retain their recovery route. Copying an old experiment creates a new editable plan. Historical submissions with an experiment-wide policy retain that meaning; copying them resolves inherited choices into explicit editable batch policies. Choosing Skip for a batch cannot be bypassed through older manual build APIs after submission.
- Setup-review rejection leaves the setup editable. Execution preflight rejection preserves the frozen setup for a later retry. A publication or launch failure after acceptance keeps the plan locked and exposes a retry using the same submission identity. Changing a locked plan requires a copy.
- Locks apply to typed experiment updates, older draft routes, adding frozen batches, and individual batch lifecycle changes. Archiving or restoring an experiment does not reopen its configuration. Existing executions also establish a lock for experiments created before this lifecycle was introduced.
- Frozen historical scientific evidence is preserved. Earlier unlaunched frozen batches remain immutable and are included when their owning experiment is submitted. Earlier draft recipes can be loaded into editable plans; they are not silently launched.
- A coordinator that finds the project busy (another operation holds the project lock) keeps retrying. After 120 s it exits with code 75 and the Task Center requeues it with a backoff (coordinators pinned before 2026-09-27 record the busy error instead and are requeued the same way). Items that could not proceed because the project was busy stay waiting and are never marked failed. Finished items are kept.
- The selected-run history endpoint reads bounded, validated optional metadata. Missing or malformed history produces an explanation without hiding execution state or saved results. Curves use one-based epochs and show at most the latest 2,000 epochs; complete history remains on disk.

## Execution status

A submitted experiment has one execution status, read from the Task Center without taking the project lock. From most to least urgent:

| Status | Meaning |
| --- | --- |
| Running | At least one of its tasks is running. |
| Queued | A task is queued and none is running; the reason says what it waits for. |
| Held | Work remains, but the experiment is held in the Task Center. |
| Waiting | Only blocked tasks remain, the runner will requeue work (a busy project, an out-of-memory retry), or the runner is stopped. |
| Needs attention | A task failed or was interrupted, or the coordinator stopped with an error. **Resume** retries the owner. |
| Cancelled | Its work was cancelled. Task Center experiments stay resumable. |
| Completed | All of its tasks succeeded. |

`statusReason` explains Waiting, Held and Needs attention, for example "Another operation is changing this workspace." Before submission a setup reads created, planned or ready. Experiments started before the Task Center derive the same vocabulary from their saved execution records.

## Tracking conventions

The workspace draws on W&B's [run table and chart workspace](https://docs.wandb.ai/models/track/workspaces), [individual run metrics and system measurements](https://docs.wandb.ai/models/runs/view-logged-runs), and [explicit run states](https://docs.wandb.ai/models/runs/run-states). It uses HistoPilot's local records and workers; no W&B service integration is required.

Validation curves describe checkpoint selection. Development assessment/OOF results describe held-out development folds and remain separate. Comparing those scores for model selection does not turn them into independent test estimates. GPU utilization describes the measured device, including other programs, and recorded samples are labeled with their timestamps.

## Checking this branch

The frontend must be rebuilt and bundled after source changes. The user starts or restarts the HistoPilot server; development verification does not start it or submit real training. Use a disposable project when trying the submission flow.

See the [batch planning and evaluation redesign verification](dev-review/batch-workflow/README.md), the [predictor integration review and screenshots](dev-review/experiment-predictors/README.md) and the [earlier lifecycle verification](dev-review/experiments/README.md) for results and reproduction commands. Coverage includes experiment lifecycle/API regressions, predictor orchestration/recovery, batch planning and tracking tests, and offline browser fixtures with invented records and mocked submission/evaluation.
