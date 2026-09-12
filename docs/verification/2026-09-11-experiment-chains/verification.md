# Experiment and predictor chain interface verification

Verified the actual React experiment registry/detail, batch controls, post-development freeze, and model-evaluation components in an offline file:// browser harness. All service methods were replaced with in-memory synthetic responses. No HistoPilot server was started, no training or inference was launched, and no real workspace records were changed. The browser session was closed after verification.

Observed zero fetch calls and zero browser runtime errors throughout. Desktop comparison and predictor screenshots, plus 600 × 900 experiment and evaluation screenshots, were visually inspected. Both narrow pages had document width equal to viewport width; large tables scroll inside their containers.

Passed browser interactions:

- Create an experiment before choosing inputs. Simulate a committed create with a lost response; retry sends the identical request and operation ID and commits one record. Inputs open after creation.
- Select two experiments, fetch their full details on demand, compare exact dataset/protocol/feature and batch-configuration snapshots, and switch the comparison baseline. Human-readable setting labels retain the raw paths underneath.
- Delete an experiment: show its dependent batch without adding it implicitly; explicitly include dependencies; require acknowledgement; apply precisely the reviewed records. Restore the experiment to Active.
- Open one experiment's Runs view: no other experiment's batch appears and all execution requests are scoped to that experiment.
- Filter predictor choices to the source experiment. Preview verified checkpoint lineage, require acknowledgement, simulate a committed freeze with a lost response, then retry the identical selection, preview hash and operation ID. One predictor is created for that experiment.
- Select the predictor through its evaluation link. Exclude incompatible test cohorts and disable stale cohorts. Preview and acknowledge an evaluation plan; preserve the selected predictor lineage and show Planned / Not run with no inference-launch controls.
- Switch to the second predictor: clear the old cohort and show only the second chain's evaluations. Show all predictors to see both independent evaluation records.
- Filter the experiment registry to Trash to inspect deleted legacy history, then return to Active and search for the queued study.

Automated verification: all 260 frontend tests passed before the final capability/tag regressions; the final focused suite passed 25 tests and TypeScript passed. Those regressions verify current service execution capability enables Launch even when the immutable historical manifest flag is false, and identical batch names in different experiments receive different version tags. Root performs the final full suite/build after integration.

Artifacts: experiment-comparison.png, predictor-frozen.png, evaluation-mobile.png, experiments-mobile.png, harness.tsx, build.mjs, index.html. The comparison screenshot was refreshed after the final readable-label change. The other screenshots show the verified interactions before the final count-label/service-capability/tag refinements.

These browser checks exercise UI behavior against mocked contracts. Backend API, dependency, lifecycle, publication and scientific validity checks are covered separately by Python tests. No live-server end-to-end request was attempted under the user's server-start preference.

Final integration verification: **262 tests across 39 frontend test files passed**, TypeScript passed, and the production frontend build and Python static bundle completed. Final Python registry/API suite: **35 passed**; training execution regressions: **47 distinct cases passed** across the full run and focused final additions. Ruff and whitespace checks passed.
