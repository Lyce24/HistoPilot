import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { evaluation } from '../api/evaluation';
import { modelEvaluations, predictors, predictorMethodLabel, computeActive, type ModelEvaluation, type EvaluationSelection } from '../api/predictors';
import type { Workspace } from '../api/types';
import { ErrorNotice, PageHeader, Panel } from '../components/ui';
import { Findings } from '../components/ScientificUI';
import PublicationConfirmation from '../components/PublicationConfirmation';
import { useReviewedPublication } from '../components/useReviewedPublication';
import { cleanupLink, useHashParameters } from '../lib/hashRoute';
import { versionLabelText } from '../lib/versionLabels';
import ComputeJobControls from '../components/ComputeJobControls';
import { shortRecordId } from '../lib/recordLabels';
import { downloadJSON } from '../lib/download';
import BulkEvaluationRunner from '../components/BulkEvaluationRunner';
import EvaluationResultsTable from '../components/EvaluationResultsTable';
import EvidenceChain, { evidenceLink } from '../components/EvidenceChain';
import { groupPredictors, predictorConfigurationLabel, experimentPredictorLink } from '../lib/predictorGroups';
import './ModelChains.css';

export default function LocalModelEvaluation({ workspace }: { workspace: Workspace }) {
  const parameters = useHashParameters();
  const linkedExperiment = parameters.get('experiment') ?? '';
  const linkedPredictor = parameters.get('predictor') ?? '';
  const linkedCohort = parameters.get('cohort') ?? '';
  const linkedEvaluation = parameters.get('evaluation') ?? '';
  return <EvaluationWorkspace key={`${workspace.project.id}:${linkedPredictor}:${linkedCohort}:${linkedEvaluation}:${linkedExperiment}`} workspace={workspace} linkedPredictor={linkedPredictor} linkedCohort={linkedCohort} linkedEvaluation={linkedEvaluation} linkedExperiment={linkedExperiment} />;
}

function EvaluationWorkspace({ workspace, linkedPredictor, linkedCohort, linkedEvaluation, linkedExperiment }: { workspace: Workspace; linkedPredictor: string; linkedCohort: string; linkedEvaluation: string; linkedExperiment: string }) {
  const project = workspace.project.id;
  const client = useQueryClient();
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), refetchInterval: 5000 });
  const cohorts = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project) });
  const records = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project), refetchInterval: 15000 });
  const [experimentId, setExperimentId] = useState(linkedExperiment);
  const [predictorId, setPredictorId] = useState<string | null>(null);
  const chosenPredictor = predictorId ?? linkedPredictor;
  const [cohortId, setCohortId] = useState(linkedCohort);
  const [name, setName] = useState('');
  const [mode, setMode] = useState<'batch' | 'single'>(linkedPredictor ? 'single' : 'batch');
  const [selectedRecord, setSelectedRecord] = useState(linkedEvaluation);
  const publication = useReviewedPublication(
    (selection: EvaluationSelection) => modelEvaluations.preview(project, selection),
    (selection, hash, operation) => modelEvaluations.save(project, selection, hash, operation),
    (preview) => preview.canSave && !preview.findings.some((finding) => finding.severity === 'error'),
    async (record) => { setSelectedRecord(record.id); await Promise.all([client.invalidateQueries({ queryKey: ['model-evaluations', project] }), client.invalidateQueries({ queryKey: ['cleanup', project] })]); },
  );
  const predictor = registry.data?.items.find((item) => item.id === chosenPredictor && item.lifecycleState !== 'trashed');
  const cohort = cohorts.data?.items.find((item) => item.id === cohortId);
  const matchingCohorts = (cohorts.data?.items ?? []).filter((item) => !predictor || (item.manifest.spec.protocolId === predictor.manifest.inputs.protocol.id && item.manifest.spec.developmentFeatureBundleId === predictor.manifest.inputs.features.bundle.id));
  const cohortReady = cohort?.current === true && !cohort.findings?.some((finding) => finding.severity === 'error') && matchingCohorts.some((item) => item.id === cohortId);
  const canReview = predictor?.lifecycleState === 'active' && cohortReady && name.trim() && !registry.isError && !cohorts.isError;
  const sources = groupPredictors((registry.data?.items ?? []).filter((item) => item.lifecycleState === 'active' || (item.id === chosenPredictor && item.lifecycleState === 'archived')));
  const choices = sources.filter((item) => !experimentId || item.id === experimentId);
  const visible = (records.data?.items ?? []).filter((item) => (!experimentId || item.manifest.experimentId === experimentId) && (mode !== 'single' || !chosenPredictor || item.manifest.predictorId === chosenPredictor));
  const detail = (records.data?.items ?? []).find((item) => item.id === selectedRecord);
  const predictorName = (id: string) => registry.data?.items.find((item) => item.id === id)?.manifest.name ?? id;
  const cohortName = (id: string) => { const item = cohorts.data?.items.find((item) => item.id === id); return item ? versionLabelText(item, 'Test cohort') : id; };
  return <div className="clinical-workspace model-chains">
    <PageHeader eyebrow="03 EVALUATE" title="Evaluate models" description="Evaluate ready predictors produced by your experiments. Compare configurations, training seeds and predictor methods on the same test cohort." actions={<div className="inline-actions"><button className="btn btn-primary" onClick={() => { setMode('batch'); setPredictorId(''); setExperimentId(''); publication.reset(); }}>Choose predictors</button><a className="btn btn-secondary" href="#test-data">Test cohorts</a></div>} />
    <EvidenceChain current="evaluation" experimentId={predictor?.manifest.experimentId ?? experimentId} predictorId={chosenPredictor} evaluationId={selectedRecord} />
    <ErrorNotice error={publication.error ?? records.error ?? registry.error ?? cohorts.error} />
    <p className="callout">Evaluate an ensemble or refit predictor on a compatible test cohort. Each run saves predictions and metrics independently. Unlabeled slides receive predictions; metrics use labeled records only.</p>
    {publication.saved ? <p className="callout science-success" role="status">Evaluation plan <strong>{publication.saved.manifest.name}</strong> saved with its predictor and test-cohort lineage.</p> : null}
    <nav className="run-tabs" aria-label="Evaluation setup"><button className={mode === 'batch' ? 'selected' : ''} onClick={() => setMode('batch')}>All / selected predictors</button><button className={mode === 'single' ? 'selected' : ''} onClick={() => setMode('single')}>Single predictor</button></nav>
    {mode === 'batch' ? <Panel title="Run predictors on a test cohort" subtitle="Choose an experiment or compare several. Each configuration and seed keeps its own ensemble or refit result."><BulkEvaluationRunner key={linkedExperiment} project={project} predictors={registry.data?.items ?? []} cohorts={cohorts.data?.items ?? []} linkedCohort={linkedCohort} linkedExperiment={experimentId} onExperimentChange={setExperimentId} onOpenEvaluation={(id) => { setSelectedRecord(id); void records.refetch(); }} /></Panel> : null}
    {mode === 'single' ? <Panel title="Select evaluation inputs" subtitle="Each plan uses one frozen predictor and one test cohort. The same predictor can have several evaluations.">
      <fieldset className="chain-fields" disabled={publication.locked} onChange={() => publication.reset()}>
        <legend className="sr-only">Evaluation inputs</legend>
        <label className="label">Source experiment<select className="field" value={experimentId} onChange={(event) => { setExperimentId(event.target.value); setPredictorId(''); setSelectedRecord(''); }}><option value="">All experiments</option>{experimentId && !sources.some((item) => item.id === experimentId) ? <option value={experimentId}>Linked experiment · no ready predictors</option> : null}{sources.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.items.length} predictors</option>)}</select></label>
        <label className="label">Predictor<select className="field" value={chosenPredictor} onChange={(event) => { setPredictorId(event.target.value); setCohortId(''); setSelectedRecord(''); }}><option value="">Choose a ready predictor</option>{chosenPredictor && !predictor ? <option value={chosenPredictor} disabled>Linked predictor unavailable</option> : null}{choices.map((group) => <optgroup key={group.id} label={group.name}>{group.items.map((item) => <option key={item.id} value={item.id}>{predictorConfigurationLabel(item.manifest)} · Train {item.manifest.trainingSeed} / split {item.manifest.splitSeed} · {predictorMethodLabel(item.manifest.method)}{item.lifecycleState === 'archived' ? ' · archived' : ''}</option>)}</optgroup>)}</select></label>
        <label className="label">Test cohort<select className="field" value={cohortId} disabled={!predictor} onChange={(event) => setCohortId(event.target.value)}><option value="">Choose a compatible test cohort</option>{cohortId && !matchingCohorts.some((item) => item.id === cohortId) ? <option value={cohortId} disabled>Selected test cohort unavailable or incompatible — choose another</option> : null}{matchingCohorts.map((item) => <option key={item.id} value={item.id} disabled={!item.current || item.findings?.some((finding) => finding.severity === 'error')}>{versionLabelText(item, 'Test cohort')} · {item.manifest.summary.includedSlides} slides{!item.current ? ' · needs verification' : ''}</option>)}</select></label>
        <label className="label">Evaluation name<input className="field" value={name} maxLength={120} onChange={(event) => setName(event.target.value)} placeholder="For example: External validation" /></label>
      </fieldset>
      {predictor ? <p className="muted">Source: <a href={experimentPredictorLink(predictor.manifest.experimentId, predictor.id)}>{predictor.manifest.experiment?.name ?? shortRecordId(predictor.manifest.experimentId)}</a> · {predictorMethodLabel(predictor.manifest.method)} · {predictor.manifest.checkpoints.length} checkpoints · class target {predictor.manifest.target.field}</p> : <p>Choose a ready predictor from <a href="#experiments">Experiments</a>. Predictors appear automatically when the selected ensemble or refit work finishes. Experiments using Skip have no predictors; use one as a template to choose a different policy.</p>}
      {predictor?.lifecycleState === 'archived' ? <p className="callout">This predictor is archived. <a href={cleanupLink(predictor.id)}>Restore it to Active</a> before creating another evaluation; existing results remain available below.</p> : null}
      {predictor && !matchingCohorts.length ? <p className="callout">No test cohort uses this predictor’s development protocol and feature version. Prepare one in <a href="#test-data">Test cohorts</a>.</p> : null}
      <button type="button" className="btn btn-secondary" disabled={publication.locked || !canReview} onClick={() => { if (canReview) void publication.preview({ predictorId: chosenPredictor, cohortId, name: name.trim() }); }}>Review evaluation</button>
    </Panel> : null}
    {mode === 'single' && publication.review ? <Panel title="Review evaluation inputs">
      <Findings findings={publication.review.preview.findings} />
      <p><strong>{publication.review.selection.name}</strong>: {predictorName(publication.review.selection.predictorId)} → {cohortName(publication.review.selection.cohortId)}</p>
      <p>The review checks frozen weights, target encoding, development provenance, feature compatibility and test-slide coverage.</p>
      {publication.review.preview.canSave ? <PublicationConfirmation busy={publication.busy} uncertain={publication.review.uncertain} acknowledged={publication.acknowledged} onAcknowledge={publication.setAcknowledged} onConfirm={() => void publication.publish()} onReset={publication.reset} label="Save evaluation plan" /> : null}
    </Panel> : null}
    <Panel title="Evaluations and results" subtitle="Compare results within the same test cohort. Each row retains its source experiment, configuration, seeds and predictor method.">
      {mode === 'single' && chosenPredictor ? <button className="text-button" onClick={() => { setPredictorId(''); publication.reset(); }}>Show all predictors</button> : null}
      <EvaluationResultsTable records={visible} predictors={registry.data?.items ?? []} cohorts={cohorts.data?.items ?? []} loading={records.isPending} onOpen={setSelectedRecord} />
    </Panel>
    {detail ? <EvaluationDetail key={detail.id} project={project} record={detail} /> : null}
  </div>;
}

function EvaluationDetail({ project, record }: { project: string; record: ModelEvaluation }) {
  const [error, setError] = useState<Error | null>(null);
  const [downloading, setDownloading] = useState(false);
  const trashed = record.lifecycleState === 'trashed';
  const shouldPoll = !trashed || computeActive(record.execution);
  const execution = useQuery({ queryKey: ['compute-job', project, 'evaluation', record.id], queryFn: () => modelEvaluations.execution(project, record.id), initialData: record.execution, enabled: shouldPoll, refetchInterval: (query) => shouldPoll && computeActive(query.state.data) ? 3000 : false });
  const current = shouldPoll ? execution.data : record.execution;
  const result = !execution.isError && current?.status === 'completed' ? current.result : null;
  const metrics = result?.metrics;
  const [unit, setUnit] = useState<'selected' | 'slide' | 'patient'>('selected');
  const selected = metrics?.[unit];
  async function download(filename: 'slide-predictions.csv' | 'patient-predictions.csv' | 'metrics.json') {
    if (trashed || !result || downloading) return;
    setError(null); setDownloading(true);
    try { await modelEvaluations.download(project, record.id, filename); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Download failed.')); }
    finally { setDownloading(false); }
  }
  return <Panel title={record.manifest.name} subtitle="Evaluation run and results">
    <ComputeJobControls project={project} id={record.id} kind="evaluation" initial={record.execution} readOnly={record.lifecycleState !== 'active'} />
    <ErrorNotice error={error} />
    {result ? <>
      {trashed ? <p className="callout">These retained results belong to an evaluation in Trash. <a href={cleanupLink(record.id)}>Restore the evaluation</a> to download its files.</p> : null}
      <label className="label">Metrics by prediction unit<select className="field" value={unit} onChange={(event) => setUnit(event.target.value as typeof unit)}><option value="selected">Target unit ({metrics?.unit ?? 'configured'})</option><option value="slide">Slide</option><option value="patient">Patient</option></select></label>
      {metrics?.positiveClass && typeof metrics.decisionThreshold === 'number' ? <p className="muted">Predict {metrics.positiveClass} when its probability is at least {metrics.decisionThreshold}. Ranking metrics use the original probabilities.</p> : null}
      {typeof selected?.unlabeledCount === 'number' && selected.unlabeledCount > 0 ? <p className="muted">{selected.unlabeledCount} unlabeled {unit === 'selected' ? metrics?.unit : unit} records have predictions and are excluded from these metrics.</p> : null}
      {selected?.available ? <><p>{selected.count} labeled {unit === 'selected' ? metrics?.unit : unit} records. Classes: {metrics?.classOrder.join(', ')}.</p><div className="chain-metrics">{(['accuracy', 'balancedAccuracy', 'macroF1', 'auroc', 'auprc', 'loss'] as const).map((key) => <div key={key}><strong>{typeof selected[key] === 'number' ? selected[key].toFixed(4) : 'Unavailable'}</strong><span>{{ accuracy: 'Accuracy', balancedAccuracy: 'Balanced accuracy', macroF1: 'Macro F1', auroc: 'AUROC', auprc: 'AUPRC', loss: 'Log loss' }[key]}</span></div>)}</div>
      {selected.missingClasses?.length ? <p className="callout">Classes absent from labeled test records: {selected.missingClasses.join(', ')}. Some metrics cannot be estimated.</p> : null}
      {selected.confusionMatrix ? <details><summary>Confusion matrix</summary><p>Rows: actual class. Columns: predicted class.</p><div className="table-wrap"><table className="chain-table"><thead><tr><th>Actual / predicted</th>{metrics?.classOrder.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{selected.confusionMatrix.map((row, index) => <tr key={index}><th scope="row">{metrics?.classOrder[index]}</th>{row.map((value, column) => <td key={column}>{value}</td>)}</tr>)}</tbody></table></div></details> : null}</> : <p className="callout">{selected?.reason ?? 'No labeled records are available for these metrics.'} Predictions are still available.</p>}
      <div className="inline-actions"><button className="btn btn-secondary" disabled={downloading || trashed} onClick={() => void download('slide-predictions.csv')}>Download slide predictions</button><button className="btn btn-secondary" disabled={downloading || trashed} onClick={() => void download('patient-predictions.csv')}>Download patient predictions</button><button className="btn btn-secondary" disabled={downloading || trashed} onClick={() => void download('metrics.json')}>Download metrics</button></div>
      {!trashed ? <div className="inline-actions"><a className="btn btn-primary" href={evidenceLink('clinical-utility', { experimentId: record.manifest.experimentId, predictorId: record.manifest.predictorId, evaluationId: record.id })}>Next: analyze clinical utility</a><a className="btn btn-secondary" href={evidenceLink('interpretation', { experimentId: record.manifest.experimentId, predictorId: record.manifest.predictorId, evaluationId: record.id })}>Inspect slide attention</a></div> : null}
    </> : <p>{execution.isError ? 'Results cannot be verified. Resolve the execution error before viewing or downloading results.' : 'No completed results yet. Run this evaluation to generate predictions and metrics.'}</p>}
    <div className="inline-actions"><button className="btn btn-secondary" type="button" onClick={() => downloadJSON(`${record.manifest.name}.json`, { ...record, execution: current })}>Export evaluation record</button></div>
    <details><summary>Saved inputs, settings and results</summary><pre className="chain-details">{JSON.stringify({ manifest: record.manifest, execution: current }, null, 2)}</pre></details>
  </Panel>;
}
