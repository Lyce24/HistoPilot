import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { EvaluationCohort } from '../api/evaluation';
import { references, type ReferencePreview, type ReferenceSelection, type ReferenceStandard } from '../api/references';
import { cohortName } from '../lib/applyModels';
import { datasetVersionLabel } from '../lib/versionLabels';
import PublicationConfirmation from './PublicationConfirmation';
import { Findings, useDatasets } from './ScientificUI';
import { useReviewedPublication } from './useReviewedPublication';
import { ErrorNotice } from './ui';
import './ApplyRunDetail.css';

/** The class a source value names, ignoring case and surrounding spaces, or null. */
export function suggestedClass(value: string, classes: readonly string[]) {
  const key = value.trim().toLowerCase();
  return classes.find((item) => item.trim().toLowerCase() === key) ?? null;
}

/**
 * Labels attached to a frozen cohort after it was frozen: one column of frozen datasets (the
 * cohort's own, or newer versions matched by slide ID), mapped to a class set. The cohort and
 * its runs never change; every run on the cohort with these classes can be scored against it.
 */
export default function ReferenceStandardEditor({ project, cohort, classes, onSaved, onCancel }: {
  project: string; cohort: EvaluationCohort; classes: readonly string[];
  onSaved: (standard: ReferenceStandard) => void; onCancel: () => void;
}) {
  const client = useQueryClient();
  const datasets = useDatasets(project);
  const [name, setName] = useState('');
  const [datasetIds, setDatasetIds] = useState<string[]>([cohort.manifest.datasetId]);
  const [field, setField] = useState('');
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [added, setAdded] = useState<string[]>([]);
  const [extra, setExtra] = useState('');
  const [seeded, setSeeded] = useState('');
  const publication = useReviewedPublication(
    (selection: ReferenceSelection) => references.preview(project, selection),
    (selection, hash, operation) => references.save(project, selection, hash, operation),
    (preview: ReferencePreview) => preview.canSave,
    async (saved) => {
      // A new label source changes the cohort's references and every agreement on it.
      await Promise.all([client.invalidateQueries({ queryKey: ['reference-standards', project] }), client.invalidateQueries({ queryKey: ['run-agreement', project] })]);
      onSaved(saved);
    },
  );
  // The cohort's own dataset first, then the others, newest first.
  const ordered = [...datasets.data?.datasets ?? []].sort((a, b) => Number(b.id === cohort.manifest.datasetId) - Number(a.id === cohort.manifest.datasetId) || b.createdAt.localeCompare(a.createdAt));
  const chosen = ordered.filter((item) => datasetIds.includes(item.id));
  // A column must be in every chosen dataset's data dictionary.
  const columns = chosen.length ? (chosen[0].manifest.dictionary ?? []).filter((column) => chosen.every((item) => item.manifest.dictionary?.some((other) => other.key === column.key))) : [];
  const selection: ReferenceSelection = { cohortId: cohort.id, name: name.trim(), datasetIds, field, classes: [...classes], labels };
  // The column's values over the cohort's slides, before anything is mapped.
  const source = JSON.stringify([datasetIds, field]);
  const values = useQuery({
    queryKey: ['reference-values', project, cohort.id, datasetIds, field],
    queryFn: () => references.preview(project, { ...selection, name: 'Values', labels: {} }),
    enabled: Boolean(field && datasetIds.length),
    staleTime: 60000,
  });
  const summary = values.data?.manifest?.summary;
  // Values that name a class are mapped to it once, when the column's values first arrive.
  if (summary && seeded !== source) {
    setSeeded(source);
    setLabels(Object.fromEntries(summary.values.flatMap((item) => { const label = suggestedClass(item.value, classes); return label ? [[item.value, label]] : []; })));
  }
  const listed = [...summary?.values ?? [], ...added.filter((value) => !summary?.values.some((item) => item.value === value)).map((value) => ({ value, count: null }))];
  const change = (update: () => void) => { publication.reset(); update(); };
  function chooseSource(ids: string[], column: string) {
    change(() => { setDatasetIds(ids); setField(column); setLabels({}); setAdded([]); setSeeded(''); });
  }
  const mapped = Object.keys(labels).length;
  const canReview = Boolean(selection.name && field && datasetIds.length && mapped && !values.isFetching);
  const review = publication.review?.preview;
  const reviewed = review?.manifest?.summary;
  return <section className="run-section reference-editor" aria-labelledby={`reference-editor-${cohort.id}`}>
    <div className="run-section-heading"><h3 id={`reference-editor-${cohort.id}`}>Add a reference standard</h3></div>
    <p className="muted">Labels for {cohortName(cohort, cohort.id)} that arrived after it was frozen, such as a reader&rsquo;s grades or a consensus. Choose the dataset column that holds them and map its values to {classes.join(', ')}. The cohort and its runs never change; every run on this cohort with these classes can then be scored against the reference.</p>
    <ErrorNotice error={datasets.error ?? values.error ?? publication.error} />
    {!review ? <fieldset className="chain-fields" disabled={publication.locked}><legend className="sr-only">Reference standard settings</legend>
      <label className="label">Name<input className="field" value={name} maxLength={120} placeholder="For example, Reader A or Consensus" onChange={(event) => change(() => setName(event.target.value))} /></label>
      <label className="label">Column<select className="field" value={field} onChange={(event) => { const column = columns.find((item) => item.key === event.target.value); chooseSource(datasetIds, event.target.value); if (column && !name.trim()) setName(column.sourceColumn.slice(0, 120)); }}><option value="">Choose a column</option>{columns.map((column) => <option key={column.key} value={column.key}>{column.sourceColumn}</option>)}</select></label>
      <fieldset className="chain-wide reference-datasets"><legend>Datasets</legend>
        <p className="muted">Slides are matched to the cohort by slide ID, so a newer version of the cohort&rsquo;s dataset can supply labels its frozen version lacked.</p>
        {datasets.isPending ? <p role="status">Loading datasets…</p> : ordered.map((dataset) => <label key={dataset.id} className="science-check"><input type="checkbox" checked={datasetIds.includes(dataset.id)} onChange={(event) => { const ids = event.target.checked ? [...datasetIds, dataset.id] : datasetIds.filter((item) => item !== dataset.id); chooseSource(ids, field); }} /><span>{datasetVersionLabel(dataset)}<small>{dataset.id === cohort.manifest.datasetId ? 'This cohort’s dataset · ' : ''}saved {new Date(dataset.createdAt).toLocaleDateString()}</small></span></label>)}
        {chosen.length && !columns.length ? <p className="callout">The chosen datasets share no column.</p> : null}
      </fieldset>
      {field ? <div className="chain-wide stack">
        {values.isFetching ? <p role="status">Reading the column&rsquo;s values over this cohort…</p> : null}
        {summary ? <>
          <div className="table-wrap"><table className="chain-table" aria-label="Value mapping"><thead><tr><th scope="col">Value</th><th scope="col">Cohort slides</th><th scope="col">Class</th></tr></thead><tbody>
            {listed.map((item) => <tr key={item.value}><th scope="row">{item.value || '(empty)'}</th><td>{item.count === null ? '—' : item.count.toLocaleString()}</td><td><select className="field" aria-label={`Class for ${item.value || 'empty'}`} value={labels[item.value] ?? ''} onChange={(event) => { const label = event.target.value; change(() => setLabels((current) => { const next = { ...current }; if (label) next[item.value] = label; else delete next[item.value]; return next; })); }}><option value="">Leave unlabeled</option>{classes.map((label) => <option key={label} value={label}>{label}</option>)}</select></td></tr>)}
          </tbody></table></div>
          {summary.valuesTruncated ? <div className="inline-actions"><p className="muted">Only the most frequent values are listed. Add any other exact value to map it.</p><label className="label">Exact value<input className="field" value={extra} maxLength={4096} onChange={(event) => setExtra(event.target.value)} /></label><button type="button" className="btn btn-secondary btn-small" disabled={!extra || listed.some((item) => item.value === extra)} onClick={() => { setAdded((current) => [...current, extra]); setExtra(''); }}>Add value</button></div> : null}
          <p className="muted">{[
            summary.missingSlides ? `${summary.missingSlides.toLocaleString()} cohort slides have no value.` : '',
            summary.unmatchedSlides ? `${summary.unmatchedSlides.toLocaleString()} cohort slides are not in the chosen datasets.` : '',
          ].filter(Boolean).join(' ')} Slides without a value, outside the chosen datasets or with a value left unlabeled are never scored against this reference.</p>
        </> : null}
      </div> : null}
      <div className="inline-actions chain-wide">
        <button type="button" className="btn btn-primary" disabled={!canReview || publication.locked} onClick={() => { if (canReview) void publication.preview(selection); }}>{publication.busy ? 'Checking…' : 'Review reference standard'}</button>
        <button type="button" className="text-button" disabled={publication.locked} onClick={onCancel}>Cancel</button>
      </div>
    </fieldset> : <>
      {reviewed ? <>
        <p className="evidence-counts"><strong>{reviewed.labeledSlides.toLocaleString()} of {reviewed.slides.toLocaleString()} cohort slides labeled</strong>{classes.map((label) => ` · ${label} ${(reviewed.classCounts[label] ?? 0).toLocaleString()}`).join('')}</p>
        <p className="muted">From the column {field} of {chosen.map(datasetVersionLabel).join(', ')}. {[
          reviewed.missingSlides ? `${reviewed.missingSlides.toLocaleString()} without a value` : '',
          reviewed.unmappedSlides ? `${reviewed.unmappedSlides.toLocaleString()} with an unmapped value` : '',
          reviewed.unmatchedSlides ? `${reviewed.unmatchedSlides.toLocaleString()} not in the chosen datasets` : '',
        ].filter(Boolean).join(' · ') || 'Every cohort slide is labeled.'}</p>
      </> : null}
      <Findings findings={review.findings} />
      <div className="inline-actions"><button type="button" className="text-button" disabled={publication.locked} onClick={() => publication.reset()}>Back to the mapping</button></div>
      {review.canSave ? <PublicationConfirmation busy={publication.busy} uncertain={publication.review!.uncertain} acknowledged={publication.acknowledged} onAcknowledge={publication.setAcknowledged} onConfirm={() => void publication.publish()} label="Save reference standard" /> : null}
    </>}
  </section>;
}
