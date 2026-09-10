import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { api } from '../api/client';
import { workspaceKey } from '../api/queries';
import { Badge, ErrorNotice, Icon, Metric, PageHeader, Panel } from '../components/ui';
import { useUIStore } from '../store/ui';
import { downloadJSON } from '../lib/download';

export default function Cohort({ workspace: w }: { workspace: Workspace }) {
  const [specimenType, setSpecimenType] = useState('Primary');
  const [msi, setMsi] = useState('Any');
  const [braf, setBraf] = useState('Any');
  const selectedId = useUIStore((state) => state.selectedCohortId);
  const setSelected = useUIStore((state) => state.setSelectedCohortId);
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: api.saveCohort,
    onSuccess: async (cohort) => {
      setSelected(cohort.id);
      await client.invalidateQueries({ queryKey: workspaceKey });
    },
  });
  const matching = w.patients.filter(
    (patient) =>
      (specimenType === 'Any' || patient.specimenType === specimenType) &&
      (msi === 'Any' || patient.msi === msi) &&
      (braf === 'Any' || patient.braf === braf),
  );
  const ids = new Set(matching.map((patient) => patient.id));
  const slides = w.slides.filter((slide) => ids.has(slide.patientId));
  const mutants = matching.filter((patient) => patient.kras === 'Mutant').length;
  const frozen =
    w.cohortSnapshots.find((cohort) => cohort.id === selectedId) ?? w.cohortSnapshots.at(-1);
  const partitions = ['train', 'validation', 'test'].map((partition) => ({
    partition,
    ids: new Set(
      matching.filter((patient) => patient.partition === partition).map((patient) => patient.id),
    ),
  }));
  const overlap = [...ids].filter(
    (id) => partitions.filter((partition) => partition.ids.has(id)).length > 1,
  ).length;
  const changed =
    frozen &&
    (frozen.filters.specimenType !== specimenType ||
      frozen.filters.msi !== msi ||
      frozen.filters.braf !== braf);
  const saveCurrent = () => save.mutate({ datasetId: w.dataset.id, specimenType, msi, braf });
  return (
    <>
      <PageHeader
        eyebrow="02 / POPULATION"
        title="Cohort builder"
        description="Define who belongs in the analysis. The service validates the selection and saves an immutable cohort snapshot in your local workspace."
        actions={
          <>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setSpecimenType('Any');
                setMsi('Any');
                setBraf('Any');
              }}
            >
              Reset filters
            </button>
            <button
              className="btn btn-primary"
              disabled={!matching.length || save.isPending}
              onClick={saveCurrent}
            >
              <Icon name="lock" />
              {save.isPending ? 'Saving cohort…' : 'Save cohort'}
            </button>
          </>
        }
      />
      <ErrorNotice error={save.error} />
      <div className="grid-4">
        <Metric
          label="Preview patients"
          value={
            <>
              {matching.length}
              <span className="metric-denominator"> / {w.patients.length}</span>
            </>
          }
          note="Unsaved filter preview"
        />
        <Metric label="Included slides" value={slides.length} note="All linked slides retained" />
        <Metric
          label="KRAS mutant"
          value={mutants}
          note={`${matching.length ? ((mutants / matching.length) * 100).toFixed(1) : 0}% of selected patients`}
        />
        <Metric label="Analysis unit" value="Patient" note="Slides stay grouped by patient" />
      </div>
      <div className="grid-2">
        <Panel
          title="Population definition"
          subtitle="All conditions must match"
          actions={<Badge tone="purple">Live preview</Badge>}
        >
          <div className="stack">
            {[
              {
                label: 'Specimen type',
                value: specimenType,
                values: ['Any', 'Primary', 'Metastatic'],
                set: setSpecimenType,
              },
              { label: 'MSI status', value: msi, values: ['Any', 'MSS', 'MSI-H'], set: setMsi },
              { label: 'BRAF status', value: braf, values: ['Any', 'WT', 'Mutant'], set: setBraf },
            ].map((filter) => (
              <label className="filter-definition" key={filter.label}>
                <span className="filter-label">{filter.label}</span>
                <span className="muted">is</span>
                <select
                  className="field"
                  value={filter.value}
                  onChange={(event) => filter.set(event.target.value)}
                >
                  {filter.values.map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className="query-preview">
            <span className="eyebrow">Analysis target</span>
            <strong>KRAS mutation status</strong>
            <span className="muted">Binary classification · patient labels</span>
          </div>
          <div className="callout">
            <strong>{matching.length} patients match this preview.</strong> Save cohort sends the
            definition to Python, which resolves and stores the authoritative patient IDs.
          </div>
        </Panel>
        <Panel
          title="Split & entity checks"
          subtitle="Demonstration checks on synthetic service records"
        >
          <ul className="detail-list">
            <li>
              <div>
                <strong>Patient partition overlap</strong>
                <p className="muted">IDs present in more than one fixture partition</p>
              </div>
              <Badge tone={overlap ? 'amber' : 'green'}>{overlap} overlapping</Badge>
            </li>
            <li>
              <div>
                <strong>Selected slide-to-patient links</strong>
                <p className="muted">Slide records resolving to the preview population</p>
              </div>
              <Badge tone="green">{slides.length} linked</Badge>
            </li>
            <li>
              <div>
                <strong>Source split</strong>
                <p className="muted mono">{w.split.id}</p>
              </div>
              <Badge>Seed {w.split.seed}</Badge>
            </li>
          </ul>
          <div className="callout">
            These checks illustrate the audit surface. Real slide duplicate detection, label
            reconciliation, and CV split generation remain planned.
          </div>
        </Panel>
      </div>
      <div className="grid-2">
        <Panel
          title="Partition composition"
          subtitle="Existing fixture assignments · no split is generated here"
        >
          <div className="partition-bar" aria-label="Selected patient partition composition">
            {partitions
              .filter((partition) => partition.ids.size)
              .map(({ partition, ids }) => (
                <span
                  key={partition}
                  className={partition}
                  style={{ flex: ids.size }}
                  title={`${partition}: ${ids.size}`}
                />
              ))}
          </div>
          <div className="partition-legend">
            {partitions.map(({ partition, ids }) => (
              <div key={partition}>
                <span className={`partition-dot ${partition}`} />
                <span>{partition}</span>
                <strong>{ids.size}</strong>
              </div>
            ))}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Site</th>
                  <th>Patients</th>
                  <th>KRAS mutant</th>
                  <th>Mutant fraction</th>
                </tr>
              </thead>
              <tbody>
                {[...new Set(w.patients.map((patient) => patient.site))].map((site) => {
                  const patients = matching.filter((patient) => patient.site === site);
                  const count = patients.filter((patient) => patient.kras === 'Mutant').length;
                  return (
                    <tr key={site}>
                      <td>{site}</td>
                      <td>{patients.length}</td>
                      <td>{count}</td>
                      <td>
                        {patients.length ? `${Math.round((count / patients.length) * 100)}%` : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel
          title="Saved cohort snapshots"
          subtitle="Persisted in the service workspace · available after refresh"
          actions={<Badge>{w.cohortSnapshots.length} saved</Badge>}
        >
          {frozen ? (
            <>
              <label className="label">
                Selected snapshot
                <select
                  className="field"
                  value={frozen.id}
                  onChange={(event) => setSelected(event.target.value)}
                >
                  {w.cohortSnapshots.map((cohort) => (
                    <option key={cohort.id} value={cohort.id}>
                      {cohort.name} · {cohort.patientIds.length} patients · {cohort.id}
                    </option>
                  ))}
                </select>
              </label>
              <div className="snapshot-heading">
                <span className="snapshot-icon">
                  <Icon name="lock" size={22} />
                </span>
                <div>
                  <h3>{frozen.name}</h3>
                  <span className="muted mono">{frozen.id}</span>
                </div>
                <Badge tone="purple">Saved on server</Badge>
              </div>
              <ul className="detail-list">
                <li>
                  <span>Dataset</span>
                  <strong className="mono">{frozen.datasetId}</strong>
                </li>
                <li>
                  <span>Population</span>
                  <strong>
                    {frozen.patientIds.length} patients · {frozen.slideIds.length} slides
                  </strong>
                </li>
                <li>
                  <span>Definition</span>
                  <strong>
                    {frozen.filters.specimenType} · {frozen.filters.msi} · BRAF{' '}
                    {frozen.filters.braf}
                  </strong>
                </li>
              </ul>
              {changed ? (
                <p className="muted">
                  Current filters differ from this saved snapshot. Save a new cohort to use the new
                  definition.
                </p>
              ) : null}
              <div className="inline-actions">
                <button
                  className="btn btn-secondary"
                  onClick={() => downloadJSON(`${frozen.id}.json`, frozen)}
                >
                  <Icon name="download" />
                  Export cohort
                </button>
                <a className="btn btn-primary" href="#experiments">
                  Configure experiment <Icon name="arrow" />
                </a>
              </div>
            </>
          ) : (
            <div className="cohort-empty">
              <Icon name="features" size={32} />
              <h3>Your population, pinned</h3>
              <p className="muted">
                Review the preview counts, then save a cohort. The service assigns an ID and retains
                the selected patient and slide references.
              </p>
              <button
                className="btn btn-secondary"
                disabled={!matching.length || save.isPending}
                onClick={saveCurrent}
              >
                Save {matching.length} patients
              </button>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
