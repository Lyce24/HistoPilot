import { Fragment, useState } from 'react';
import type { Workspace } from '../api/types';
import { Badge, EmptyState, Icon, Metric, PageHeader, Panel } from '../components/ui';
import FolderPicker from '../components/FolderPicker';
import { download } from '../lib/download';

const columns = [
  'patient_id',
  'specimen_id',
  'slide_id',
  'filename',
  'site',
  'specimen_type',
  'msi',
  'braf',
  'kras',
  'partition',
];
export default function Dataset({ workspace: w }: { workspace: Workspace }) {
  const [search, setSearch] = useState('');
  const [site, setSite] = useState('All');
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [schema, setSchema] = useState(false);
  const filtered = w.patients.filter((patient) => {
    const text = [
      patient.id,
      patient.site,
      ...w.slides
        .filter((slide) => slide.patientId === patient.id)
        .flatMap((slide) => [slide.filename, slide.id, slide.specimenId]),
    ]
      .join(' ')
      .toLowerCase();
    return (site === 'All' || patient.site === site) && text.includes(search.trim().toLowerCase());
  });
  function exportRows() {
    const rows = filtered.flatMap((patient) =>
      w.slides
        .filter((slide) => slide.patientId === patient.id)
        .map((slide) =>
          [
            patient.id,
            slide.specimenId,
            slide.id,
            slide.filename,
            patient.site,
            patient.specimenType,
            patient.msi,
            patient.braf,
            patient.kras,
            patient.partition,
          ]
            .map((value) => `"${value.replaceAll('"', '""')}"`)
            .join(','),
        ),
    );
    download(
      'histopilot-synthetic-dataset.csv',
      [columns.join(','), ...rows].join('\n'),
      'text/csv',
    );
  }
  return (
    <>
      <PageHeader
        eyebrow="01 / WORKSPACE"
        title="Dataset workspace"
        description="Start with the data. Explore the patient, specimen, and slide hierarchy before defining your analysis population."
        actions={
          <>
            <button className="btn btn-secondary" onClick={() => setSchema(!schema)}>
              <Icon name="branch" />
              {schema ? 'Hide' : 'View'} data contract
            </button>
            <FolderPicker />
          </>
        }
      />
      <div className="grid-4">
        <Metric label="Patients" value={w.patients.length} note="Patient is the analysis unit" />
        <Metric
          label="Specimens"
          value={w.dataset.specimenCount}
          note="Linked to patient records"
        />
        <Metric label="Whole-slide images" value={w.slides.length} note="Synthetic slide records" />
        <Metric
          label="Dataset version"
          value="v1.0"
          note={<span className="mono">{w.dataset.id}</span>}
        />
      </div>
      <Panel
        title="Source directories"
        subtitle="References on the Python host · original WSIs stay in place"
        actions={<Badge>{w.sources.length} saved references</Badge>}
      >
        {w.sources.length ? (
          <ul className="detail-list">
            {w.sources.map((source) => (
              <li key={source.id}>
                <div>
                  <strong className="mono">{source.path}</strong>
                  <p className="muted">Directory reference · slide import not implemented</p>
                </div>
                <Badge tone="green">Read-only source</Badge>
              </li>
            ))}
          </ul>
        ) : (
          <div className="callout">
            Choose a folder on the server to save its location. The synthetic dataset below remains
            separate from these references.
          </div>
        )}
      </Panel>
      {schema ? (
        <Panel title="Dataset import contract" subtitle="The future table-ingestion boundary">
          <p className="muted">
            The service will resolve these columns into versioned entities. This workspace currently
            contains synthetic fixtures.
          </p>
          <pre className="code-block">{columns.join(',\n')}</pre>
          <button
            className="btn btn-secondary btn-small"
            onClick={() =>
              download('histopilot-dataset-template.csv', `${columns.join(',')}\n`, 'text/csv')
            }
          >
            <Icon name="download" />
            Download CSV template
          </button>
        </Panel>
      ) : null}
      <Panel
        title="Resolved entities"
        subtitle="Synthetic records supplied by the service · select a patient to inspect"
        actions={<Badge tone="purple">Server dataset</Badge>}
      >
        <div className="filter-row dataset-filters">
          <label className="search-field">
            <span className="sr-only">Search patients or slides</span>
            <Icon name="search" />
            <input
              className="field"
              type="search"
              placeholder="Search patients, specimens, or slides…"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
          <label className="inline-field">
            <span className="muted">Site</span>
            <select
              className="field"
              value={site}
              onChange={(event) => setSite(event.target.value)}
            >
              {['All', ...new Set(w.patients.map((patient) => patient.site))].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <button className="btn btn-secondary" onClick={exportRows}>
            <Icon name="download" />
            Export visible rows
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {[
                  'Patient',
                  'Site',
                  'Specimen',
                  'Slides',
                  'KRAS target',
                  'Partition',
                  'Review',
                ].map((heading) => (
                  <th key={heading} scope="col">
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((patient) => {
                const slides = w.slides.filter((slide) => slide.patientId === patient.id);
                const expanded = selectedPatient === patient.id;
                return (
                  <Fragment key={patient.id}>
                    <tr className={expanded ? 'row-selected' : ''}>
                      <td>
                        <button
                          className="table-link"
                          aria-expanded={expanded}
                          onClick={() => setSelectedPatient(expanded ? null : patient.id)}
                        >
                          <Icon name={expanded ? 'down' : 'chevron'} size={14} />
                          {patient.id}
                        </button>
                      </td>
                      <td>{patient.site}</td>
                      <td>{patient.specimenType}</td>
                      <td>{slides.length}</td>
                      <td>
                        <Badge tone={patient.kras === 'Mutant' ? 'purple' : 'neutral'}>
                          {patient.kras}
                        </Badge>
                      </td>
                      <td>
                        <span className={`partition-dot ${patient.partition}`} />
                        {patient.partition}
                      </td>
                      <td>
                        <Badge
                          tone={
                            slides.some((slide) => slide.status === 'Review') ? 'amber' : 'green'
                          }
                        >
                          {slides.some((slide) => slide.status === 'Review') ? 'Review' : 'Linked'}
                        </Badge>
                      </td>
                    </tr>
                    {expanded ? (
                      <tr className="expanded-row">
                        <td colSpan={7}>
                          <div className="entity-details">
                            <div>
                              <span className="eyebrow">Patient → specimen → slide</span>
                              <h3>{patient.id}</h3>
                              <p className="muted">
                                {patient.specimenType} · {patient.msi} · BRAF {patient.braf}
                              </p>
                            </div>
                            <div className="stack">
                              {slides.map((slide) => (
                                <div key={slide.id} className="entity-slide">
                                  <span className="entity-branch">↳</span>
                                  <div>
                                    <span className="mono">{slide.specimenId}</span>
                                    <br />
                                    <strong>{slide.filename}</strong>
                                    <br />
                                    <span className="muted mono">{slide.id}</span>
                                  </div>
                                  <Badge>{slide.status}</Badge>
                                </div>
                              ))}
                            </div>
                          </div>
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
        {filtered.length ? (
          <div className="table-footer">
            <span>
              {filtered.length} of {w.patients.length} patients
            </span>
            <span>Synthetic demonstration data</span>
          </div>
        ) : (
          <EmptyState
            title="No matching patients"
            description="Try another patient ID, filename, or site."
          />
        )}
      </Panel>
      <div className="grid-2">
        <Panel
          title="One hierarchy, every artifact"
          subtitle="Preserve the link back to the source"
        >
          <div className="hierarchy-strip">
            <span>
              <Icon name="patient" /> Patient
            </span>
            <span>→</span>
            <span>
              <Icon name="features" /> Specimen
            </span>
            <span>→</span>
            <span>
              <Icon name="explorer" /> Slide
            </span>
          </div>
          <p className="muted">
            Labels belong to a declared entity level. Features and predictions keep these source
            identifiers.
          </p>
        </Panel>
        <Panel
          title="Build the analysis population"
          subtitle="Start a reproducible cohort definition"
        >
          <p className="muted">
            Use clinical filters to preview the population, then let the service validate and save a
            cohort snapshot.
          </p>
          <a className="btn btn-primary" href="#cohort">
            Build a cohort <Icon name="arrow" />
          </a>
        </Panel>
      </div>
    </>
  );
}
