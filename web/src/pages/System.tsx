import { useSystem } from '../api/queries';
import { Badge, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';

export default function System() {
  const system = useSystem();
  const data = system.data;
  return (
    <>
      <PageHeader
        eyebrow="LOCAL WORKSPACE"
        title="System & storage"
        description="The browser is your interface. The Python control service owns workspace metadata, and future isolated workers will perform computation."
        actions={
          <button
            className="btn btn-secondary"
            disabled={system.isFetching}
            onClick={() => void system.refetch()}
          >
            <Icon name="reset" />
            Refresh status
          </button>
        }
      />
      <ErrorNotice error={system.error} />
      {system.isPending ? <p role="status">Reading control service status…</p> : null}
      {data ? (
        <>
          <div className="grid-2">
            <Panel
              title="Control service"
              subtitle="Current service-reported state"
              actions={
                <Badge tone={system.isError ? 'amber' : 'green'}>
                  {system.isError ? 'Last known state' : 'Connected'}
                </Badge>
              }
            >
              <ul className="detail-list">
                <li>
                  <span>Application mode</span>
                  <strong>{data.mode}</strong>
                </li>
                <li>
                  <span>Process role</span>
                  <strong>{data.control.process}</strong>
                </li>
                <li>
                  <span>CUDA model state in this process</span>
                  <strong>{data.control.cudaModelsLoaded ? 'Loaded' : 'None'}</strong>
                </li>
                <li>
                  <span>Browser connection</span>
                  <strong className="mono">{window.location.origin}</strong>
                </li>
              </ul>
              <p className="muted">
                The control service coordinates records and job specifications. GPU model execution
                belongs in separate worker processes.
              </p>
            </Panel>
            <Panel
              title="Local workspace"
              subtitle="Metadata and artifact roots on the Python host"
            >
              <ul className="detail-list">
                <li>
                  <span>Workspace location</span>
                  <strong className="mono break-all">{data.workspace}</strong>
                </li>
                <li>
                  <span>Application database</span>
                  <strong>{data.storage.engine}</strong>
                </li>
                <li>
                  <span>Journal mode</span>
                  <strong>{data.storage.journalMode.toUpperCase()}</strong>
                </li>
                <li>
                  <span>Schema version</span>
                  <strong>{data.storage.schemaVersion}</strong>
                </li>
                <li>
                  <span>Original WSI policy</span>
                  <strong>
                    {data.sourcesReadOnly ? 'Referenced read-only' : 'See server configuration'}
                  </strong>
                </li>
              </ul>
            </Panel>
          </div>
          <div className="grid-2">
            <Panel
              title="Compute workers"
              subtitle="Execution boundary"
              actions={
                <Badge tone="amber">
                  {data.workers.executionEnabled ? 'Enabled' : 'Not implemented'}
                </Badge>
              }
            >
              <p className="muted">
                Worker status: <strong>{data.workers.status}</strong>. This skeleton does not
                execute WSI preprocessing, feature extraction, or MIL training. GPU inventory and
                live worker diagnostics will be reported here when connected.
              </p>
              <div className="callout">
                Saving an experiment creates a persisted draft. It does not launch a compute
                process.
              </div>
            </Panel>
            <Panel
              title="Storage responsibilities"
              subtitle="Large artifacts stay on the local filesystem"
            >
              <ul className="detail-list">
                <li>
                  <span>SQLite</span>
                  <strong>Application metadata</strong>
                </li>
                <li>
                  <span>DuckDB + Parquet</span>
                  <strong>Planned analytical tables</strong>
                </li>
                <li>
                  <span>HDF5</span>
                  <strong>Planned features & coordinates</strong>
                </li>
                <li>
                  <span>JSON / YAML</span>
                  <strong>Experiment manifests</strong>
                </li>
              </ul>
              <a className="text-link" href="#dataset">
                Manage server folder references →
              </a>
            </Panel>
          </div>
          <div className="callout">
            <Icon name="info" />
            When this browser runs on a laptop, the directory picker still shows the Python host’s
            filesystem. No WSI files pass through browser upload.
          </div>
        </>
      ) : null}
    </>
  );
}
