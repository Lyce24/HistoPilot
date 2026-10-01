import { useState } from 'react';
import type { Workspace } from '../api/types';
import type { ProtocolSpec } from '../api/scientific';
import { useConfigurations, useDrafts } from '../components/ScientificUI';
import { Badge, ErrorNotice, PageHeader, Panel } from '../components/ui';
import { StageBackButton } from '../components/StageActions';
import { versionLabelText } from '../lib/versionLabels';
import { splitModeLabel, unitLabel } from '../lib/labels';
import { historicalProtocols } from '../lib/protocol';

/** Historical combined protocols stay inspectable; new training designs belong to experiments. */
export default function HistoricalProtocols({ workspace }: { workspace: Workspace }) {
  const versions = useConfigurations(workspace.project.id, 'protocol');
  const drafts = useDrafts(workspace.project.id);
  const [selected, setSelected] = useState('');
  // Training designs derived for an experiment's inputs are listed with their experiments, not here.
  const history = historicalProtocols(versions.data?.configurations ?? [], drafts.data?.drafts ?? []);
  const records = [
    ...history.versions.map((item) => ({ id: item.id, name: versionLabelText(item, 'Protocol'), status: 'Frozen', spec: item.manifest.spec as ProtocolSpec, document: item.manifest })),
    ...history.drafts.map((item) => ({ id: item.id, name: item.name, status: 'Historical draft', spec: item.payload.spec as ProtocolSpec, document: item.payload })),
  ];
  const record = records.find((item) => item.id === selected);
  return <div className="clinical-workspace">
    <PageHeader eyebrow="SAVED HISTORY" title={record?.name ?? 'Historical protocols'} description="Earlier combined target and training designs are preserved for inspection. Create new training/testing populations in Targets & Splits and training designs in Experiments." actions={record ? <StageBackButton onClick={() => setSelected('')}>Back to historical protocols</StageBackButton> : <StageBackButton href="#cohort">Back to Targets &amp; Splits</StageBackButton>} />
    <ErrorNotice error={versions.error ?? drafts.error} />
    {record ? <Panel title="Saved protocol"><Badge>{record.status}</Badge><dl className="science-summary"><div><dt>Dataset</dt><dd>{record.spec.datasetId}</dd></div><div><dt>Target</dt><dd>{record.spec.target?.field || 'Not configured'} · {unitLabel(record.spec.target?.unit ?? 'patient')}</dd></div><div><dt>Labels</dt><dd>{record.spec.target?.classes?.join(', ') || 'Not configured'}</dd></div><div><dt>Historical training design</dt><dd>{record.spec.split ? splitModeLabel(record.spec.split.mode) : 'Not configured'}</dd></div></dl><details><summary>Exact saved specification and memberships</summary><pre className="code-block">{JSON.stringify(record.document, null, 2)}</pre></details></Panel> : <Panel title="Saved protocols and drafts">
      {versions.isPending || drafts.isPending ? <p role="status">Loading historical protocols…</p> : records.length ? <div className="table-wrap"><table><thead><tr><th>Name</th><th>Status</th><th>Dataset</th><th>Target</th><th>Training design</th></tr></thead><tbody>{records.map((item) => <tr key={item.id}><th scope="row"><button className="text-button" onClick={() => setSelected(item.id)}>{item.name}</button></th><td><Badge>{item.status}</Badge></td><td>{item.spec.datasetId}</td><td>{item.spec.target?.field || 'Not configured'}</td><td>{item.spec.split ? splitModeLabel(item.spec.split.mode) : 'Not configured'}</td></tr>)}</tbody></table></div> : <p>No historical protocols are available.</p>}
      <p className="muted">Existing experiments retain their saved protocols. Use an existing experiment as a template in <a href="#experiments">Experiments</a> to prepare a new design.</p>
    </Panel>}
  </div>;
}
