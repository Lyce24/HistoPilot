import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { interpretations, type Interpretation } from '../api/interpretation';
import { computeActive } from '../api/predictors';
import { ErrorNotice, Panel } from './ui';
import ComputeJobControls from './ComputeJobControls';
import EvidenceChain from './EvidenceChain';
import AttentionSlideViewer from './AttentionSlideViewer';
export default function InterpretationStudy({ project, record }: { project: string; record: Interpretation }) {
  const execution = useQuery({ queryKey: ['compute-job', project, 'interpretation', record.id], queryFn: () => interpretations.execution(project, record.id), initialData: record.execution, refetchInterval: (query) => computeActive(query.state.data) ? 2500 : 10000, enabled: record.lifecycleState !== 'trashed' });
  const [slideId, setSlideId] = useState(record.manifest.slides[0]?.slideId ?? '');
  const slide = record.manifest.slides.find((item) => item.slideId === slideId);
  const result = !execution.isError && execution.data?.status === 'completed' ? execution.data.result?.slides?.find((item) => item.slideId === slideId) : undefined;
  return <Panel title={record.manifest.name} subtitle="Frozen slide inputs, checkpoint identity and persistent attention computation">
    <EvidenceChain current="interpretation" experimentId={record.manifest.experimentId} predictorId={record.manifest.predictorId} evaluationId={record.manifest.evaluationId ?? undefined} clinicalAnalysisId={record.manifest.clinicalAnalysisId ?? undefined} />
    <ComputeJobControls project={project} id={record.id} kind="interpretation" initial={record.execution} readOnly={record.lifecycleState === 'archived' || record.lifecycleState === 'trashed'} onComplete={() => void execution.refetch()} />
    <ErrorNotice error={execution.error} />
    <label className="label">Slide to inspect<select className="field" value={slideId} onChange={(event) => setSlideId(event.target.value)}>{record.manifest.slides.map((item) => <option key={item.slideId} value={item.slideId}>{item.slideId} · {item.patchCount.toLocaleString()} patches</option>)}</select></label>
    {slide && result ? <AttentionSlideViewer key={`${record.id}:${slide.slideId}`} project={project} record={record} slide={slide} result={result} /> : <p className="callout">{execution.isError ? 'The attention result could not be verified.' : 'Compute this study to overlay the predictor’s actual attention on each selected slide.'}</p>}
    <details><summary>Slide files and frozen provenance</summary><pre className="chain-details">{JSON.stringify(record.manifest, null, 2)}</pre></details>
  </Panel>;
}
