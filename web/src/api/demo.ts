import type { FrozenBatch, TrainingExecution, TrainingHistory, TrainingResourceHistory } from './development';
import type { RoadmapModuleId } from '../lib/roadmap';

export interface DemoStep {
  id: string;
  title: string;
  description: string;
  facts?: { label: string; value: string }[];
  notice?: string;
  table?: { title?: string; columns: string[]; rows: (string | number)[][] };
  chart?: { title: string; xLabel: string; yLabel: string; series: { label: string; points: { x: number; y: number | null }[] }[]; yRange?: [number, number] };
  runs?: { batch: FrozenBatch; execution: TrainingExecution; histories: Record<string, TrainingHistory>; resources: TrainingResourceHistory };
}
export interface DemoRecord {
  id: string;
  module: RoadmapModuleId;
  name: string;
  description: string;
  tags: string[];
  steps: DemoStep[];
}
export interface DemoPipeline {
  version: 1;
  synthetic: true;
  readOnly: true;
  seed: number;
  sourceBasis: string[];
  records: DemoRecord[];
}
