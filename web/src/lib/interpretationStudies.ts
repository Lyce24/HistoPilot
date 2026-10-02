import type { Interpretation } from '../api/interpretation';

/** One attention study: the slides interpreted together with one model and one set of features. */
export interface InterpretationStudy {
  key: string; records: Interpretation[]; predictorId: string; experimentId: string; createdAt: string; slides: number;
}

/**
 * Saved attention records as the studies they were made in. A study is saved one record per
 * slide, so records of one model, feature bundle, representation and slide folder that were
 * saved within `gapMinutes` of each other belong to one study. Newest study first; records
 * keep the order they were saved in.
 */
export function interpretationStudies(records: readonly Interpretation[], gapMinutes = 30): InterpretationStudy[] {
  const sorted = [...records].sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
  const studies: InterpretationStudy[] = [];
  const open = new Map<string, InterpretationStudy>();
  for (const record of sorted) {
    const manifest = record.manifest;
    // Pooled studies differ from the primary model's own studies by their models and function.
    const pooled = (manifest.predictors ?? []).map((item) => item.id).join(',');
    const signature = [manifest.predictorId, manifest.featureBundleId, manifest.packArtifactId ?? '', manifest.slideFolder, pooled, manifest.pooling ?? ''].join('\u0000');
    const current = open.get(signature);
    const last = current?.records.at(-1);
    if (current && last && Date.parse(record.createdAt) - Date.parse(last.createdAt) <= gapMinutes * 60000) {
      current.records.push(record);
      current.slides += manifest.slides.length;
      continue;
    }
    const study = { key: record.id, records: [record], predictorId: manifest.predictorId, experimentId: manifest.experimentId, createdAt: record.createdAt, slides: manifest.slides.length };
    studies.push(study);
    open.set(signature, study);
  }
  return studies.reverse();
}
