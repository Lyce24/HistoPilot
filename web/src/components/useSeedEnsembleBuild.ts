import { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { ModelExperiment } from '../api/experiments';
import { predictors, type FrozenPredictor, type SeedEnsembleChoice } from '../api/predictors';
import { seedEnsembleSelection } from '../lib/applyModels';

export const seedEnsembleKey = (choice: Pick<SeedEnsembleChoice, 'batchId' | 'candidateId'>) => `${choice.batchId}:${choice.candidateId}`;

/**
 * Builds one configuration's seed ensemble: review, then freeze. A lost response keeps its
 * request identity, so building again cannot create a second predictor. Nothing trains.
 */
export function useSeedEnsembleBuild(project: string, record: Pick<ModelExperiment, 'id' | 'name'>) {
  const client = useQueryClient();
  const inFlight = useRef(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState<Error | null>(null);
  const [pending, setPending] = useState<{ key: string; previewHash: string; operationId: string } | null>(null);
  async function build(choice: SeedEnsembleChoice): Promise<FrozenPredictor | null> {
    if (inFlight.current) return null;
    const key = seedEnsembleKey(choice);
    const selection = seedEnsembleSelection(record, choice);
    inFlight.current = true; setBusy(key); setError(null);
    let built: FrozenPredictor | null = null;
    try {
      let attempt = pending?.key === key ? pending : null;
      if (!attempt) {
        const preview = await predictors.previewSeedEnsemble(project, selection);
        if (!preview.canFreeze || !preview.previewHash) throw new Error(preview.findings.map((finding) => finding.message).join(' ') || 'This seed ensemble cannot be built.');
        attempt = { key, previewHash: preview.previewHash, operationId: crypto.randomUUID() };
        setPending(attempt);
      }
      built = await predictors.freezeSeedEnsemble(project, selection, attempt.previewHash, attempt.operationId);
      setPending(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Could not build the seed ensemble.'));
      if (reason instanceof ApiError) setPending(null);
    } finally { inFlight.current = false; setBusy(''); }
    await Promise.all([client.invalidateQueries({ queryKey: ['seed-ensembles', project, record.id] }), client.invalidateQueries({ queryKey: ['predictors', project] })]);
    return built;
  }
  return { busy, error, pending, build };
}
