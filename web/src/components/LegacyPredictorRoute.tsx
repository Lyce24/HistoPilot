import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { predictors } from '../api/predictors';
import { useHashParameters } from '../lib/hashRoute';
import LocalPostDevelopment from '../pages/LocalPostDevelopment';
import { ErrorNotice } from './ui';
import { experimentPredictorLink } from '../lib/predictorGroups';

export default function LegacyPredictorRoute({ workspace }: { workspace: Workspace }) {
  const parameters = useHashParameters();
  const experimentId = parameters.get('experiment') ?? '';
  const predictorId = parameters.get('predictor') ?? '';
  const refitRecovery = parameters.get('tab') === 'refits';
  const registry = useQuery({ queryKey: ['predictors', workspace.project.id], queryFn: () => predictors.list(workspace.project.id), enabled: !refitRecovery && !experimentId && Boolean(predictorId) });
  const source = experimentId || registry.data?.items.find((item) => item.id === predictorId)?.manifest.experimentId;
  const target = source && !refitRecovery ? experimentPredictorLink(source, predictorId) : null;
  useEffect(() => { if (target) window.location.replace(target); }, [target]);
  if (target) return <p role="status">Predictors now belong to their experiment. <a href={target}>Open experiment predictors</a></p>;
  if (!refitRecovery && predictorId && !experimentId && registry.isPending) return <p role="status">Finding the predictor’s source experiment…</p>;
  return <><ErrorNotice error={registry.error} /><LocalPostDevelopment workspace={workspace} historical /></>;
}
