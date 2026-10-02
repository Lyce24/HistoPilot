import { request } from './client';
import type { ModelExperiment } from './experiments';

/** How much an objective matters to the study, in a protocol's words. */
export type ObjectiveRank = 'primary' | 'secondary' | 'exploratory';
/** What an experiment does for its objective; the library lists them in this order. */
export type ExperimentRole = 'primary' | 'comparison' | 'variant' | 'generalization' | 'control' | 'tuning';
export const objectiveRanks: readonly ObjectiveRank[] = ['primary', 'secondary', 'exploratory'];
export const experimentRoles: readonly ExperimentRole[] = ['primary', 'comparison', 'variant', 'generalization', 'control', 'tuning'];
export const objectiveRankLabel: Record<ObjectiveRank, string> = {
  primary: 'Primary objective', secondary: 'Secondary objective', exploratory: 'Exploratory objective',
};
export const experimentRoleLabel: Record<ExperimentRole, string> = {
  primary: 'Primary', comparison: 'Comparison', variant: 'Variant', generalization: 'Generalization', control: 'Control', tuning: 'Tuning',
};
export const experimentRoleHelp: Record<ExperimentRole, string> = {
  primary: 'Answers the objective: the model you report and apply.',
  comparison: 'Same data, another method: model, encoder, clinical inputs, fusion or pooling.',
  variant: 'Same method, another label definition or population.',
  generalization: 'Trained on some cohorts and tested on held-out ones, such as leave one cohort out.',
  control: 'A negative or matched control.',
  tuning: 'Pilots, hyperparameter rounds and earlier versions. Folded away in the library.',
};

/** A scientific question the project's experiments are filed under. */
export interface Objective {
  id: string; name: string; rank: ObjectiveRank; description: string;
  revision: number; createdAt: string; updatedAt: string;
}
/** Where an experiment sits: its objective, its role there, and its series and member. */
export interface ExperimentFiling {
  objectiveId: string | null; role: ExperimentRole | null; series: string | null; seriesMember: string | null;
}
export const noFiling: ExperimentFiling = { objectiveId: null, role: null, series: null, seriesMember: null };
/** The four filing fields of an experiment as the service lists it (absent from older services). */
export const filingOf = (item: Partial<ExperimentFiling>): ExperimentFiling => ({
  objectiveId: item.objectiveId ?? null, role: item.role ?? null, series: item.series ?? null, seriesMember: item.seriesMember ?? null,
});
export const sameFiling = (left: ExperimentFiling, right: ExperimentFiling) =>
  left.objectiveId === right.objectiveId && left.role === right.role && left.series === right.series && left.seriesMember === right.seriesMember;

const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/objectives`;
export const objectives = {
  /** The objectives, and every filed experiment's filing by experiment ID (`filings`), which
   * runs, batches, predictors and studies use to group themselves by objective. */
  list: (project: string) => request<{ revision: number; items: Objective[]; filings?: Record<string, ExperimentFiling> }>(prefix(project)),
  create: (project: string, input: { name: string; rank: ObjectiveRank; description: string; operationId: string }) =>
    request<Objective>(prefix(project), { method: 'POST', body: JSON.stringify(input) }),
  update: (project: string, id: string, input: { expectedRevision: number; name?: string; rank?: ObjectiveRank; description?: string }) =>
    request<Objective>(`${prefix(project)}/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(input) }),
  remove: (project: string, id: string, input: { expectedRevision: number; operationId: string }) =>
    request<{ objectiveId: string; name: string; unfiled: number }>(`${prefix(project)}/${encodeURIComponent(id)}/remove`, { method: 'POST', body: JSON.stringify(input) }),
  /** Replace an experiment's filing, if it is still `expected`; the experiment itself does not change. */
  file: (project: string, experimentId: string, filing: ExperimentFiling, expected: ExperimentFiling) =>
    request<ModelExperiment>(`/projects/${encodeURIComponent(project)}/model-experiments/${encodeURIComponent(experimentId)}/objective`, { method: 'PUT', body: JSON.stringify({ ...filing, expected }) }),
};
