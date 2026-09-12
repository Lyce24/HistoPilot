import { downloadArtifact, request } from './client';
import type { LifecycleState } from './lifecycle';
import type { Finding, ProtocolSpec } from './scientific';
export interface ClinicalSelection {
  evaluationId: string; name: string; unit: 'selected' | 'slide' | 'patient'; positiveClass?: string | null;
  threshold?: number | null; bins: number; thresholdMin: number; thresholdMax: number; thresholdSteps: number;
}
export interface OperatingPoint {
  threshold: number; tp: number; fp: number; tn: number; fn: number;
  sensitivity: number | null; specificity: number | null; ppv: number | null; npv: number | null;
  accuracy: number | null; balancedAccuracy: number | null; f1: number | null;
  positiveLikelihoodRatio: number | null; negativeLikelihoodRatio: number | null;
  predictedPositive: number; predictedNegative: number; netBenefit: number | null; treatAllNetBenefit: number | null;
  treatNoneNetBenefit: number; standardizedNetBenefit: number | null; netInterventionsAvoidedPer100: number | null;
  highRiskPer100: number | null; truePositivePer100: number | null; falsePositivePer100: number | null; missedPositivePer100: number | null;
}
export interface ClinicalReport {
  unit: 'slide' | 'patient'; frozenUnit: string; positiveClass: string; classOrder: string[]; multiclass: boolean;
  decisionThreshold: number; frozenDecisionThreshold: number; thresholdSource: 'frozen_evaluation' | 'descriptive_override';
  counts: { total: number; labeled: number; unlabeled: number; positive: number; negative: number; patients: number | null; slides: number; missingPatientIds: number; fallbackPatientIds?: number };
  metrics: { prevalence: number; meanPredictedRisk?: number; observedExpectedRatio?: number | null; calibrationGap?: number; multiclassLogLoss?: number | null; brierScore: number; brierReference: number; brierSkillScore: number | null; logLoss: number; multiclassBrierScore: number | null; rocAuc: number | null; averagePrecision: number | null; ece: number; mce: number };
  uncertainty?: { method: 'wilson_95' | 'unavailable'; reason: string; operatingPoint: Partial<Record<'sensitivity' | 'specificity' | 'ppv' | 'npv', { lower: number; upper: number } | null>> };
  curveSampling?: { distinctScores: number; maximumPoints: number; returnedPoints: number; downsampled: boolean; summaryStatistics: 'exact'; csv: 'same_points_as_report' };
  operatingPoint: OperatingPoint; operatingCurve: OperatingPoint[];
  rocCurve: { threshold: number | null; falsePositiveRate: number | null; truePositiveRate: number | null }[];
  precisionRecallCurve: { threshold: number | null; recall: number | null; precision: number | null }[];
  calibration: { lower: number; upper: number; count: number; meanPredicted: number | null; observedFraction: number | null; absoluteError: number | null }[];
  warnings: string[]; definitions: Record<string, string>; sources: { title: string; url: string }[];
}
export interface ClinicalAnalysis {
  id: string; createdAt: string; contentHash: string; lifecycleState?: LifecycleState;
  manifest: { kind: 'clinical-analysis'; name: string; datasetId: string; experimentId: string; predictorId: string; evaluationId: string; selection: ClinicalSelection; target: ProtocolSpec['target']; report: ClinicalReport; [key: string]: unknown };
}
export interface ClinicalPreview { canSave: boolean; previewHash: string | null; findings: Finding[]; manifest: ClinicalAnalysis['manifest'] | null }
export type ClinicalArtifact = 'report.json' | 'operating-curves.csv' | 'calibration.csv' | 'roc.csv' | 'precision-recall.csv';
const base = (project: string) => `/projects/${encodeURIComponent(project)}/clinical-analyses`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const clinicalAnalyses = {
  list: (project: string) => request<{ items: ClinicalAnalysis[] }>(base(project)),
  get: (project: string, id: string) => request<ClinicalAnalysis>(`${base(project)}/${encodeURIComponent(id)}`),
  preview: (project: string, selection: ClinicalSelection) => request<ClinicalPreview>(`${base(project)}/preview`, post(selection)),
  save: (project: string, selection: ClinicalSelection, previewHash: string, operationId: string) => request<ClinicalAnalysis>(base(project), post({ ...selection, previewHash, operationId })),
  download: (project: string, id: string, filename: ClinicalArtifact) => downloadArtifact(`${base(project)}/${encodeURIComponent(id)}/artifacts/${filename}`, filename),
};
