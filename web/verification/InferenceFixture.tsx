import { useRef, useState } from 'react';
import type { EvaluationInference } from '../src/api/evaluation';
import type { ProtocolSpec } from '../src/api/scientific';
import { EvaluationInferenceFields, newEvaluationSpec } from '../src/pages/LocalEvaluationSetup';
import { reportEditorValidity } from '../src/components/NumericField';

const target: ProtocolSpec['target'] = {
  field: 'outcome', task: 'binary_classification', unit: 'patient',
  classes: ['negative', 'positive'], labels: { negative: 'negative', positive: 'positive' },
  positiveClass: 'positive', missing: 'block', unmapped: 'block',
};

/** Offline browser fixture: production inference controls and validation; no API calls. */
export function InferenceFixture() {
  const [value, setValue] = useState<EvaluationInference>(() => newEvaluationSpec().inference);
  const [saved, setSaved] = useState<EvaluationInference | null>(null);
  const [saveCount, setSaveCount] = useState(0);
  const [revision, setRevision] = useState(0);
  const editor = useRef<HTMLFieldSetElement>(null);
  return <section aria-label="Inference validation fixture">
    <h1>Inference validation</h1>
    <p>Uses the production inference inputs and editor validation without a running service.</p>
    <fieldset key={revision} ref={editor} className="evaluation-fields">
      <legend>Inference test inputs</legend>
      <EvaluationInferenceFields value={value} target={target} onChange={(update) => setValue((current) => ({ ...current, ...update }))} />
    </fieldset>
    <div className="inline-actions">
      <button type="button" className="btn btn-primary" onClick={() => {
        if (!reportEditorValidity(editor.current)) return;
        setSaved({ ...value });
        setSaveCount((count) => count + 1);
      }}>Save inference settings</button>
      <button type="button" className="btn btn-secondary" onClick={() => { setValue({ ...newEvaluationSpec().inference, patientAggregation: 'max' }); setRevision((current) => current + 1); }}>Load legacy maximum aggregation</button>
      <button type="button" className="btn btn-secondary" onClick={() => { setValue(newEvaluationSpec().inference); setSaved(null); setSaveCount(0); setRevision((current) => current + 1); }}>Reset inference fixture</button>
    </div>
    <p role="status" data-testid="inference-save-count">Accepted saves: {saveCount}</p>
    <pre aria-label="Accepted inference values">{JSON.stringify(value, null, 2)}</pre>
    <pre aria-label="Last saved inference values">{saved ? JSON.stringify(saved, null, 2) : 'No settings saved'}</pre>
  </section>;
}
