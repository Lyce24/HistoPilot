import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import PredictionTargetEditor from './PredictionTargetEditor';
import type { ProtocolSpec } from '../api/scientific';

const target: ProtocolSpec['target'] = {
  field: 'grade', task: 'binary_classification', unit: 'slide', classes: ['low', 'high'],
  labels: { '0': 'low', '1': 'high' }, positiveClass: 'high', missing: 'block', unmapped: 'block',
};

function render(changes: Partial<ProtocolSpec['target']> = {}, allowUnlabeled?: boolean) {
  return renderToStaticMarkup(<PredictionTargetEditor
    target={{ ...target, ...changes }} allowUnlabeled={allowUnlabeled} showFieldProfile={false}
    fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }}
    labelValues={{ isPending: false, error: null }} rawValues={['0', '1']} dataLabel="selected test records"
    onChooseTarget={() => undefined} onChange={() => undefined} />);
}

describe('prediction target label policies', () => {
  it('lets only testing targets keep slides unlabeled', () => {
    const option = 'Keep as unlabeled: predicted, not scored';
    expect(render({}, true).match(new RegExp(option, 'g'))).toHaveLength(2);
    expect(render()).not.toContain(option);
    // A saved policy stays visible where the option is not offered for new choices.
    expect(render({ missing: 'unlabeled' })).toContain(option);
  });
});
