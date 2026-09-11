import { describe, expect, it, vi } from 'vitest';
import type { AttributeMapping, TableSource } from '../api/scientific';
import { canReuseImportMapping, inspectedAttributes, readTableUpload, UploadReadState } from './datasetImport';

function deferredFile(name: string) {
  let resolve!: (bytes: ArrayBuffer) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<ArrayBuffer>((yes, no) => { resolve = yes; reject = no; });
  return { file: { name, size: 10, arrayBuffer: () => promise }, resolve, reject };
}
const bytes = (value: string) => new TextEncoder().encode(value).buffer;
function state() {
  let source: TableSource = { filename: '', contentBase64: '' };
  const reads = new UploadReadState(source);
  const onChange = vi.fn((next: TableSource) => { source = next; reads.synchronize(next); });
  const onError = vi.fn();
  return { reads, onChange, onError, source: () => source };
}

describe('metadata upload intent', () => {
  it('keeps the latest file when earlier arrayBuffer reads complete later', async () => {
    const current = state();
    const first = deferredFile('first.csv');
    const second = deferredFile('second.csv');
    const pendingFirst = readTableUpload(first.file, current.reads, current.onChange, current.onError);
    const pendingSecond = readTableUpload(second.file, current.reads, current.onChange, current.onError);
    second.resolve(bytes('new source'));
    await pendingSecond;
    first.resolve(bytes('old source'));
    await pendingFirst;
    expect(current.source()).toEqual({ filename: 'second.csv', contentBase64: btoa('new source') });
    expect(current.onChange).toHaveBeenCalledTimes(3); // Two placeholders; only the latest result.
    expect(current.onError).not.toHaveBeenCalled();
  });

  it.each([
    { path: '/server/current.csv' },
    { filename: 'first.csv', contentBase64: '' },
  ])('does not replace a changed source or newly loaded draft: %j', async (replacement) => {
    const current = state();
    const first = deferredFile('first.csv');
    const pending = readTableUpload(first.file, current.reads, current.onChange, current.onError);
    current.onChange(replacement);
    first.resolve(bytes('outdated'));
    await pending;
    expect(current.source()).toEqual(replacement);
    expect(current.onChange).toHaveBeenCalledTimes(2);
  });

  it('ignores late failures after a new selection or an unmount', async () => {
    const current = state();
    const first = deferredFile('old.csv');
    const pending = readTableUpload(first.file, current.reads, current.onChange, current.onError);
    current.reads.cancel();
    first.reject(new Error('Old read failed'));
    await pending;
    expect(current.onError).not.toHaveBeenCalled();
  });

  it('surfaces a current read failure and rejects oversized uploads without reading them', async () => {
    const current = state();
    const failed = deferredFile('failed.csv');
    const pending = readTableUpload(failed.file, current.reads, current.onChange, current.onError);
    failed.reject(new Error('Read failed'));
    await pending;
    expect(current.onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'This metadata file could not be read.' }));
    const arrayBuffer = vi.fn();
    await readTableUpload({ name: 'large.xlsx', size: 256 * 1024 + 1, arrayBuffer }, current.reads, current.onChange, current.onError);
    expect(arrayBuffer).not.toHaveBeenCalled();
    expect(current.onError).toHaveBeenLastCalledWith(expect.objectContaining({ message: expect.stringContaining('256 KB') }));
    expect(current.source()).toEqual({ filename: 'large.xlsx', contentBase64: '' });
  });

  it('preserves binary workbook bytes across encoding chunks', async () => {
    const current = state();
    const data = Uint8Array.from({ length: 19000 }, (_, index) => index % 256);
    await readTableUpload({ name: 'workbook.xlsx', size: data.length, arrayBuffer: async () => data.buffer }, current.reads, current.onChange, current.onError);
    expect(Uint8Array.from(atob(current.source().contentBase64!), (character) => character.charCodeAt(0))).toEqual(data);
    expect(current.onError).not.toHaveBeenCalled();
  });
});

describe('metadata reinspection mapping', () => {
  const headers = ['Slide_ID', 'Patient_ID', 'Grade', 'Unused'];
  const kept: AttributeMapping[] = [{ key: 'Reviewed grade', sourceColumn: 'Grade', type: 'ordered_categorical', owner: 'patient', categories: ['low', 'high'], missingValues: ['unknown'] }];

  it('preserves removed attributes, renamed fields, ownership, typing and missingness on reread', () => {
    expect(inspectedAttributes(headers, kept, ['Slide_ID', 'Patient_ID'], true, 'slide')).toEqual(kept);
    expect(inspectedAttributes(headers, [], ['Slide_ID', 'Patient_ID'], true, 'slide')).toEqual([]);
    expect(inspectedAttributes(headers, kept, ['Slide_ID', 'Patient_ID'], true, 'patient')).toEqual(kept);
  });

  it('keeps selected fields that disappeared so validation can report them instead of silently dropping data', () => {
    expect(inspectedAttributes(['Slide_ID'], kept, ['Slide_ID'], true, 'slide')).toEqual(kept);
  });

  it('creates fresh suggestions for a new source without inheriting old recoding constraints', () => {
    expect(inspectedAttributes(headers, kept, ['Slide_ID', 'Patient_ID'], false, 'slide')).toEqual([
      { key: 'Grade', sourceColumn: 'Grade', owner: 'slide', type: 'text' },
      { key: 'Unused', sourceColumn: 'Unused', owner: 'slide', type: 'text' },
    ]);
  });

  it('recognizes current rereads and saved draft sources but distinguishes different files and sheets', () => {
    const source = { path: '/metadata.xlsx', sheet: 'Data' };
    expect(canReuseImportMapping(source, source, undefined)).toBe(true);
    expect(canReuseImportMapping(source, undefined, { ...source })).toBe(true);
    expect(canReuseImportMapping(source, undefined, { ...source, sheet: 'Other' })).toBe(false);
    expect(canReuseImportMapping({ path: '/other.csv' }, source, source)).toBe(false);
    // A new unsaved import starts with a fresh source object even for the same path.
    expect(canReuseImportMapping({ ...source }, source, undefined)).toBe(false);
  });
});
