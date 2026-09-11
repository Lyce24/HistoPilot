import type { AttributeMapping, TableSource } from '../api/scientific';
import { sameJSON } from './json';

/** A source object also identifies its editing context, even when paths are equal. */
export class UploadReadState {
  private generation = 0;
  constructor(private source: TableSource) {}
  synchronize(source: TableSource) {
    if (source !== this.source) {
      this.source = source;
      this.cancel();
    }
  }
  begin(source: TableSource): number {
    this.source = source;
    return ++this.generation;
  }
  current(ticket: number) { return this.generation === ticket; }
  accept(ticket: number, source: TableSource): boolean {
    if (!this.current(ticket)) return false;
    this.source = source;
    return true;
  }
  cancel() { this.generation += 1; }
}

export async function readTableUpload(
  file: Pick<File, 'name' | 'size' | 'arrayBuffer'>,
  reads: UploadReadState,
  onChange: (source: TableSource) => void,
  onError: (error: Error) => void,
): Promise<void> {
  const placeholder = { filename: file.name, contentBase64: '' };
  const ticket = reads.begin(placeholder);
  onChange(placeholder);
  if (file.size > 256 * 1024) {
    onError(new Error('Browser metadata uploads are limited to 256 KB. Use a server file path for larger tables (up to 16 MB).'));
    return;
  }
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (!reads.current(ticket)) return;
    let text = '';
    for (let offset = 0; offset < bytes.length; offset += 8192)
      text += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
    const complete = { filename: file.name, contentBase64: btoa(text) };
    if (reads.accept(ticket, complete)) onChange(complete);
  } catch {
    if (reads.current(ticket)) onError(new Error('This metadata file could not be read.'));
  }
}

/** Reinspection never adds back removed fields or drops an explicitly selected one. */
export function inspectedAttributes(
  headers: string[],
  existing: AttributeMapping[],
  identityColumns: (string | undefined)[],
  preserve: boolean,
  owner: AttributeMapping['owner'],
): AttributeMapping[] {
  if (preserve) return existing;
  return headers.filter((column) => !identityColumns.includes(column)).map((column) => ({
    key: column, sourceColumn: column, owner, type: 'text',
  }));
}

export function canReuseImportMapping(
  source: TableSource,
  inspectedSource: TableSource | undefined,
  savedSource: TableSource | undefined,
): boolean {
  return source === inspectedSource || (savedSource !== undefined && sameJSON(source, savedSource));
}
