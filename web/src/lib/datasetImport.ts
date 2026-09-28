import type { AttributeMapping, Inspection, TableSource } from '../api/scientific';
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

export type SlidePathStyle = 'absolute' | 'relative' | 'mixed' | 'unknown';

/**
 * How a slide file column names its files, judged from the inspected sample values. The service
 * reads a path starting with / or a backslash as a full path and any other path inside the slide folder.
 */
export function slidePathStyle(inspection: Pick<Inspection, 'rows' | 'columnSummaries'> | null | undefined, column: string | undefined): SlidePathStyle {
  if (!inspection || !column) return 'unknown';
  const values = [...(inspection.columnSummaries?.[column]?.examples ?? []), ...inspection.rows.map((row) => row[column])]
    .map((value) => value?.trim() ?? '').filter(Boolean);
  if (!values.length) return 'unknown';
  const absolute = values.filter((value) => /^[\\/]/.test(value)).length;
  return absolute === values.length ? 'absolute' : absolute ? 'mixed' : 'relative';
}

/** The note under the slide file column: a slide folder matters only for relative paths. */
export function slideFileColumnNote(column: string | undefined, style: SlidePathStyle): string {
  if (!column) return 'Optional. A column naming each slide file: a full path such as /data/rih/SL-145.svs, or a path inside the slide folder such as rih/SL-145.svs.';
  if (style === 'absolute') return 'This column holds full file paths, so no slide folder is needed. If you also choose one, every file must be inside it.';
  if (style === 'relative') return 'This column holds paths inside the slide folder, for example rih/SL-145.svs. Choose that folder above.';
  if (style === 'mixed') return 'This column mixes full paths and paths inside the slide folder. Full paths are used as they are; the others need the slide folder above.';
  return 'Full paths, such as /data/rih/SL-145.svs, are used as they are. Other paths are read inside the slide folder above.';
}

/** What to do next in the source step, or null when the slide source is complete. */
export function slideSourceNextNote(spec: { slideRoot?: string; includeMissingSlides?: boolean; slidePathColumn?: string }, style: SlidePathStyle): string {
  if (spec.slideRoot || spec.includeMissingSlides || (spec.slidePathColumn && style === 'absolute')) return 'Next, review the ID columns and attributes from your metadata.';
  const keep = 'or select Keep metadata rows above if you are starting from existing features.';
  if (!spec.slidePathColumn) return `Choose a slide folder, ${keep}`;
  return style === 'unknown' ? `Choose a slide folder unless the file column holds full paths, ${keep}` : `Choose the slide folder that the file paths are inside, ${keep}`;
}
