/** CSV slide selection. Paths in its wsi column are relative to the chosen slide folder. */
export interface SlideListSource {
  path?: string;
  filename?: string;
  contentBase64?: string;
}

export function slideListReady(source: SlideListSource | null | undefined): boolean {
  return !source || Boolean(source.path?.trim() || source.filename && source.contentBase64);
}
