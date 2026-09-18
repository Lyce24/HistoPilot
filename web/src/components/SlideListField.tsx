import { useEffect, useRef, useState } from 'react';
import type { SlideListSource } from '../api/slideLists';
import ServerFolderPicker from './ServerFolderPicker';
import { ErrorNotice } from './ui';

/** The same optional selection for existing features and new extraction runs. */
export default function SlideListField({ value, onChange }: {
  value: SlideListSource | null | undefined;
  onChange: (value: SlideListSource | null) => void;
}) {
  const [error, setError] = useState<Error | null>(null);
  const read = useRef(0);
  useEffect(() => () => { read.current += 1; }, []);
  const mode = value?.contentBase64 !== undefined ? 'upload' : value ? 'server' : 'all';
  function change(next: SlideListSource | null) {
    read.current += 1;
    setError(null);
    onChange(next);
  }
  async function upload(file?: File) {
    if (!file) return;
    change({ filename: file.name, contentBase64: '' });
    const token = read.current;
    try {
      if (!/\.csv$/i.test(file.name)) throw new Error('Choose a CSV slide list.');
      if (!file.size || file.size > 1024 * 1024) throw new Error('Choose a nonempty CSV file up to 1 MB. Larger lists can be selected from the server.');
      const bytes = new Uint8Array(await file.arrayBuffer());
      if (token !== read.current) return;
      const parts: string[] = [];
      for (let offset = 0; offset < bytes.length; offset += 8192) {
        parts.push(String.fromCharCode(...bytes.subarray(offset, offset + 8192)));
      }
      onChange({ filename: file.name, contentBase64: btoa(parts.join('')) });
    } catch (reason) {
      if (token === read.current) setError(reason instanceof Error ? reason : new Error('The slide list could not be read.'));
    }
  }
  return <div className="science-field-group stack">
    <label className="label">Slide selection
      <select className="field" value={mode} onChange={(event) => change(event.target.value === 'all' ? null : event.target.value === 'upload' ? { filename: '', contentBase64: '' } : { path: '' })}>
        <option value="all">All eligible slides in the folder</option>
        <option value="server">Use a slide list on the server</option>
        <option value="upload">Upload a slide list CSV</option>
      </select>
    </label>
    {mode === 'server' ? <>
      <label className="label">Slide list CSV path
        <input className="field mono" value={value?.path ?? ''} placeholder="/path/to/slides.csv" onChange={(event) => change({ path: event.target.value })} />
      </label>
      <ServerFolderPicker selection="table" label="Browse slide lists" onSelect={(path) => change({ path })} />
    </> : null}
    {mode === 'upload' ? <label className="label">Upload slide list CSV
      <input className="field" type="file" accept=".csv,text/csv" onChange={(event) => void upload(event.target.files?.[0])} />
      <small>{value?.filename || 'Choose a CSV file up to 1 MB.'}</small>
    </label> : null}
    {mode !== 'all' ? <small>Use a <code>wsi</code> column with slide paths relative to the slide folder. Add <code>mpp</code> to specify source pixel size for every row. Existing features are matched by slide filename stem.</small> : null}
    <ErrorNotice error={error} />
  </div>;
}
