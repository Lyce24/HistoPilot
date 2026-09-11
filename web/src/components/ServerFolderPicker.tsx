import { useId, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { ErrorNotice, Icon } from './ui';
import './ServerFolderPicker.css';

interface Props {
  onSelect: (path: string) => void | Promise<void>;
  title?: string;
  label?: string;
  purpose?: 'data' | 'storage';
  selection?: 'folder' | 'table' | 'file';
  initialPath?: string;
}

export default function ServerFolderPicker({
  onSelect,
  title = 'Choose a server folder',
  label = 'Browse server',
  purpose = 'data',
  selection = 'folder',
  initialPath,
}: Props) {
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState<string | null>(null);
  const [pathInput, setPathInput] = useState('');
  const pathInputId = useId();
  const folderNameId = useId();
  const folderDestinationId = useId();
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [folderName, setFolderName] = useState('');
  const [notice, setNotice] = useState('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const navigate = (next: string | null) => {
    if (busy) return;
    setPath(next);
    setPathInput(next ?? '');
    setSelectedFile(null);
    setError(null);
    setNewFolderOpen(false);
    setFolderName('');
    setNotice('');
  };
  const [error, setError] = useState<Error | null>(null);
  const roots = useQuery({
    queryKey: ['filesystem', 'roots', purpose],
    queryFn: () => api.roots(purpose),
    enabled: open,
  });
  const directory = useQuery({
    queryKey: ['filesystem', 'directory', purpose, path],
    queryFn: () => api.directory(path!, purpose),
    enabled: open && path !== null,
  });
  async function select() {
    const selected = selection === 'folder' ? directory.data?.path : selectedFile;
    if (!selected || busy || newFolderOpen) return;
    setBusy(true);
    setError(null);
    try {
      await onSelect(selected);
      setOpen(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The folder could not be selected.'));
    } finally {
      setBusy(false);
    }
  }
  async function createFolder() {
    const parent = directory.data?.path;
    if (!parent || !folderName.trim() || busy || directory.isFetching || directory.isError || pathInput.trim() !== path) return;
    setBusy(true);
    setCreating(true);
    setError(null);
    setNotice('');
    try {
      const created = await api.createDirectory(parent, folderName.trim(), purpose);
      await client.invalidateQueries({
        queryKey: ['filesystem', 'directory'],
        predicate: (query) => query.queryKey[3] === parent || query.queryKey[3] === created.path,
      });
      setPath(created.path);
      setPathInput(created.path);
      setSelectedFile(null);
      setFolderName('');
      setNewFolderOpen(false);
      setNotice(`Created “${created.name}”. ${selection === 'folder' ? 'Use this folder to select it.' : 'You are now inside the new folder.'}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The folder could not be created.'));
    } finally {
      setCreating(false);
      setBusy(false);
    }
  }
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        if (!busy) {
          setOpen(next);
          setError(null);
          setNotice('');
          setNewFolderOpen(false);
          setFolderName('');
          if (next && initialPath) {
            setPath(initialPath);
            setPathInput(initialPath);
            setSelectedFile(null);
          }
        }
      }}
    >
      <Dialog.Trigger asChild>
        <button type="button" className="btn btn-secondary">
          <Icon name="folder" />
          {label}
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="folder-dialog">
          <div className="panel-header">
            <div>
              <Dialog.Title>{title}</Dialog.Title>
              <Dialog.Description>
                Folders on the computer running HistoPilot.{' '}
                {purpose === 'storage'
                  ? 'Browse a location or create a new folder.'
                  : 'Source data stays in place and is referenced read-only.'}
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button
                type="button"
                className="icon-button"
                aria-label="Close folder picker"
                disabled={busy}
              >
                <Icon name="close" />
              </button>
            </Dialog.Close>
          </div>
          <div className="folder-body">
            <div className="folder-roots">
              <button type="button" className="folder-entry" disabled={busy} onClick={() => navigate(null)}>
                <Icon name="system" /> Available locations
              </button>
              {roots.data?.roots.map((root) => (
                <button
                  type="button"
                  key={root.path}
                  className="folder-entry"
                  disabled={busy}
                  onClick={() => navigate(root.path)}
                >
                  <Icon name="folder" />
                  {root.name}
                </button>
              ))}
            </div>
            <div className="folder-list">
              <form
                className="folder-path-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  if (pathInput.trim() && !busy) navigate(pathInput.trim());
                }}
              >
                <label className="label" htmlFor={pathInputId}>
                  Folder path on the server
                </label>
                <div className="folder-path-controls">
                  <input
                    id={pathInputId}
                    className="field mono"
                    value={pathInput}
                    onChange={(event) => setPathInput(event.target.value)}
                    placeholder="Paste an absolute folder path"
                    maxLength={4096}
                    spellCheck={false}
                    disabled={busy}
                  />
                  <button
                    type="submit"
                    className="btn btn-secondary"
                    disabled={!pathInput.trim() || busy}
                  >
                    Go
                  </button>
                </div>
              </form>
              <ErrorNotice error={roots.error ?? directory.error} />
              {notice ? <p className="folder-create-notice" role="status"><Icon name="check" size={16} />{notice}</p> : null}
              {roots.isPending || (path !== null && directory.isPending) ? (
                <p role="status" className="muted">
                  Reading folders…
                </p>
              ) : null}
              {roots.data?.roots.length === 0 ? (
                <div className="callout folder-location-help" role="status">
                  <strong>
                    {purpose === 'storage'
                      ? 'No experiment storage locations are configured.'
                      : 'No data locations are configured.'}
                  </strong>
                  <p>
                    In the terminal running HistoPilot, stop the service with Ctrl+C. Add{' '}
                    <code>--data-root /path/to/data</code> to your startup command, keeping your
                    other options, then restart and refresh this page. Replace{' '}
                    <code>/path/to/data</code> with a folder on the server.
                  </p>
                </div>
              ) : null}
              {roots.data && roots.data.roots.length > 0 ? (
                <p className="muted">
                  Click an available location, or enter a folder path inside one and press Go.
                </p>
              ) : null}
              {roots.data && roots.data.roots.length > 0 ? (
                <details className="folder-location-help muted">
                  <summary>Folder outside these locations?</summary>
                  <p>
                    Restart HistoPilot with an additional <code>--data-root /path/to/data</code>{' '}
                    option for that folder, then refresh this page. Keep your existing startup
                    options. You can repeat <code>--data-root</code> for multiple locations.
                  </p>
                </details>
              ) : null}
              {path === null
                ? roots.data?.roots.map((root) => (
                    <button
                      type="button"
                      key={root.path}
                      className="folder-entry"
                      disabled={busy}
                      onClick={() => navigate(root.path)}
                    >
                      <Icon name="folder" />
                      <span>
                        {root.name}
                        <small className="mono">{root.path}</small>
                      </span>
                      <Icon name="chevron" />
                    </button>
                  ))
                : null}
              {directory.data && path !== null ? (
                <>
                  <div className="folder-directory-actions">
                    <button
                      type="button"
                      className="folder-entry"
                      disabled={!directory.data.parent || busy}
                      onClick={() => navigate(directory.data!.parent)}
                    >
                      <Icon name="arrowUp" /> Parent directory
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-small"
                      disabled={busy || directory.isFetching || directory.isError || pathInput.trim() !== path}
                      aria-expanded={newFolderOpen}
                      onClick={() => { setNewFolderOpen(true); setError(null); setNotice(''); }}
                    >
                      <Icon name="plus" size={16} /> New folder
                    </button>
                  </div>
                  {newFolderOpen ? (
                    <form
                      className="folder-create-form"
                      onSubmit={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        void createFolder();
                      }}
                    >
                      <label className="label" htmlFor={folderNameId}>New folder name</label>
                      <input
                        id={folderNameId}
                        className="field"
                        value={folderName}
                        placeholder="e.g. blca"
                        onChange={(event) => { setFolderName(event.target.value); setError(null); }}
                        disabled={busy}
                        required
                        maxLength={255}
                        autoFocus
                        autoCapitalize="off"
                        spellCheck={false}
                        aria-describedby={folderDestinationId}
                      />
                      <p id={folderDestinationId} className="folder-create-destination">
                        <span>{folderName.trim() ? 'Create in this location' : 'Parent directory'}</span>
                        <span className="mono">{directory.data.path.replace(/\/$/, '') || '/'}{folderName.trim() ? `${directory.data.path === '/' ? '' : '/'}${folderName.trim()}` : ''}</span>
                      </p>
                      <div className="inline-actions">
                        <button type="submit" className="btn btn-primary btn-small" disabled={busy || !folderName.trim() || directory.isFetching || directory.isError || pathInput.trim() !== path}>
                          <Icon name="folder" size={16} /> {creating ? 'Creating…' : 'Create folder'}
                        </button>
                        <button type="button" className="btn btn-secondary btn-small" disabled={busy} onClick={() => { setNewFolderOpen(false); setFolderName(''); setError(null); }}>
                          Cancel new folder
                        </button>
                      </div>
                    </form>
                  ) : null}
                  {directory.data.entries
                    .filter(
                      (entry) =>
                        ['directory', 'dir'].includes(entry.kind) ||
                        (selection === 'file' && entry.kind === 'file') ||
                        (selection === 'table' && /\.(csv|xlsx)$/i.test(entry.name)),
                    )
                    .map((entry) => (
                      <button
                        type="button"
                        className={`folder-entry ${selectedFile === entry.path ? 'row-selected' : ''}`}
                        aria-pressed={
                          ['directory', 'dir'].includes(entry.kind)
                            ? undefined
                            : selectedFile === entry.path
                        }
                        key={entry.path}
                        disabled={busy}
                        onClick={() =>
                          ['directory', 'dir'].includes(entry.kind)
                            ? navigate(entry.path)
                            : setSelectedFile(entry.path)
                        }
                      >
                        <Icon name="folder" />
                        <span>{entry.name}</span>
                        <Icon name="chevron" />
                      </button>
                    ))}
                  {!directory.data.entries.some((entry) =>
                    ['directory', 'dir'].includes(entry.kind),
                  ) ? (
                    <p className="muted">No subfolders in this location.</p>
                  ) : null}
                  {directory.data.truncated ? (
                    <p className="muted">
                      This folder listing is limited. You can also enter an exact path in the form.
                    </p>
                  ) : null}
                </>
              ) : null}
            </div>
          </div>
          <div className="folder-footer">
            <ErrorNotice error={error} />
            <p className="muted">
              {selectedFile
                ? selectedFile
                : purpose === 'storage'
                  ? 'The selected path is on this computer.'
                  : 'Choosing a folder does not upload or import its contents.'}
            </p>
            <div className="inline-actions">
              <Dialog.Close asChild>
                <button type="button" className="btn btn-secondary" disabled={busy}>
                  Cancel
                </button>
              </Dialog.Close>
              <button
                type="button"
                className="btn btn-primary"
                disabled={
                  !path ||
                  pathInput.trim() !== path ||
                  (selection !== 'folder' && !selectedFile) ||
                  !directory.data ||
                  directory.isError ||
                  directory.isFetching ||
                  newFolderOpen ||
                  busy
                }
                onClick={() => void select()}
              >
                {busy && !creating ? 'Selecting…' : selection === 'folder' ? 'Use this folder' : 'Use this file'}
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
