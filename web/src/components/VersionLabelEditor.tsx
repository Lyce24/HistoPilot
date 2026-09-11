import { useEffect, useId, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { scientific } from '../api/scientific';
import type { VersionLabel, VersionLabelResource } from '../api/scientific';
import { ErrorNotice, Icon } from './ui';
import './VersionLabelEditor.css';

interface Props {
  project: string;
  resourceType: 'dataset' | 'configuration';
  resource: VersionLabelResource;
  defaultOpen?: boolean;
  tagLabel?: string;
  description?: string;
}

export default function VersionLabelEditor(props: Props) {
  return <VersionLabelForm key={`${props.project}:${props.resourceType}:${props.resource.id}`} {...props} />;
}

function updateCachedLabel(value: unknown, id: string, label: VersionLabel): unknown {
  if (!value || typeof value !== 'object') return value;
  const record = value as Record<string, unknown>;
  if (record.id === id) return { ...record, versionLabel: label };
  for (const key of ['datasets', 'configurations']) {
    if (Array.isArray(record[key])) {
      return {
        ...record,
        [key]: record[key].map((item: unknown) => {
          if (item && typeof item === 'object' && (item as { id?: string }).id === id)
            return { ...item, versionLabel: label };
          return item;
        }),
      };
    }
  }
  return value;
}

function VersionLabelForm({
  project, resourceType, resource, defaultOpen = false, tagLabel = 'Version tag', description,
}: Props) {
  const client = useQueryClient();
  const tagId = useId();
  const noteId = useId();
  const descriptionId = useId();
  const [open, setOpen] = useState(defaultOpen);
  const [base, setBase] = useState<VersionLabel | null>(resource.versionLabel ?? null);
  const [latestFetched, setLatestFetched] = useState<VersionLabel | null>(null);
  const [tag, setTag] = useState(resource.versionLabel?.tag ?? '');
  const [note, setNote] = useState(resource.versionLabel?.note ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const incoming = resource.versionLabel ?? null;
  const latest = (latestFetched?.revision ?? 0) > (incoming?.revision ?? 0) ? latestFetched : incoming;
  const baseRevision = base?.revision ?? 0;
  const dirty = tag !== (base?.tag ?? '') || note !== (base?.note ?? '');
  const changedElsewhere = (latest?.revision ?? 0) > baseRevision;
  const conflict = dirty && changedElsewhere;

  useEffect(() => {
    if (defaultOpen) setOpen(true);
  }, [defaultOpen]);

  useEffect(() => {
    if (!dirty && !busy && changedElsewhere) {
      setBase(latest);
      setTag(latest?.tag ?? '');
      setNote(latest?.note ?? '');
    }
  }, [dirty, busy, changedElsewhere, latest]);

  function useSavedValues() {
    const saved = changedElsewhere ? latest : base;
    setBase(saved);
    setTag(saved?.tag ?? '');
    setNote(saved?.note ?? '');
    setError(null);
    setMessage('');
  }

  async function save(expectedRevision: number) {
    if (busy) return;
    setBusy(true);
    setError(null);
    setMessage('');
    try {
      const saved = await scientific.setVersionLabel(project, resourceType, resource.id, {
        tag: tag.trim(),
        note,
        expectedRevision,
      });
      setBase(saved);
      setLatestFetched(saved);
      setTag(saved.tag);
      setNote(saved.note);
      client.setQueriesData({ queryKey: ['scientific', project] }, (cached: unknown) =>
        updateCachedLabel(cached, resource.id, saved));
      await Promise.all([
        client.invalidateQueries({ queryKey: ['scientific', project] }),
        client.invalidateQueries({ queryKey: ['workspace', project] }),
      ]);
      setMessage(saved.tag || saved.note ? 'Version tag and note saved.' : 'Version tag and note cleared.');
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The version label could not be saved.'));
      if (reason instanceof ApiError && reason.status === 409) {
        try {
          const current = resourceType === 'dataset'
            ? await scientific.dataset(project, resource.id)
            : await scientific.configuration(project, resource.id);
          setLatestFetched(current.versionLabel ?? null);
        } catch {
          // Preserve the local draft and original save error if the latest version cannot be read.
        }
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="version-label-editor" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span><Icon name="provenance" size={17} /><strong>{base?.tag || 'Add a personal version tag'}</strong></span>
        <small>{dirty ? 'Unsaved changes' : base?.tag || base?.note ? 'Edit tag & note' : 'Optional · editable later'}</small>
      </summary>
      <div className="version-label-body">
        <p id={descriptionId} className="version-label-description">
          {description ?? 'Give this saved version a recognizable tag and a note about what changed. You can edit these later; the saved data and settings stay fixed.'}
        </p>
        <form onSubmit={(event) => { event.preventDefault(); event.stopPropagation(); if (!conflict) void save(baseRevision); }}>
          <fieldset className="version-label-fields" disabled={busy}>
            <label className="label" htmlFor={tagId}>
              {tagLabel}
              <input
                id={tagId}
                className="field"
                value={tag}
                maxLength={80}
                placeholder="e.g. baseline-v1"
                autoComplete="off"
                aria-describedby={descriptionId}
                onChange={(event) => { setTag(event.target.value); setError(null); setMessage(''); }}
              />
              <small>Up to 80 characters. Tags are unique among versions of the same kind in this project.</small>
            </label>
            <label className="label" htmlFor={noteId}>
              Commit note (optional)
              <textarea
                id={noteId}
                className="field version-label-note"
                value={note}
                rows={3}
                maxLength={2000}
                placeholder="What changed in this version, or why you want to keep it."
                onChange={(event) => { setNote(event.target.value); setError(null); setMessage(''); }}
              />
              <small>{note.length.toLocaleString()} / 2,000 characters</small>
            </label>
            {conflict ? (
              <div className="version-label-conflict" role="alert">
                <strong>This version label changed elsewhere. Your edits are preserved.</strong>
                <p>Latest saved tag: <b>{latest?.tag || 'No tag'}</b></p>
                {latest?.note ? <p className="version-label-latest-note">{latest.note}</p> : <p>No saved note.</p>}
                <div className="inline-actions">
                  <button type="button" className="btn btn-secondary btn-small" onClick={useSavedValues}>Use latest saved values</button>
                  <button type="button" className="btn btn-primary btn-small" onClick={() => void save(latest?.revision ?? 0)}>Save my changes instead</button>
                </div>
              </div>
            ) : null}
            <ErrorNotice error={error} />
            {message ? <p className="version-label-saved" role="status"><Icon name="check" size={16} />{message}</p> : null}
            <div className="version-label-actions">
              <p>Clear both fields and save to remove the tag and note.</p>
              <div className="inline-actions">
                {dirty && !conflict ? <button type="button" className="btn btn-secondary btn-small" onClick={useSavedValues}>Discard edits</button> : null}
                <button type="submit" className="btn btn-primary btn-small" disabled={!dirty || conflict}>
                  <Icon name="check" size={15} />{busy ? 'Saving…' : 'Save tag & note'}
                </button>
              </div>
            </div>
          </fieldset>
        </form>
      </div>
    </details>
  );
}
