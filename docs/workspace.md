# Workspace and artifact layout

The application workspace owns the recent-project registry and explicit synthetic demo. Each local experiment owns its setup and scientific state in the folder chosen on the start page. Original WSIs and feature arrays remain external read-only sources.

The default central workspace is `~/.histopilot/workspace`. Set `[storage].workspace` in `~/.histopilot/config.toml` or use `histopilot serve --workspace /path/to/workspace` to change it.

## Persisted state

```text
<central service workspace>/
  histopilot.db               # Recent projects, demo and model registry
  histopilot.db-wal / -shm    # SQLite-managed sidecars when present
  service.lock               # One launcher per central workspace

<chosen experiment folder>/
  histopilot-project.json     # Authoritative identity, setup and source references
  histopilot-state.sqlite     # Drafts, datasets, configurations, version labels, publication receipts
  .histopilot-write.lock      # Retained cross-process operation lock
  .staging/                  # Private, incomplete publication artifacts
  datasets/<dataset-id>/
    manifest.json            # Published snapshot and per-artifact hashes
    records.json             # Canonical Slide_ID / Patient_ID / attributes
    dictionary.json          # Attribute ownership, type and source mapping
    inventory.json           # Complete bounded slide discovery
    exclusions.json          # Explicit excluded source rows
    provenance.json          # Mapping, source hashes and verification scope
    sources/main.csv|xlsx     # Exact metadata bytes (optional patient source alongside)
```

The central database remains schema version 1. The scientific database is schema version 4 with its own project identity. Opening a valid schema-1, schema-2, or schema-3 scientific database atomically adds missing configuration, retry-receipt, version-label, and publication-label tables; dataset artifact format stays version 1. New local projects initialize both descriptor and scientific store. Opening an older folder containing only a valid version-1 descriptor adds the scientific store without rewriting setup. Unsupported future or inconsistent scientific stores are rejected, not reset. Directory existence alone does not establish a completed dataset.

The descriptor owns setup fields; the project database owns scientific drafts/revisions and publication state. Dataset files own immutable published bytes checked against recorded fingerprints. No scientific object depends on the original central registry to reopen. Absolute external-source references remain source metadata; generated artifact locations are relative to their experiment folder.

## Creation, reopening, and moving

**Start a new experiment** requires a name and an exact empty/new folder whose parent exists within the central workspace or a configured data root. Data/slide/feature paths and primary configuration remain optional. Creation publishes the descriptor before initializing scientific storage. If scientific initialization fails after setup publication, the valid descriptor remains so that **Load an existing experiment** can retry; creation must not overwrite the folder.

**Load an existing experiment** accepts a registered recent project or its folder. Reopening initializes compatible scientific storage and reconciles interrupted publication. A fresh central registry can register the same folder without reconstructing drafts or datasets. Moving the complete folder preserves generated artifact references; reopen its new permitted path. An old registry retains the unavailable former path until the moved folder is opened. Two existing folders with the same ID cannot both be registered in one registry.

The UI URL remains `?experiment=<id>#overview`. Dataset import, records, feature attachment and target/split editors read their scientific state from this folder. The synthetic demo stays isolated and non-executable.

## Draft consistency and snapshot publication

Import and experiment drafts carry server-owned project identity, revision, status and timestamps. Updates replace name/payload in one transaction and require the caller's expected revision. Stale updates fail, and frozen import drafts cannot be edited. Draft payloads are unvalidated intent, never proof of scientific readiness.

One advisory writer lock in the experiment folder serializes operations across threads/processes and different service registries. The lock file remains in place after release; a killed process releases its kernel lock. Each operation uses a fresh database connection, avoiding persistent handles tied to a former folder location. Setup source/configuration updates reread the descriptor under the same folder lock, but remain separate from scientific transactions.

Internal snapshot publication records its intent in a durable journal, writes staged artifacts and a checksum manifest, synchronizes files/directories, atomically publishes the dataset directory, and commits visibility plus draft status. Only published, verified snapshots appear through dataset reads. An identical publication-operation retry returns the same result; reusing that operation ID with different inputs fails.

Reopen/retry recovery checks journaled artifacts. A complete valid staged or published snapshot can finish publication. An incomplete operation is recorded as interrupted and leaves the last saved draft intact; incomplete files never become a dataset. Existing frozen snapshots have no overwrite endpoint. Artifact reads verify hashes and reject unexpected changes.

The CSV/XLSX importer validates complete bounded inventories, identifier mapping, source revisions and canonical scientific records before publication. Frozen storage does not imply valid patient grouping, labels, feature contents or execution readiness.

## Filesystem and backup boundary

This implementation requires POSIX file locking, directory synchronization, atomic same-filesystem rename, and SQLite WAL support. It fails explicitly when required operations are unsupported. Automated verification uses local temporary POSIX folders; it does not certify every mounted drive or network filesystem. Native Windows scientific storage is not supported by these durability primitives yet; a functioning browser path picker alone cannot establish storage compatibility.

Scientific SQLite connections use full synchronization. File and directory synchronization reduce the risk of losing acknowledged changes after interruption; actual power-loss durability still depends on the filesystem and hardware honoring those operations. No tests here simulate a physical power failure.

For a consistent manual backup, stop all writers using the experiment folder, then copy the entire folder including any SQLite sidecars. A database file copied alone during writes is not a guaranteed snapshot. Copy the central workspace separately if recent history/demo state is wanted; it is not required to reopen a local scientific project. Original external data needs its own backup.

## External sources and future artifacts

Source browsing remains confined to configured roots (`purpose=source`). Storage browsing (`purpose=storage`) includes the central workspace and configured roots. Registering a directory saves a read-only `data`, `slides` or `features` reference; it does not scan, copy or validate its contents.

The importer persists the [standard identifier/attribute mapping](data-stage-schema-design.md), source snapshots, complete inventory and exclusion records. Patient links can remain explicitly unresolved at import; patient-grouped execution requires verified mapping. Adding identity or source information creates a new dataset version rather than rewriting frozen inputs.

Parquet tables, preprocessing, run checkpoints and results can be added under the chosen folder through explicit versioned adapters. Large WSI/feature arrays do not belong inside draft JSON or SQLite rows. The current storage publication helper is intended for bounded metadata artifacts, not bulk image transfer or extraction.

Frozen target/split protocols and feature header bindings are bounded checksummed JSON in the scientific SQLite database. One FULL-sync transaction commits a configuration, its idempotency receipt and the protocol draft transition. Exact per-seed/per-fold patient/slide memberships are stored; reopening does not rerun a random split. Feature arrays stay external and require source verification before execution.

Personal version tags and commit notes are mutable presentation metadata in `version_labels`, keyed by dataset/configuration ID. API reads add `versionLabel` after verifying the immutable document. All public dataset, protocol, and feature freeze requests require `versionLabel: {tag, note?}` with a nonempty tag. The tag participates in the publication request identity, never in scientific manifests or content hashes. Dataset publication journals persist the original label intent separately in `publication_labels` so recovery commits the label, dataset index, and draft freeze together; preparing publications reserve their requested tags. Configuration labels and publication receipts commit in one transaction. Same-operation retries validate the original naming intent and return the current label without reverting later edits. A new operation cannot silently relabel identical content that already has a different tag or note.

Label updates require the current label revision, enforce case-insensitive uniqueness per scientific kind, and retain a revision when cleared. `PUT datasets/{id}/label` and `PUT configurations/{id}/label` accept `tag`, `note`, and `expectedRevision`; unlabelled records start at revision zero. Current cohort tags name frozen protocols. Future experiment/cohort publishers should reuse this metadata boundary and the shared freeze dialog without substituting mutable tags for canonical references. The session endpoint advertises `scientificCapabilities.versionLabels` and `scientificCapabilities.taggedFreeze`; the browser checks these before sending label or freeze writes to an older running server.
