# P0.0 implementation handoff

> Historical P0.0 handoff. P0.1/P0.2 now adds an atomic scientific schema-2 migration, immutable protocol/feature configurations and the import/split UI; see [current implementation notes](p0-import-protocol-implementation.md).

P0.0 adds durable scientific state inside the folder selected when creating an experiment. Import/experiment drafts and internally published dataset snapshots can be reopened from another central service registry or after moving the complete experiment folder. The CSV/XLSX mapping and freeze UI remains the next P0.1 slice.

## What changed

| Component | Implemented behavior |
| --- | --- |
| `histopilot-project.json` | Retains ownership of experiment identity, setup and source references. Existing version-1 descriptors open without rewriting their bytes. Descriptor publication now also synchronizes directory entries. |
| `histopilot-state.sqlite` | New project-local SQLite database for scientific drafts, dataset metadata and publication operations. Uses WAL, full synchronization, project identity and an independent schema version. |
| `.histopilot-write.lock` | One retained operation-lock inode in the experiment folder, shared by separate processes and service registries. A competing writer receives a retryable conflict. Process death releases the kernel lock. |
| `.staging/` | Private staged publication data. Incomplete attempts remain non-visible; interrupted staging is retained for inspection. |
| `datasets/<dataset-id>/` | Immutable published manifest and declared metadata artifacts, addressed by a content hash. Generated paths are relative to the experiment folder. |

The central database continues to own recent-project discovery and the explicit demo. It contains no authoritative copy of local scientific drafts or frozen datasets. Opening through a fresh registry reads the folder's own state.

## Scientific drafts

Drafts have a server-generated ID, owning project, kind (`import` or `experiment`), name, JSON payload, revision, status and timestamps. A new draft starts at revision 1. An update supplies `expectedRevision`; an atomic transaction either applies the entire replacement and increments the revision or rejects it without changing the saved draft.

Draft payloads are unvalidated planning data. They cannot overwrite the server-owned envelope or declare scientific readiness. Successful internal dataset publication freezes its import draft and increments its revision. Frozen drafts have no edit path; new work uses a new draft.

Setup configuration still uses the existing descriptor API. Its writes now take the folder lock and reread current setup before replacement, but it does not acquire the scientific draft revision contract. This keeps existing UI settings compatible while making the stronger draft behavior explicit.

## Dataset publication and interruption recovery

The internal `ScientificStore.publish_dataset(...)` accepts a pinned import-draft revision, a caller-derived manifest, bounded metadata artifact bytes, and an operation ID. It is not exposed as a browser-authored dataset upload endpoint.

Publication proceeds through these durable boundaries:

1. Validate paths, document bounds, owning draft/revision and any parent dataset; compute the manifest/artifact fingerprints.
2. Commit publication intent in the project-local database.
3. Write staged files, synchronize their bytes/directories, and write the generated manifest after the declared artifacts.
4. Verify the complete staged inventory and hashes, then rename it into its final dataset directory on the same filesystem.
5. Synchronize the published files and namespace, then atomically record dataset visibility, frozen draft state and successful operation status.

Reopening or performing the next storage operation reconciles journal entries. Complete valid staged/published files finish publication idempotently. Incomplete or invalid staged data is marked interrupted and remains outside visible datasets; the previous saved draft remains available. Synchronization/database failures keep the operation recoverable. Unknown conflicting final directories are preserved for explicit repair, not overwritten.

An identical retry with the same operation ID returns the same published dataset. Reusing that ID for different inputs fails. Equivalent manifest/artifact content can reuse one immutable dataset within a project. A child version explicitly names an existing parent; changes never overwrite its artifacts.

Dataset metadata reads and artifact reads check stored content against declared hashes. Missing, changed, aliased, unsafe or undeclared artifacts fail. These checks establish storage integrity only; they do not validate patient identity, labels, split leakage or extraction provenance.

## Compatibility and limits

The first scientific schema initializes an empty store alongside an existing valid setup descriptor. Its version, application identity, owning project and table constraints are checked before use. Unknown/future/corrupt stores are rejected instead of reset. A missing or empty database alongside existing scientific directories/sidecars does not silently become a new empty experiment.

The initial materialized storage boundary is deliberately bounded: 1 MiB per JSON document, 64 MiB per artifact, 256 MiB total and at most 128 artifacts per publication. These limits cover metadata fixtures and the initial bladder importer; future larger table publication can add streaming. WSIs and extracted feature arrays remain external references.

Durability primitives currently require POSIX locking and directory synchronization plus a filesystem supporting SQLite WAL and atomic rename. Local temporary POSIX folders are tested. Native Windows, arbitrary network mounts and physical power-loss behavior are not certified. The D: clinical source directories were not modified or used as write-test locations.

Back up or move the complete experiment folder with all writers stopped, including any SQLite sidecars. The original central registry is optional for reopening; external data is backed up independently.

## Review surface

The [API documentation](api.md#project-local-scientific-storage) includes draft create/read/update commands, storage status, and dataset metadata reads. The existing workspace response also includes `scientificStorage`. The Dataset UI continues to use its current empty/import-not-implemented state until P0.1 connects it to actual parsed records.

Implementation files:

- [Scientific store](../histopilot/storage/scientific.py): database compatibility, revisions, publication, recovery and integrity checks.
- [Project lock](../histopilot/storage/project_lock.py): process locking, managed directory creation and synchronization.
- [Project workspace](../histopilot/application/project_workspace.py): create/open integration and descriptor coordination.
- [API](../histopilot/api/app.py) and [draft request schemas](../histopilot/schemas/scientific.py): authenticated project-scoped intent and read endpoints.

Verification covers old descriptor preservation, fresh-registry and moved-folder roundtrips, stale and concurrent edits, process-lock release after death, publication interruptions, retry identity, frozen immutability, source/path isolation, schema rejection and artifact corruption. Automated fixtures contain no clinical data.

Completed validation: **151 Python tests passed** in the full suite; the final focused scientific-storage run passed **59/59**. Ruff lint and formatting checks passed. A verification wheel was built and checked against the final changed Python files and packaged frontend/demo assets. HTTP tests ran outside the sandbox after a bare FastAPI TestClient reproduced the sandbox's event-loop startup hang; no application workaround was added.

The main choices available for refinement are the draft kinds/payload contract, metadata size limits, interruption-retention policy, and how the next Dataset UI displays persisted state. Scientific import validation and target/split preflight remain separate milestones.
