import { useState } from "react";
import { api } from "../api/client";
import type {
  ImportResult,
  JobOut,
  ReferenceEntryIn,
  ReferenceEntryOut,
  ReferenceLabel,
} from "../api/types";
import { Badge } from "../components/Badge";
import { BarChart, type BarDatum } from "../components/BarChart";
import { ErrorBanner } from "../components/ErrorBanner";
import { ProgressBar } from "../components/ProgressBar";
import { StatCard } from "../components/StatCard";
import { useAction, useApi } from "../hooks/useApi";
import { DASH, fmtDate, fmtNumber, fmtText, humanize } from "../utils/format";

export function ReferenceDatasetPage() {
  const [categoryFilter, setCategoryFilter] = useState("");
  const [labelFilter, setLabelFilter] = useState("");
  const [qInput, setQInput] = useState("");
  const [q, setQ] = useState("");
  const [rebuildJobId, setRebuildJobId] = useState<string | null>(null);

  // Entry Add/Edit modal state
  const [editingEntry, setEditingEntry] = useState<ReferenceEntryOut | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [formConcept, setFormConcept] = useState("");
  const [formCategory, setFormCategory] = useState("financial_statements");
  const [formLabel, setFormLabel] = useState<ReferenceLabel>("positive");
  const [formDefinition, setFormDefinition] = useState("");
  const [formExamples, setFormExamples] = useState("");
  const [formSource, setFormSource] = useState("custom");
  const [formLicense, setFormLicense] = useState("CC-BY-4.0");

  // Import dataset state
  const [importOpen, setImportOpen] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importName, setImportName] = useState("");
  const [importVersion, setImportVersion] = useState("1.1.0");
  const [importSource, setImportSource] = useState("");
  const [importLicense, setImportLicense] = useState("CC-BY-4.0");
  const [importAttribution, setImportAttribution] = useState("");
  const [importUrl, setImportUrl] = useState("");
  const [importRebuild, setImportRebuild] = useState(true);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);

  const statsState = useApi((signal) => api.referenceStats(signal), []);
  const entriesState = useApi(
    (signal) =>
      api.listEntries(
        { category: categoryFilter, label: labelFilter, q, limit: 100, offset: 0 },
        signal,
      ),
    [categoryFilter, labelFilter, q],
  );

  const jobState = useApi<JobOut>(
    (signal) => api.getJob(rebuildJobId!, signal),
    [rebuildJobId],
    {
      enabled: Boolean(rebuildJobId),
      pollMs: 1500,
      shouldPoll: (j) => {
        if (!j) return false;
        if (j.status === "succeeded" || j.status === "failed") {
          statsState.reload();
          return false;
        }
        return true;
      },
    },
  );

  const action = useAction();
  const stats = statsState.data;
  const entries = entriesState.data?.items ?? [];

  const categoryChart: BarDatum[] = stats
    ? Object.entries(stats.category_distribution ?? {})
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => ({ label: k, value: v }))
    : [];

  const openAddModal = () => {
    setEditingEntry(null);
    setFormConcept("");
    setFormCategory(stats?.categories?.[0] ?? "financial_statements");
    setFormLabel("positive");
    setFormDefinition("");
    setFormExamples("");
    setFormSource("custom");
    setFormLicense("CC-BY-4.0");
    setModalOpen(true);
  };

  const openEditModal = (entry: ReferenceEntryOut) => {
    setEditingEntry(entry);
    setFormConcept(entry.concept);
    setFormCategory(entry.category);
    setFormLabel(entry.label === "negative" ? "negative" : "positive");
    setFormDefinition(entry.definition ?? "");
    setFormExamples((entry.examples ?? []).join("\n"));
    setFormSource(entry.source ?? "custom");
    setFormLicense(entry.license ?? "CC-BY-4.0");
    setModalOpen(true);
  };

  const handleSaveEntry = async (e: React.FormEvent) => {
    e.preventDefault();
    const examples = formExamples
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    const body: ReferenceEntryIn = {
      concept: formConcept.trim(),
      category: formCategory.trim(),
      label: formLabel,
      definition: formDefinition.trim() || null,
      examples,
      source: formSource.trim() || "custom",
      license: formLicense.trim() || "CC-BY-4.0",
    };
    const res = await action.run(async () => {
      if (editingEntry) {
        return api.updateEntry(editingEntry.id, body);
      }
      return api.createEntry(body);
    });
    if (res) {
      setModalOpen(false);
      statsState.reload();
      entriesState.reload();
    }
  };

  const handleDeactivate = async (id: string) => {
    await action.run(async () => {
      await api.deactivateEntry(id);
      statsState.reload();
      entriesState.reload();
    });
  };

  const handleRebuild = async () => {
    const job = await action.run(() => api.rebuildEmbeddings());
    if (job) {
      setRebuildJobId(job.id);
      statsState.reload();
    }
  };

  const handleImport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!importFile) return;
    const res = await action.run(() =>
      api.importDataset({
        file: importFile,
        name: importName.trim(),
        version: importVersion.trim(),
        source: importSource.trim(),
        license: importLicense.trim(),
        attribution: importAttribution.trim() || undefined,
        source_url: importUrl.trim() || undefined,
        rebuild_index: importRebuild,
      }),
    );
    if (res) {
      setImportResult(res);
      if (res.index_job) setRebuildJobId(res.index_job.id);
      statsState.reload();
      entriesState.reload();
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Reference Dataset & Vector Index</h1>
          <p className="page-subtitle">
            Manage financial concepts, negative look-alike examples, dataset versions, and semantic vector embeddings
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-secondary"
            disabled={action.busy}
            onClick={() => void handleRebuild()}
          >
            Rebuild Embeddings
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setImportOpen((v) => !v)}
          >
            {importOpen ? "Close Import" : "Import JSON/CSV"}
          </button>
          <button type="button" className="btn btn-primary" onClick={openAddModal}>
            + Add Entry
          </button>
        </div>
      </header>

      <ErrorBanner error={action.error ?? statsState.error ?? entriesState.error} />

      {stats?.index_stale && (
        <div className="callout callout-warn">
          <strong>Vector index is marked stale:</strong> Reference entries have changed since the index was last built.
          Click <strong>Rebuild Embeddings</strong> to re-index active entries.
        </div>
      )}

      {jobState.data && (
        <section className="card">
          <div className="card-body">
            <ProgressBar
              value={jobState.data.progress}
              label={`Embedding index job (${jobState.data.status}${
                jobState.data.stage ? ` · ${jobState.data.stage}` : ""
              })`}
              tone={
                jobState.data.status === "succeeded"
                  ? "success"
                  : jobState.data.status === "failed"
                    ? "error"
                    : "default"
              }
            />
          </div>
        </section>
      )}

      <section className="stat-grid">
        <StatCard
          label="Active Reference Entries"
          value={fmtNumber(stats?.active_entries)}
          hint={`of ${fmtNumber(stats?.total_entries)} total entries`}
        />
        <StatCard
          label="Positive Financial Concepts"
          value={fmtNumber(stats?.positive_entries)}
          tone="relevant"
        />
        <StatCard
          label="Negative Look-Alikes"
          value={fmtNumber(stats?.negative_entries)}
          hint="False-positive guards"
          tone="irrelevant"
        />
        <StatCard
          label="Indexed Vectors"
          value={fmtNumber(stats?.index_vectors)}
          hint={stats ? `${fmtNumber(stats.total_examples)} example phrases` : DASH}
        />
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Category Distribution</h2>
          </div>
          <div className="card-body">
            <BarChart items={categoryChart} emptyText={DASH} />
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Embedding & Vector Index Details</h2>
          </div>
          <div className="card-body">
            <dl className="kv-grid">
              <div>
                <dt>Embedding Model</dt>
                <dd className="mono">{fmtText(stats?.embedding_model)}</dd>
              </div>
              <div>
                <dt>Index Backend</dt>
                <dd className="mono">{fmtText(stats?.index_backend)}</dd>
              </div>
              <div>
                <dt>Index Version</dt>
                <dd className="mono">{fmtText(stats?.index_version)}</dd>
              </div>
              <div>
                <dt>Index Built At</dt>
                <dd>{fmtDate(stats?.index_built_at)}</dd>
              </div>
              <div>
                <dt>Index Stale</dt>
                <dd>
                  <Badge tone={stats?.index_stale ? "uncertain" : "relevant"} size="sm">
                    {stats ? (stats.index_stale ? "Stale" : "Up to date") : DASH}
                  </Badge>
                </dd>
              </div>
            </dl>

            <h3 className="section-sub" style={{ marginTop: "1rem" }}>
              Dataset Versions
            </h3>
            {stats?.dataset_versions && stats.dataset_versions.length > 0 ? (
              <div className="table-scroll">
                <table className="data-table data-table-compact">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Name</th>
                      <th>Source</th>
                      <th>License</th>
                      <th>Entries</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.dataset_versions.map((dv) => (
                      <tr key={dv.id}>
                        <td className="mono">{dv.version}</td>
                        <td>{dv.name}</td>
                        <td>{dv.source}</td>
                        <td className="mono">{dv.license}</td>
                        <td>{fmtNumber(dv.entry_count)}</td>
                        <td>
                          <Badge tone={dv.is_active ? "relevant" : "neutral"} size="sm">
                            {dv.is_active ? "Active" : "Archived"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted-sm">{DASH}</p>
            )}
          </div>
        </section>
      </div>

      {/* Dataset Import Panel */}
      {importOpen && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Import Reference Dataset (JSON / CSV)</h2>
          </div>
          <div className="card-body">
            <form className="stack" onSubmit={(e) => void handleImport(e)}>
              <div className="form-row">
                <div className="form-field">
                  <label htmlFor="imp-file">Dataset File (.json or .csv)</label>
                  <input
                    id="imp-file"
                    type="file"
                    accept=".json,.csv"
                    required
                    onChange={(e) => setImportFile(e.target.files?.[0] ?? null)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="imp-name">Dataset Name</label>
                  <input
                    id="imp-name"
                    type="text"
                    className="input"
                    required
                    placeholder="e.g. Custom Financial Taxonomy"
                    value={importName}
                    onChange={(e) => setImportName(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="imp-ver">Version</label>
                  <input
                    id="imp-ver"
                    type="text"
                    className="input"
                    required
                    value={importVersion}
                    onChange={(e) => setImportVersion(e.target.value)}
                  />
                </div>
              </div>
              <div className="form-row">
                <div className="form-field">
                  <label htmlFor="imp-src">Source</label>
                  <input
                    id="imp-src"
                    type="text"
                    className="input"
                    required
                    placeholder="e.g. Internal Accounting Standards"
                    value={importSource}
                    onChange={(e) => setImportSource(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="imp-lic">License</label>
                  <input
                    id="imp-lic"
                    type="text"
                    className="input"
                    required
                    value={importLicense}
                    onChange={(e) => setImportLicense(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="imp-attr">Attribution (optional)</label>
                  <input
                    id="imp-attr"
                    type="text"
                    className="input"
                    value={importAttribution}
                    onChange={(e) => setImportAttribution(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="imp-url">Source URL (optional)</label>
                  <input
                    id="imp-url"
                    type="url"
                    className="input"
                    value={importUrl}
                    onChange={(e) => setImportUrl(e.target.value)}
                  />
                </div>
              </div>
              <div className="form-actions">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={importRebuild}
                    onChange={(e) => setImportRebuild(e.target.checked)}
                  />
                  Rebuild vector embeddings immediately after import
                </label>
                <button type="submit" className="btn btn-primary" disabled={action.busy || !importFile}>
                  {action.busy ? "Importing…" : "Import Dataset"}
                </button>
              </div>
            </form>
            {importResult && (
              <div className="callout callout-info" style={{ marginTop: "0.75rem" }}>
                Imported <strong>{importResult.imported}</strong> new, updated{" "}
                <strong>{importResult.updated}</strong>, skipped <strong>{importResult.skipped}</strong>{" "}
                into version <code>{importResult.dataset_version.version}</code>.
              </div>
            )}
          </div>
        </section>
      )}

      {/* Add / Edit Entry Modal */}
      {modalOpen && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">
              {editingEntry ? `Edit Entry: ${editingEntry.concept}` : "Add Reference Entry"}
            </h2>
            <button type="button" className="btn btn-sm btn-ghost" onClick={() => setModalOpen(false)}>
              Close
            </button>
          </div>
          <div className="card-body">
            <form className="stack" onSubmit={(e) => void handleSaveEntry(e)}>
              <div className="form-row">
                <div className="form-field">
                  <label htmlFor="entry-concept">Concept Name</label>
                  <input
                    id="entry-concept"
                    type="text"
                    className="input"
                    required
                    minLength={2}
                    value={formConcept}
                    onChange={(e) => setFormConcept(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="entry-category">Category</label>
                  <input
                    id="entry-category"
                    type="text"
                    className="input"
                    required
                    value={formCategory}
                    onChange={(e) => setFormCategory(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="entry-label">Polarity Label</label>
                  <select
                    id="entry-label"
                    className="select"
                    value={formLabel}
                    onChange={(e) => setFormLabel(e.target.value as ReferenceLabel)}
                  >
                    <option value="positive">positive (financial)</option>
                    <option value="negative">negative (non-financial look-alike)</option>
                  </select>
                </div>
              </div>
              <div className="form-field">
                <label htmlFor="entry-def">Definition</label>
                <textarea
                  id="entry-def"
                  className="textarea"
                  rows={2}
                  value={formDefinition}
                  onChange={(e) => setFormDefinition(e.target.value)}
                />
              </div>
              <div className="form-field">
                <label htmlFor="entry-ex">Example Snippets (one per line)</label>
                <textarea
                  id="entry-ex"
                  className="textarea"
                  rows={4}
                  placeholder="Net accounts receivable as of December 31 totaled $1,420,000."
                  value={formExamples}
                  onChange={(e) => setFormExamples(e.target.value)}
                />
              </div>
              <div className="form-row">
                <div className="form-field">
                  <label htmlFor="entry-src">Source</label>
                  <input
                    id="entry-src"
                    type="text"
                    className="input"
                    value={formSource}
                    onChange={(e) => setFormSource(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label htmlFor="entry-lic">License</label>
                  <input
                    id="entry-lic"
                    type="text"
                    className="input"
                    value={formLicense}
                    onChange={(e) => setFormLicense(e.target.value)}
                  />
                </div>
              </div>
              <div className="form-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setModalOpen(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={action.busy}>
                  {action.busy ? "Saving…" : editingEntry ? "Update Entry" : "Create Entry"}
                </button>
              </div>
            </form>
          </div>
        </section>
      )}

      {/* Searchable & Filterable Entries Table */}
      <section className="card">
        <div className="card-header">
          <h2 className="card-title">
            Reference Entries ({fmtNumber(entriesState.data?.total ?? entries.length)})
          </h2>
        </div>
        <div className="card-body stack">
          <div className="filter-bar">
            <form
              className="filter-search"
              onSubmit={(e) => {
                e.preventDefault();
                setQ(qInput.trim());
              }}
            >
              <input
                type="search"
                className="input"
                placeholder="Search concept, definition, or example…"
                value={qInput}
                onChange={(e) => setQInput(e.target.value)}
              />
              <button type="submit" className="btn btn-secondary">
                Search
              </button>
            </form>

            <div className="filter-group">
              <label htmlFor="ref-cat" className="filter-label">
                Category
              </label>
              <select
                id="ref-cat"
                className="select"
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
              >
                <option value="">All categories</option>
                {(stats?.categories ?? []).map((c) => (
                  <option key={c} value={c}>
                    {humanize(c)}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label htmlFor="ref-lbl" className="filter-label">
                Label
              </label>
              <select
                id="ref-lbl"
                className="select"
                value={labelFilter}
                onChange={(e) => setLabelFilter(e.target.value)}
              >
                <option value="">All labels</option>
                <option value="positive">Positive</option>
                <option value="negative">Negative</option>
              </select>
            </div>
          </div>

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Concept</th>
                  <th>Category</th>
                  <th>Label</th>
                  <th>Definition & Examples</th>
                  <th>Source</th>
                  <th>Active</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={entry.id}>
                    <td>
                      <strong>{entry.concept}</strong>
                      <div className="mono muted-sm">{entry.id}</div>
                    </td>
                    <td>{humanize(entry.category)}</td>
                    <td>
                      <Badge value={entry.label} size="sm" />
                    </td>
                    <td>
                      {entry.definition && <div>{entry.definition}</div>}
                      {entry.examples && entry.examples.length > 0 && (
                        <div className="muted-sm">
                          e.g. “{entry.examples[0]}”
                          {entry.examples.length > 1 ? ` (+${entry.examples.length - 1} more)` : ""}
                        </div>
                      )}
                    </td>
                    <td className="mono muted-sm">{entry.source}</td>
                    <td>
                      <Badge tone={entry.is_active ? "relevant" : "neutral"} size="sm">
                        {entry.is_active ? "Yes" : "No"}
                      </Badge>
                    </td>
                    <td>
                      <div className="row-actions">
                        <button
                          type="button"
                          className="btn btn-sm btn-ghost"
                          onClick={() => openEditModal(entry)}
                        >
                          Edit
                        </button>
                        {entry.is_active && (
                          <button
                            type="button"
                            className="btn btn-sm btn-ghost"
                            disabled={action.busy}
                            onClick={() => void handleDeactivate(entry.id)}
                          >
                            Deactivate
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
