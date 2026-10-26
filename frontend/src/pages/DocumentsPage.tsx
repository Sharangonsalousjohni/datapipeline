import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { DOCUMENT_RELEVANCE_VALUES } from "../api/types";
import { Badge, labelFor } from "../components/Badge";
import { ErrorBanner } from "../components/ErrorBanner";
import { useApi } from "../hooks/useApi";
import { fmtBytes, fmtDate, fmtDecimal, fmtMs, truncateHash } from "../utils/format";

const STATUS_OPTIONS = ["queued", "processing", "completed", "failed"];

export function DocumentsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const relevance = searchParams.get("relevance") ?? "";
  const status = searchParams.get("status") ?? "";
  const [qInput, setQInput] = useState(searchParams.get("q") ?? "");
  const q = searchParams.get("q") ?? "";

  const docsState = useApi(
    (signal) => api.listDocuments({ relevance, status, q, limit: 100, offset: 0 }, signal),
    [relevance, status, q],
    {
      pollMs: 3500,
      shouldPoll: (d) =>
        Boolean(d?.items.some((doc) => doc.status === "queued" || doc.status === "processing")),
    },
  );

  const updateFilter = (key: string, val: string) => {
    const next = new URLSearchParams(searchParams);
    if (val) next.set(key, val);
    else next.delete(key);
    setSearchParams(next);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    updateFilter("q", qInput.trim());
  };

  const items = docsState.data?.items ?? [];
  const total = docsState.data?.total ?? 0;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Documents</h1>
          <p className="page-subtitle">
            Inspect ingested documents, relevance classifications, and processing telemetry ({total} total)
          </p>
        </div>
        <div className="header-actions">
          <button type="button" className="btn btn-secondary" onClick={docsState.reload}>
            Refresh
          </button>
          <Link to="/upload" className="btn btn-primary">
            Upload
          </Link>
        </div>
      </header>

      <section className="card">
        <div className="card-body filter-bar">
          <form onSubmit={handleSearch} className="filter-search">
            <input
              type="search"
              className="input"
              placeholder="Search filename or content hash…"
              value={qInput}
              onChange={(e) => setQInput(e.target.value)}
            />
            <button type="submit" className="btn btn-secondary">
              Search
            </button>
          </form>

          <div className="filter-group">
            <label htmlFor="filter-relevance" className="filter-label">
              Relevance
            </label>
            <select
              id="filter-relevance"
              className="select"
              value={relevance}
              onChange={(e) => updateFilter("relevance", e.target.value)}
            >
              <option value="">All relevance outcomes</option>
              {DOCUMENT_RELEVANCE_VALUES.map((r) => (
                <option key={r} value={r}>
                  {labelFor(r)}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="filter-status" className="filter-label">
              Status
            </label>
            <select
              id="filter-status"
              className="select"
              value={status}
              onChange={(e) => updateFilter("status", e.target.value)}
            >
              <option value="">All statuses</option>
              {STATUS_OPTIONS.map((s) => (
                <option key={s} value={s}>
                  {labelFor(s)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <ErrorBanner error={docsState.error} onRetry={docsState.reload} />

      <section className="card">
        <div className="card-body">
          {docsState.loading && !docsState.data ? (
            <p className="muted">Loading documents…</p>
          ) : items.length === 0 ? (
            <div className="empty-state">
              <p>No documents match the current filters.</p>
            </div>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Format</th>
                    <th>Size</th>
                    <th>SHA-256</th>
                    <th>Status</th>
                    <th>Relevance Decision</th>
                    <th>Score</th>
                    <th>Latency</th>
                    <th>Uploaded</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((doc) => (
                    <tr key={doc.id}>
                      <td>
                        <Link to={`/documents/${doc.id}`} className="table-link">
                          {doc.filename}
                        </Link>
                      </td>
                      <td className="mono">{doc.detected_type}</td>
                      <td>{fmtBytes(doc.size_bytes)}</td>
                      <td className="mono" title={doc.content_hash}>
                        {truncateHash(doc.content_hash, 10)}
                      </td>
                      <td>
                        <Badge value={doc.status} size="sm" />
                      </td>
                      <td>
                        <Badge value={doc.relevance_status} size="sm" />
                      </td>
                      <td className="mono">{fmtDecimal(doc.relevance_score, 2)}</td>
                      <td>{fmtMs(doc.processing_ms)}</td>
                      <td>{fmtDate(doc.uploaded_at)}</td>
                      <td>
                        <div className="row-actions">
                          <Link to={`/documents/${doc.id}`} className="btn btn-sm btn-ghost">
                            Details
                          </Link>
                          <Link to={`/documents/${doc.id}/chunks`} className="btn btn-sm btn-ghost">
                            Chunks
                          </Link>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
