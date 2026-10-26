import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Badge } from "../components/Badge";
import { BarChart, type BarDatum } from "../components/BarChart";
import { ErrorBanner } from "../components/ErrorBanner";
import { StatCard } from "../components/StatCard";
import { useApi } from "../hooks/useApi";
import {
  asRecord,
  DASH,
  fmtBytes,
  fmtDate,
  fmtDecimal,
  fmtMs,
  fmtNumber,
  fmtPercent,
  fmtText,
  fmtUsd,
  get,
  isNum,
} from "../utils/format";

export function DashboardPage() {
  const statsState = useApi(
    (signal) => api.stats(signal),
    [],
    {
      pollMs: 4000,
      shouldPoll: (d) => Boolean(d && d.in_progress > 0),
    },
  );
  const healthState = useApi((signal) => api.health(signal), []);
  const docsState = useApi(
    (signal) => api.listDocuments({ limit: 8, offset: 0 }, signal),
    [statsState.data?.processed_documents, statsState.data?.total_documents],
  );

  const stats = statsState.data;
  const health = healthState.data;
  const docs = docsState.data?.items ?? [];

  const filterRate =
    stats && stats.processed_documents > 0
      ? stats.not_financial_filtered / stats.processed_documents
      : null;

  const cacheRate =
    stats && stats.embedding_calls + stats.embedding_cache_hits > 0
      ? stats.embedding_cache_hits / (stats.embedding_calls + stats.embedding_cache_hits)
      : null;

  const chunkChartItems: BarDatum[] = stats
    ? [
        {
          label: "RELEVANT",
          value: stats.chunk_decisions?.RELEVANT ?? 0,
          tone: "relevant",
        },
        {
          label: "UNCERTAIN",
          value: stats.chunk_decisions?.UNCERTAIN ?? 0,
          tone: "uncertain",
        },
        {
          label: "IRRELEVANT",
          value: stats.chunk_decisions?.IRRELEVANT ?? 0,
          tone: "irrelevant",
        },
      ]
    : [];

  const byTypeItems: BarDatum[] = stats
    ? Object.entries(stats.by_type ?? {}).map(([k, v]) => ({
        label: k,
        value: v,
        tone: "default",
      }))
    : [];

  const evalSummary = asRecord(stats?.evaluation);
  const testChunkF1 = get(evalSummary, "chunk_metrics", "test", "f1") ?? get(evalSummary, "test_f1");
  const testPrAuc = get(evalSummary, "chunk_metrics", "test", "pr_auc") ?? get(evalSummary, "pr_auc");
  const docAcc = get(evalSummary, "document_metrics", "accuracy") ?? get(evalSummary, "document_accuracy");
  const irrelFilterRate =
    get(evalSummary, "document_metrics", "irrelevant_filtering_rate") ??
    get(evalSummary, "irrelevant_filtering_rate");

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">System Overview</h1>
          <p className="page-subtitle">
            Document ingestion, financial relevance filtering, and classification telemetry
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              statsState.reload();
              healthState.reload();
              docsState.reload();
            }}
          >
            Refresh
          </button>
          <Link to="/upload" className="btn btn-primary">
            Upload Documents
          </Link>
        </div>
      </header>

      <ErrorBanner error={statsState.error ?? healthState.error} onRetry={statsState.reload} />

      <section className="stat-grid" aria-label="Document statistics">
        <StatCard
          label="Total Documents"
          value={fmtNumber(stats?.total_documents)}
          hint={stats ? `${fmtNumber(stats.processed_documents)} processed` : DASH}
          to="/documents"
        />
        <StatCard
          label="Financially Relevant"
          value={fmtNumber(stats?.financial_relevant)}
          hint="Forwarded for extraction"
          tone="relevant"
          to="/documents?relevance=FINANCIAL_RELEVANT"
        />
        <StatCard
          label="Not Financial (Filtered)"
          value={fmtNumber(stats?.not_financial_filtered)}
          hint={isNum(filterRate) ? `${fmtPercent(filterRate)} filter rate` : DASH}
          tone="irrelevant"
          to="/documents?relevance=NOT_FINANCIAL"
        />
        <StatCard
          label="Mixed Content"
          value={fmtNumber(stats?.mixed_content)}
          hint="Contains relevant & non-financial sections"
          tone="mixed"
          to="/documents?relevance=MIXED_CONTENT"
        />
        <StatCard
          label="Needs Review"
          value={fmtNumber(stats?.needs_review)}
          hint="Ambiguous / human review suggested"
          tone="uncertain"
          to="/documents?relevance=NEEDS_REVIEW"
        />
        <StatCard
          label="Processing Failed"
          value={fmtNumber(stats?.processing_failed)}
          hint={stats && stats.in_progress > 0 ? `${stats.in_progress} currently in progress` : "Errors quarantined"}
          tone={stats && stats.processing_failed > 0 ? "failed" : "default"}
          to="/documents?relevance=PROCESSING_FAILED"
        />
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Chunk Decisions & Formats</h2>
            <span className="muted-sm">Total chunks: {fmtNumber(stats?.total_chunks)}</span>
          </div>
          <div className="card-body">
            <h3 className="section-sub">Chunk-Level Decisions</h3>
            <BarChart items={chunkChartItems} emptyText={ DASH } />
            <h3 className="section-sub" style={{ marginTop: "1rem" }}>
              Documents by Detected Format
            </h3>
            <BarChart items={byTypeItems} emptyText={DASH} />
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Performance & Runtime Health</h2>
            {health && (
              <Badge tone={health.status === "ok" ? "relevant" : "uncertain"}>{health.status.toUpperCase()}</Badge>
            )}
          </div>
          <div className="card-body">
            <dl className="kv-grid">
              <div>
                <dt>Avg Processing Latency</dt>
                <dd>{fmtMs(stats?.avg_processing_ms)}</dd>
              </div>
              <div>
                <dt>P95 Processing Latency</dt>
                <dd>{fmtMs(stats?.p95_processing_ms)}</dd>
              </div>
              <div>
                <dt>Embedding Calls / Cache Hits</dt>
                <dd>
                  {stats
                    ? `${fmtNumber(stats.embedding_calls)} / ${fmtNumber(stats.embedding_cache_hits)}${
                        isNum(cacheRate) ? ` (${fmtPercent(cacheRate)} hit)` : ""
                      }`
                    : DASH}
                </dd>
              </div>
              <div>
                <dt>LLM Tie-Breaker Calls / Cost</dt>
                <dd>
                  {stats ? `${fmtNumber(stats.llm_calls)} (${fmtUsd(stats.llm_cost_usd)})` : DASH}
                </dd>
              </div>
              <div>
                <dt>Reviewer Feedback Items</dt>
                <dd>{fmtNumber(stats?.feedback_count)}</dd>
              </div>
              <div>
                <dt>Embedding Model</dt>
                <dd className="mono">{fmtText(health?.embedding_model)}</dd>
              </div>
              <div>
                <dt>Vector Index</dt>
                <dd className="mono">
                  {health
                    ? `${fmtText(health.vector_backend)} (${fmtNumber(health.index_vectors)} vectors)`
                    : DASH}
                </dd>
              </div>
              <div>
                <dt>OCR Engine</dt>
                <dd>
                  {health ? `${health.ocr_provider} (${health.ocr_available ? "ready" : "unavailable"})` : DASH}
                </dd>
              </div>
            </dl>
          </div>
        </section>
      </div>

      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Latest Evaluation Benchmark</h2>
          <Link to="/evaluation" className="btn btn-sm btn-secondary">
            View Full Report
          </Link>
        </div>
        <div className="card-body">
          {evalSummary ? (
            <div className="stat-grid stat-grid-compact">
              <StatCard label="Test Chunk F1" value={fmtDecimal(testChunkF1, 3)} />
              <StatCard label="Test PR-AUC" value={fmtDecimal(testPrAuc, 3)} />
              <StatCard label="Document Accuracy" value={fmtPercent(docAcc, 1)} />
              <StatCard label="Irrelevant Filter Rate" value={fmtPercent(irrelFilterRate, 1)} />
            </div>
          ) : (
            <p className="muted">
              No evaluation benchmark has been recorded yet ({DASH}). Visit the{" "}
              <Link to="/evaluation">Evaluation</Link> page for details.
            </p>
          )}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Recent Documents</h2>
          <Link to="/documents" className="btn btn-sm btn-secondary">
            All Documents
          </Link>
        </div>
        <div className="card-body">
          {docs.length === 0 ? (
            <div className="empty-state">
              <p>No documents uploaded yet.</p>
              <Link to="/upload" className="btn btn-primary">
                Upload First Document
              </Link>
            </div>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Type</th>
                    <th>Size</th>
                    <th>Status</th>
                    <th>Relevance</th>
                    <th>Score</th>
                    <th>Latency</th>
                    <th>Uploaded</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((doc) => (
                    <tr key={doc.id}>
                      <td>
                        <Link to={`/documents/${doc.id}`} className="table-link">
                          {doc.filename}
                        </Link>
                      </td>
                      <td className="mono">{doc.detected_type}</td>
                      <td>{fmtBytes(doc.size_bytes)}</td>
                      <td>
                        <Badge value={doc.status} size="sm" />
                      </td>
                      <td>
                        <Badge value={doc.relevance_status} size="sm" />
                      </td>
                      <td className="mono">{fmtDecimal(doc.relevance_score, 2)}</td>
                      <td>{fmtMs(doc.processing_ms)}</td>
                      <td>{fmtDate(doc.uploaded_at)}</td>
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
