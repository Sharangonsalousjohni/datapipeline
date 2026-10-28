import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, downloadJson } from "../api/client";
import type { DocumentDetail, RelevantChunksResponse } from "../api/types";
import { Badge } from "../components/Badge";
import { ErrorBanner } from "../components/ErrorBanner";
import { ProgressBar } from "../components/ProgressBar";
import { StatCard } from "../components/StatCard";
import { useAction, useApi } from "../hooks/useApi";
import {
  DASH,
  fmtBytes,
  fmtDate,
  fmtDecimal,
  fmtMs,
  fmtNumber,
  fmtText,
  fmtUsd,
  humanize,
} from "../utils/format";

const STAGES = ["queued", "extracting", "chunking", "classifying", "done"];

function stageIndex(doc: DocumentDetail): number {
  const job = doc.latest_job;
  if (doc.status === "completed" || job?.status === "succeeded") return 4;
  const stage = (job?.stage ?? doc.status ?? "").toLowerCase();
  if (stage.includes("classif") || stage.includes("embed")) return 3;
  if (stage.includes("chunk")) return 2;
  if (stage.includes("extract") || stage.includes("ocr") || stage === "processing") return 1;
  return 0;
}

export function DocumentDetailPage() {
  const { id = "" } = useParams<{ id: string }>();
  const [handoffPreview, setHandoffPreview] = useState<RelevantChunksResponse | null>(null);
  const [includeUncertain, setIncludeUncertain] = useState(true);
  const [showFields, setShowFields] = useState(false);

  const structuredState = useApi(
    (signal) => api.structuredExtraction(id, signal),
    [id],
  );

  const docState = useApi(
    (signal) => api.getDocument(id, signal),
    [id],
    {
      pollMs: 1800,
      shouldPoll: (d) => {
        if (!d) return false;
        const jStatus = d.latest_job?.status;
        return (
          d.status === "queued" ||
          d.status === "processing" ||
          jStatus === "queued" ||
          jStatus === "running"
        );
      },
    },
  );

  const action = useAction();
  const doc = docState.data;

  const handleReclassify = async (reextract: boolean) => {
    if (!doc) return;
    await action.run(async () => {
      await api.classify(doc.id, { force: true, reextract });
      docState.reload();
    });
  };

  const handleRetryJob = async () => {
    if (!doc?.latest_job) return;
    const jobId = doc.latest_job.id;
    await action.run(async () => {
      await api.retryJob(jobId);
      docState.reload();
    });
  };

  const handleDownloadReport = async () => {
    if (!doc) return;
    await action.run(async () => {
      const report = await api.report(doc.id);
      downloadJson(report, `${doc.filename}.relevance-report.json`);
    });
  };

  const handleDownloadOriginal = async () => {
    if (!doc) return;
    await action.run(async () => {
      await api.downloadOriginal(doc.id, doc.filename);
    });
  };

  const handlePreviewHandoff = async (incUncertain: boolean) => {
    if (!doc) return;
    setIncludeUncertain(incUncertain);
    await action.run(async () => {
      const res = await api.relevantChunks(doc.id, incUncertain);
      setHandoffPreview(res);
    });
  };

  if (docState.loading && !doc) {
    return (
      <div className="page">
        <p className="muted">Loading document details…</p>
      </div>
    );
  }

  if (docState.error && !doc) {
    return (
      <div className="page">
        <ErrorBanner error={docState.error} onRetry={docState.reload} />
        <Link to="/documents" className="btn btn-secondary">
          Back to Documents
        </Link>
      </div>
    );
  }

  if (!doc) return null;

  const job = doc.latest_job;
  const isRunning =
    doc.status === "queued" ||
    doc.status === "processing" ||
    job?.status === "queued" ||
    job?.status === "running";
  const currentStageIdx = stageIndex(doc);
  const qualityEntries = doc.extraction_quality ? Object.entries(doc.extraction_quality) : [];

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <div className="breadcrumb">
            <Link to="/documents">Documents</Link> / <span>{doc.filename}</span>
          </div>
          <h1 className="page-title">{doc.filename}</h1>
          <div className="badge-row">
            <Badge value={doc.status} />
            <Badge value={doc.relevance_status} size="lg" />
            <span className="mono muted-sm">
              {doc.detected_type.toUpperCase()} · {fmtBytes(doc.size_bytes)}
            </span>
          </div>
        </div>
        <div className="header-actions">
          <Link to={`/documents/${doc.id}/chunks`} className="btn btn-primary">
            Explore Chunks ({fmtNumber(doc.chunk_counts?.total)})
          </Link>
          <Link to={`/documents/${doc.id}/summary`} className="btn btn-primary">
            Financial Summary
          </Link>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={action.busy || isRunning}
            onClick={() => void handleReclassify(false)}
          >
            Reclassify
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={action.busy || isRunning}
            onClick={() => void handleReclassify(true)}
          >
            Re-extract & Reclassify
          </button>
          {job?.status === "failed" && (
            <button
              type="button"
              className="btn btn-secondary"
              disabled={action.busy}
              onClick={() => void handleRetryJob()}
            >
              Retry Job
            </button>
          )}
          <button
            type="button"
            className="btn btn-ghost"
            disabled={action.busy}
            onClick={() => void handleDownloadOriginal()}
          >
            Original File
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={action.busy}
            onClick={() => void handleDownloadReport()}
          >
            JSON Report
          </button>
        </div>
      </header>

      <ErrorBanner error={action.error ?? docState.error} onDismiss={() => action.setError(null)} />

      {doc.error_message && (
        <ErrorBanner error={doc.error_message} title="Document Processing Error" />
      )}

      {/* Stage Timeline & Job Progress */}
      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Pipeline Stage Timeline</h2>
          {job && (
            <span className="muted-sm mono">
              Job {job.id.slice(0, 8)} · {job.status} (attempt {job.attempts}/{job.max_attempts})
            </span>
          )}
        </div>
        <div className="card-body">
          <div className="stage-timeline" role="list" aria-label="Processing stages">
            {STAGES.map((st, idx) => {
              const state =
                doc.status === "failed" && idx === currentStageIdx
                  ? "failed"
                  : idx < currentStageIdx || doc.status === "completed"
                    ? "done"
                    : idx === currentStageIdx && isRunning
                      ? "active"
                      : "pending";
              return (
                <div key={st} role="listitem" className={`stage-step stage-${state}`}>
                  <div className="stage-dot" />
                  <div className="stage-name">{humanize(st)}</div>
                </div>
              );
            })}
          </div>
          {isRunning && (
            <div style={{ marginTop: "0.75rem" }}>
              <ProgressBar
                value={job?.progress ?? null}
                label={`Stage: ${humanize(job?.stage ?? doc.status)}`}
              />
            </div>
          )}
        </div>
      </section>

      {/* Score & Chunk Counts */}
      <section className="stat-grid">
        <StatCard
          label="Overall Relevance Score"
          value={fmtDecimal(doc.relevance_score, 2)}
          hint={doc.is_filtered ? "Filtered out (not forwarded)" : "Eligible for downstream extraction"}
          tone={
            doc.relevance_status === "FINANCIAL_RELEVANT"
              ? "relevant"
              : doc.relevance_status === "MIXED_CONTENT"
                ? "mixed"
                : doc.relevance_status === "NEEDS_REVIEW"
                  ? "uncertain"
                  : doc.relevance_status === "NOT_FINANCIAL"
                    ? "irrelevant"
                    : "default"
          }
        />
        <StatCard
          label="Relevant Chunks"
          value={fmtNumber(doc.chunk_counts?.relevant)}
          hint={`of ${fmtNumber(doc.chunk_counts?.total)} total chunks`}
          tone="relevant"
          to={`/documents/${doc.id}/chunks?decision=RELEVANT`}
        />
        <StatCard
          label="Uncertain Chunks"
          value={fmtNumber(doc.chunk_counts?.uncertain)}
          hint="Candidate review items"
          tone="uncertain"
          to={`/documents/${doc.id}/chunks?decision=UNCERTAIN`}
        />
        <StatCard
          label="Irrelevant Chunks"
          value={fmtNumber(doc.chunk_counts?.irrelevant)}
          hint="Suppressed from handoff"
          tone="irrelevant"
          to={`/documents/${doc.id}/chunks?decision=IRRELEVANT`}
        />
      </section>

      {/* Score explanation + Limitations note */}
      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Relevance Score Explanation & Limitations</h2>
        </div>
        <div className="card-body stack">
          <p className="explanation-text">{doc.score_explanation || DASH}</p>
          <div className="callout callout-info">
            <strong>Interpretability & Calibration Note:</strong> Document and chunk relevance scores are{" "}
            <em>heuristic composite confidence scores</em> combining lexical/structural financial signals
            with semantic similarity against the reference dataset (and optional LLM tie-breaker for
            borderline chunks). They are <strong>not calibrated probabilities</strong>; threshold cutoffs are
            selected empirically on the validation benchmark.
          </div>
          {doc.warnings && doc.warnings.length > 0 && (
            <div className="callout callout-warn">
              <strong>Extraction Warnings ({doc.warnings.length}):</strong>
              <ul className="compact-list">
                {doc.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Document Metadata & Versions</h2>
          </div>
          <div className="card-body">
            <dl className="kv-grid">
              <div>
                <dt>Document ID</dt>
                <dd className="mono">{doc.id}</dd>
              </div>
              <div>
                <dt>Content SHA-256</dt>
                <dd className="mono">{doc.content_hash}</dd>
              </div>
              <div>
                <dt>Detected Format / MIME</dt>
                <dd className="mono">
                  {doc.detected_type} ({doc.mime_type})
                </dd>
              </div>
              <div>
                <dt>Pages</dt>
                <dd>{fmtNumber(doc.page_count)}</dd>
              </div>
              <div>
                <dt>Processing Time</dt>
                <dd>{fmtMs(doc.processing_ms)}</dd>
              </div>
              <div>
                <dt>Classifier Version</dt>
                <dd className="mono">{fmtText(doc.classifier_version)}</dd>
              </div>
              <div>
                <dt>Embedding Model</dt>
                <dd className="mono">{fmtText(doc.embedding_model)}</dd>
              </div>
              <div>
                <dt>Dataset Version</dt>
                <dd className="mono">{fmtText(doc.dataset_version)}</dd>
              </div>
              <div>
                <dt>Embedding Calls / Cache Hits</dt>
                <dd>
                  {fmtNumber(doc.embedding_calls)} / {fmtNumber(doc.embedding_cache_hits)}
                </dd>
              </div>
              <div>
                <dt>LLM Calls / Cost</dt>
                <dd>
                  {fmtNumber(doc.llm_calls)} ({fmtUsd(doc.llm_cost_usd)})
                </dd>
              </div>
              <div>
                <dt>Uploaded At</dt>
                <dd>{fmtDate(doc.uploaded_at)}</dd>
              </div>
              <div>
                <dt>Completed At</dt>
                <dd>{fmtDate(doc.processing_completed_at)}</dd>
              </div>
            </dl>
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Extraction Quality Diagnostics</h2>
          </div>
          <div className="card-body">
            {qualityEntries.length === 0 ? (
              <p className="muted">{DASH}</p>
            ) : (
              <dl className="kv-grid">
                {qualityEntries.map(([k, v]) => (
                  <div key={k}>
                    <dt>{humanize(k)}</dt>
                    <dd className="mono">{fmtText(v)}</dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
        </section>
      </div>

      {/* Financial Quality Gate & Validation Section */}
      {structuredState.data && (
        <section className="card">
          <div className="card-header">
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <h2 className="card-title">Financial Quality Gate & Accounting Validation</h2>
                <Badge value={structuredState.data.validation_summary.quality_gate_decision} size="lg" />
              </div>
              <p className="muted-sm">
                Independent deterministic validation layer (<code>GET /api/documents/{doc.id}/quality-gate</code>)
              </p>
            </div>
            <div className="header-actions">
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => setShowFields(!showFields)}
              >
                {showFields ? "Hide Extracted Fields" : "View Extracted Fields"}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-primary"
                onClick={() =>
                  downloadJson(structuredState.data, `${doc.filename}.structured-validation.json`)
                }
              >
                Download Validation JSON
              </button>
            </div>
          </div>
          <div className="card-body stack" style={{ gap: "16px" }}>
            {structuredState.data.validation_summary.gate_reasons?.length > 0 && (
              <div className={`callout ${structuredState.data.validation_summary.quality_gate_decision === "ACCEPT" ? "callout-info" : "callout-warn"}`}>
                <strong>Quality Gate Evaluation:</strong>
                <ul className="compact-list" style={{ marginTop: "4px" }}>
                  {structuredState.data.validation_summary.gate_reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Accounting Validations Table */}
            {structuredState.data.validation_summary.accounting_validations?.length > 0 && (
              <div>
                <h3 className="section-sub">
                  Mathematical & Accounting Identities ({structuredState.data.validation_summary.accounting_validations.length})
                </h3>
                <div style={{ overflowX: "auto" }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Rule ID</th>
                        <th>Statement</th>
                        <th>Period</th>
                        <th>Formula</th>
                        <th>Expected</th>
                        <th>Actual</th>
                        <th>Diff</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {structuredState.data.validation_summary.accounting_validations.map((chk, i) => (
                        <tr key={i}>
                          <td className="mono" style={{ fontSize: "0.8rem" }}>{chk.rule_id}</td>
                          <td>{chk.statement}</td>
                          <td>{chk.period}</td>
                          <td style={{ fontSize: "0.85rem" }}>{chk.formula}</td>
                          <td className="mono">{chk.expected}</td>
                          <td className="mono">{chk.actual}</td>
                          <td className="mono">{chk.difference}</td>
                          <td>
                            <Badge value={chk.status} size="sm" />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Extracted Financial Fields Table */}
            {showFields && (
              <div>
                <h3 className="section-sub">
                  Normalized Financial Fields with Audit Provenance
                </h3>
                <div style={{ overflowX: "auto", maxHeight: "400px" }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Label / Field</th>
                        <th>Original Text</th>
                        <th>Period</th>
                        <th>Reported Scale</th>
                        <th>Normalized Amount</th>
                        <th>Signed Accounting Value</th>
                        <th>Change Direction</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {structuredState.data.items.flatMap((item) => item.financial_fields).map((f, i) => (
                        <tr key={i}>
                          <td><strong>{f.row_label || f.field}</strong></td>
                          <td className="mono">{f.original_text}</td>
                          <td>{f.reporting_period || DASH}</td>
                          <td>{f.reported_scale || "units"}</td>
                          <td className="mono">{f.normalized_amount || f.normalized_value}</td>
                          <td className="mono">{f.signed_accounting_value || DASH}</td>
                          <td>{f.change_direction || DASH}</td>
                          <td><Badge value={f.validation_status} size="sm" /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Downstream Handoff JSON Preview */}
      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="card-title">Downstream Extraction Handoff Payload</h2>
            <p className="muted-sm">
              Structured contract (<code>GET /api/documents/{doc.id}/relevant-chunks</code>) for downstream
              financial extraction services
            </p>
          </div>
          <div className="header-actions">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={includeUncertain}
                onChange={(e) => void handlePreviewHandoff(e.target.checked)}
              />
              Include UNCERTAIN chunks
            </label>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              disabled={action.busy}
              onClick={() => void handlePreviewHandoff(includeUncertain)}
            >
              {handoffPreview ? "Refresh Payload" : "Preview Handoff JSON"}
            </button>
            {handoffPreview && (
              <button
                type="button"
                className="btn btn-sm btn-primary"
                onClick={() =>
                  downloadJson(handoffPreview, `${doc.filename}.extraction-handoff.json`)
                }
              >
                Download Handoff JSON
              </button>
            )}
          </div>
        </div>
        {handoffPreview && (
          <div className="card-body">
            <div className="muted-sm" style={{ marginBottom: "0.5rem" }}>
              {handoffPreview.items.length} chunk(s) in handoff payload ·{" "}
              <code>untrusted_content: true</code> enforced on every chunk
            </div>
            <pre className="json-preview">{JSON.stringify(handoffPreview, null, 2)}</pre>
          </div>
        )}
      </section>
    </div>
  );
}
