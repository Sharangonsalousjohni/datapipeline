import { api } from "../api/client";
import { Badge } from "../components/Badge";
import { ErrorBanner } from "../components/ErrorBanner";
import { PRCurve, type PRPoint } from "../components/PRCurve";
import { StatCard } from "../components/StatCard";
import { useApi } from "../hooks/useApi";
import {
  asArray,
  asRecord,
  DASH,
  fmtDate,
  fmtDecimal,
  fmtMs,
  fmtNumber,
  fmtPercent,
  fmtText,
  fmtUsd,
  get,
  humanize,
  isNum,
} from "../utils/format";

function renderChunkSplitCard(title: string, splitObj: Record<string, unknown> | null) {
  const cm = asRecord(splitObj?.confusion_matrix);
  return (
    <section className="card">
      <div className="card-header">
        <h2 className="card-title">{title}</h2>
        {isNum(splitObj?.threshold) && (
          <span className="mono muted-sm">threshold = {fmtDecimal(splitObj?.threshold, 2)}</span>
        )}
      </div>
      <div className="card-body stack">
        <div className="stat-grid stat-grid-compact">
          <StatCard label="Precision" value={fmtPercent(splitObj?.precision, 1)} tone="relevant" />
          <StatCard label="Recall" value={fmtPercent(splitObj?.recall, 1)} tone="relevant" />
          <StatCard label="F1 Score" value={fmtDecimal(splitObj?.f1, 3)} />
          <StatCard label="PR-AUC" value={fmtDecimal(splitObj?.pr_auc, 3)} />
        </div>
        {cm && (
          <div>
            <h3 className="section-sub">Confusion Matrix (TP / FP / TN / FN)</h3>
            <dl className="kv-grid kv-grid-compact">
              <div>
                <dt>True Positives (TP)</dt>
                <dd className="mono">{fmtNumber(cm.tp)}</dd>
              </div>
              <div>
                <dt>False Positives (FP)</dt>
                <dd className="mono">{fmtNumber(cm.fp)}</dd>
              </div>
              <div>
                <dt>True Negatives (TN)</dt>
                <dd className="mono">{fmtNumber(cm.tn)}</dd>
              </div>
              <div>
                <dt>False Negatives (FN)</dt>
                <dd className="mono">{fmtNumber(cm.fn)}</dd>
              </div>
            </dl>
          </div>
        )}
      </div>
    </section>
  );
}

export function EvaluationPage() {
  const evalState = useApi((signal) => api.latestEvaluation(signal), []);
  const report = evalState.data;
  const isNotFound = evalState.error?.status === 404;

  if (evalState.loading && !report) {
    return (
      <div className="page">
        <p className="muted">Loading latest evaluation report…</p>
      </div>
    );
  }

  if (isNotFound || (!report && !evalState.error)) {
    return (
      <div className="page">
        <header className="page-header">
          <div>
            <h1 className="page-title">Evaluation Benchmark</h1>
            <p className="page-subtitle">
              Empirical precision, recall, PR-AUC, document filtering rate, and format breakdown
            </p>
          </div>
        </header>
        <section className="card">
          <div className="card-body empty-state">
            <h2>No Evaluation Report Available Yet</h2>
            <p className="muted">
              Run the evaluation harness (<code>python -m app.evaluate</code>) to generate{" "}
              <code>evaluation_report.json</code> with validation/test metrics and precision-recall curves.
            </p>
          </div>
        </section>
      </div>
    );
  }

  if (evalState.error && !report) {
    return (
      <div className="page">
        <ErrorBanner error={evalState.error} onRetry={evalState.reload} />
      </div>
    );
  }

  const valMetrics =
    asRecord(get(report, "chunk_metrics", "validation")) ?? asRecord(get(report, "validation"));
  const testMetrics =
    asRecord(get(report, "chunk_metrics", "test")) ?? asRecord(get(report, "test"));
  const docMetrics = asRecord(get(report, "document_metrics"));

  const rawCurve =
    asArray(testMetrics?.pr_curve).length > 0
      ? asArray(testMetrics?.pr_curve)
      : asArray(valMetrics?.pr_curve).length > 0
        ? asArray(valMetrics?.pr_curve)
        : asArray(get(report, "pr_curve"));

  const prPoints: PRPoint[] = rawCurve
    .map((pt) => asRecord(pt))
    .filter((pt): pt is Record<string, unknown> => Boolean(pt))
    .map((pt) => ({
      threshold: isNum(pt.threshold) ? pt.threshold : undefined,
      precision: isNum(pt.precision) ? pt.precision : 0,
      recall: isNum(pt.recall) ? pt.recall : 0,
      f1: isNum(pt.f1) ? pt.f1 : undefined,
    }));

  const chosenThreshold =
    (isNum(get(report, "chosen_threshold")) ? (get(report, "chosen_threshold") as number) : null) ??
    (isNum(testMetrics?.threshold) ? (testMetrics?.threshold as number) : null) ??
    (isNum(valMetrics?.threshold) ? (valMetrics?.threshold as number) : null);

  const perDocList = asArray(docMetrics?.per_document ?? get(report, "per_document"))
    .map((d) => asRecord(d))
    .filter((d): d is Record<string, unknown> => Boolean(d));

  const byCategory = asRecord(get(report, "by_category") ?? testMetrics?.by_category);
  const byFormat = asRecord(get(report, "by_format") ?? docMetrics?.by_format);
  const latency = asRecord(get(report, "latency"));
  const cost = asRecord(get(report, "cost"));
  const ocrStats = asRecord(get(report, "ocr"));

  const ocrFailureRate =
    get(report, "ocr_failure_rate") ?? ocrStats?.failure_rate ?? docMetrics?.ocr_failure_rate;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Evaluation Benchmark Report</h1>
          <p className="page-subtitle">
            Generated: {fmtDate(get(report, "generated_at"))} · Classifier:{" "}
            <span className="mono">{fmtText(get(report, "classifier_version"))}</span> · Embedding:{" "}
            <span className="mono">{fmtText(get(report, "embedding_model"))}</span>
          </p>
        </div>
        <div className="header-actions">
          <button type="button" className="btn btn-secondary" onClick={evalState.reload}>
            Refresh
          </button>
        </div>
      </header>

      <div className="grid-2">
        {renderChunkSplitCard("Validation Split (Threshold Calibration)", valMetrics)}
        {renderChunkSplitCard("Held-Out Test Split", testMetrics)}
      </div>

      <div className="grid-2">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Precision-Recall Curve</h2>
          </div>
          <div className="card-body">
            <PRCurve
              points={prPoints}
              chosenThreshold={chosenThreshold}
              prAuc={isNum(testMetrics?.pr_auc) ? (testMetrics.pr_auc as number) : null}
            />
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Document-Level Filtering & Telemetry</h2>
          </div>
          <div className="card-body stack">
            <div className="stat-grid stat-grid-compact">
              <StatCard
                label="Document Accuracy"
                value={fmtPercent(docMetrics?.accuracy ?? docMetrics?.exact_match_accuracy, 1)}
                tone="relevant"
              />
              <StatCard
                label="Irrelevant Filtering Rate"
                value={fmtPercent(docMetrics?.irrelevant_filtering_rate, 1)}
                tone="irrelevant"
              />
              <StatCard
                label="Relevant Retention Rate"
                value={fmtPercent(docMetrics?.relevant_retention_rate ?? docMetrics?.relevant_recall, 1)}
                tone="relevant"
              />
              <StatCard label="OCR Failure Rate" value={fmtPercent(ocrFailureRate, 1)} />
            </div>

            <dl className="kv-grid">
              <div>
                <dt>Mean Latency per Doc</dt>
                <dd>{fmtMs(latency?.mean_ms ?? latency?.avg_ms ?? get(report, "avg_processing_ms"))}</dd>
              </div>
              <div>
                <dt>P95 Latency per Doc</dt>
                <dd>{fmtMs(latency?.p95_ms ?? get(report, "p95_processing_ms"))}</dd>
              </div>
              <div>
                <dt>Embedding Calls / Cache Hits</dt>
                <dd>
                  {fmtNumber(cost?.embedding_calls ?? get(report, "embedding_calls"))} /{" "}
                  {fmtNumber(cost?.embedding_cache_hits ?? get(report, "embedding_cache_hits"))}
                </dd>
              </div>
              <div>
                <dt>LLM Calls / Est. Cost</dt>
                <dd>
                  {fmtNumber(cost?.llm_calls ?? get(report, "llm_calls"))} (
                  {fmtUsd(cost?.llm_cost_usd ?? get(report, "llm_cost_usd"))})
                </dd>
              </div>
            </dl>
          </div>
        </section>
      </div>

      {/* Breakdown by Category and Format */}
      <div className="grid-2">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Per-Category Performance Breakdown</h2>
          </div>
          <div className="card-body">
            {byCategory && Object.keys(byCategory).length > 0 ? (
              <div className="table-scroll">
                <table className="data-table data-table-compact">
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Support</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1 / Recall</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(byCategory).map(([cat, raw]) => {
                      const m = asRecord(raw);
                      return (
                        <tr key={cat}>
                          <td>{humanize(cat)}</td>
                          <td className="mono">{fmtNumber(m?.support ?? m?.count ?? m?.total)}</td>
                          <td className="mono">{fmtPercent(m?.precision, 1)}</td>
                          <td className="mono">{fmtPercent(m?.recall ?? m?.detection_rate, 1)}</td>
                          <td className="mono">{fmtDecimal(m?.f1 ?? m?.mean_score, 2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted-sm">{DASH}</p>
            )}
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Per-Format Performance Breakdown</h2>
          </div>
          <div className="card-body">
            {byFormat && Object.keys(byFormat).length > 0 ? (
              <div className="table-scroll">
                <table className="data-table data-table-compact">
                  <thead>
                    <tr>
                      <th>Format</th>
                      <th>Documents / Chunks</th>
                      <th>Accuracy / F1</th>
                      <th>Mean Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(byFormat).map(([fmt, raw]) => {
                      const m = asRecord(raw);
                      return (
                        <tr key={fmt}>
                          <td className="mono">{fmt}</td>
                          <td className="mono">{fmtNumber(m?.documents ?? m?.count ?? m?.support)}</td>
                          <td className="mono">
                            {isNum(m?.accuracy)
                              ? fmtPercent(m?.accuracy, 1)
                              : fmtDecimal(m?.f1, 3)}
                          </td>
                          <td className="mono">{fmtMs(m?.mean_ms ?? m?.avg_ms)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted-sm">{DASH}</p>
            )}
          </div>
        </section>
      </div>

      {/* Per-document evaluation table */}
      {perDocList.length > 0 && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Per-Document Benchmark Results ({perDocList.length})</h2>
          </div>
          <div className="card-body">
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Document</th>
                    <th>Format</th>
                    <th>Expected Relevance</th>
                    <th>Predicted Relevance</th>
                    <th>Match</th>
                    <th>Score</th>
                    <th>Relevant / Total Chunks</th>
                  </tr>
                </thead>
                <tbody>
                  {perDocList.map((d, i) => {
                    const expected = String(d.expected_relevance ?? d.expected ?? "");
                    const predicted = String(d.predicted_relevance ?? d.predicted ?? "");
                    const correct =
                      typeof d.correct === "boolean" ? d.correct : expected !== "" && expected === predicted;
                    return (
                      <tr key={String(d.filename ?? d.id ?? i)}>
                        <td className="mono">{fmtText(d.filename ?? d.id)}</td>
                        <td className="mono">{fmtText(d.format ?? d.detected_type)}</td>
                        <td>
                          <Badge value={expected} size="sm" />
                        </td>
                        <td>
                          <Badge value={predicted} size="sm" />
                        </td>
                        <td>
                          <Badge tone={correct ? "relevant" : "failed"} size="sm">
                            {correct ? "Correct" : "Mismatch"}
                          </Badge>
                        </td>
                        <td className="mono">{fmtDecimal(d.score ?? d.relevance_score, 2)}</td>
                        <td className="mono">
                          {fmtNumber(d.relevant_chunks)} / {fmtNumber(d.total_chunks)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
