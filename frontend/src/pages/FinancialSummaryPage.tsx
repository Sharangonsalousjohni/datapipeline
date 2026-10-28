import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, downloadJson } from "../api/client";
import type { FinancialSummary, KeyFigure } from "../api/types";
import { Badge } from "../components/Badge";
import { ErrorBanner } from "../components/ErrorBanner";
import { useApi } from "../hooks/useApi";
import { DASH, fmtDate, humanize } from "../utils/format";

type View = "readable" | "json";

const MEASURE_LABELS: Record<string, string> = {
  level: "",
  price: "",
  rate: "",
  count: "",
  maturity_year: "",
  change_pct: "change",
  change_amount: "change",
  change_bps: "change",
  pct_of_revenue: "% of revenue",
};

function measureLabel(k: KeyFigure): string {
  return MEASURE_LABELS[k.measure] ?? humanize(k.measure).toLowerCase();
}

export function FinancialSummaryPage() {
  const { id = "" } = useParams<{ id: string }>();
  const [view, setView] = useState<View>("readable");
  const [copied, setCopied] = useState(false);
  const state = useApi((signal) => api.financialSummary(id, signal), [id]);
  const s = state.data;

  const handleCopy = async () => {
    if (!s) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(s, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable: the Download button still works */
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <div className="breadcrumb">
            <Link to="/documents">Documents</Link> / <Link to={`/documents/${id}`}>{s?.document.filename ?? "Document"}</Link> /{" "}
            <span>Financial summary</span>
          </div>
          <h1 className="page-title">Financial Summary</h1>
          <p className="page-subtitle">
            Validated figures, statements and terms extracted from the document (<code>GET /api/documents/{id}/summary</code>)
          </p>
        </div>
        <div className="header-actions">
          <button type="button" className="btn btn-secondary" disabled={state.loading} onClick={state.reload}>
            {state.loading ? "Generating…" : "Regenerate"}
          </button>
          <button type="button" className="btn btn-secondary" disabled={!s} onClick={() => void handleCopy()}>
            {copied ? "Copied" : "Copy JSON"}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!s}
            onClick={() => s && downloadJson(s, `${s.document.filename}.financial-summary.json`)}
          >
            Download JSON
          </button>
        </div>
      </header>

      <ErrorBanner error={state.error} onRetry={state.reload} />
      {state.loading && !s && <p className="muted-sm">Generating summary…</p>}

      {s && (
        <>
          <div className="summary-tabs" role="tablist" aria-label="Summary view">
            {(["readable", "json"] as const).map((v) => (
              <button
                key={v}
                type="button"
                role="tab"
                aria-selected={view === v}
                className={`summary-tab${view === v ? " summary-tab-active" : ""}`}
                onClick={() => setView(v)}
              >
                {v === "readable" ? "Readable" : "JSON payload"}
              </button>
            ))}
          </div>
          {view === "json" ? <pre className="summary-json">{JSON.stringify(s, null, 2)}</pre> : <ReadableSummary s={s} />}
        </>
      )}
    </div>
  );
}

function ReadableSummary({ s }: { s: FinancialSummary }) {
  return (
    <div className="stack summary-stack">
      <section className="card">
        <div className="card-header">
          <div className="summary-title-row">
            <h2 className="card-title">Outcome</h2>
            <Badge value={s.outcome.decision} size="lg" />
          </div>
          <span className="muted-sm">Generated {fmtDate(s.generated_at)}</span>
        </div>
        <div className="card-body stack summary-stack">
          <p className="summary-headline">{s.outcome.headline}</p>
          <dl className="summary-meta">
            <div>
              <dt>Document type</dt>
              <dd>{s.document.document_type}</dd>
            </div>
            <div>
              <dt>Currency</dt>
              <dd>{s.reporting.currency ?? DASH}</dd>
            </div>
            <div>
              <dt>Reported in</dt>
              <dd>{s.reporting.reported_in ? humanize(s.reporting.reported_in) : DASH}</dd>
            </div>
            <div>
              <dt>Periods</dt>
              <dd>{s.reporting.periods.length ? s.reporting.periods.join(", ") : DASH}</dd>
            </div>
            <div>
              <dt>Validation checks</dt>
              <dd>
                {s.checks.total ? `${s.checks.passed} of ${s.checks.total} passed` : DASH}
                {s.checks.warnings > 0 && `, ${s.checks.warnings} warning(s)`}
                {s.checks.failed > 0 && `, ${s.checks.failed} failed`}
              </dd>
            </div>
          </dl>
          {s.checks.issues.length > 0 && (
            <ul className="summary-issues">
              {s.checks.issues.map((msg, i) => (
                <li key={i}>{msg}</li>
              ))}
            </ul>
          )}
        </div>
      </section>

      {s.needs_review.length > 0 && (
        <section className="card summary-review">
          <div className="card-header">
            <h2 className="card-title">Needs review ({s.needs_review.length})</h2>
            <span className="muted-sm">Not verified, so excluded from the figures below</span>
          </div>
          <div className="summary-scroll">
            <table>
              <thead>
                <tr>
                  <th>Item</th>
                  <th>As reported</th>
                  <th>Reason</th>
                  <th>Page</th>
                </tr>
              </thead>
              <tbody>
                {s.needs_review.map((r, i) => (
                  <tr key={i}>
                    <td>{r.label}</td>
                    <td className="mono">{r.as_reported}</td>
                    <td>{r.reason}</td>
                    <td>{r.page ?? DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {s.key_figures.length > 0 && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Key figures ({s.key_figures.length})</h2>
          </div>
          <div className="summary-scroll">
            <table>
              <thead>
                <tr>
                  <th>Figure</th>
                  <th>Period</th>
                  <th className="summary-num">Value</th>
                  <th>As reported</th>
                  <th>Page</th>
                </tr>
              </thead>
              <tbody>
                {s.key_figures.map((k, i) => (
                  <tr key={i}>
                    <td>
                      {k.label}
                      {measureLabel(k) && <span className="muted-sm"> · {measureLabel(k)}</span>}
                    </td>
                    <td>{k.period ?? DASH}</td>
                    <td className="summary-num">
                      <strong>{k.display}</strong>
                    </td>
                    <td className="mono muted-sm">{k.as_reported}</td>
                    <td>{k.page ?? DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {s.terms.length > 0 && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Contract terms ({s.terms.length})</h2>
          </div>
          <div className="summary-scroll">
            <table>
              <thead>
                <tr>
                  <th>Term</th>
                  <th>Value</th>
                  <th>Frequency</th>
                  <th>Due</th>
                  <th>As reported</th>
                </tr>
              </thead>
              <tbody>
                {s.terms.map((t) => (
                  <tr key={t.term}>
                    <td>{t.label}</td>
                    <td>
                      <strong>{t.display}</strong>
                    </td>
                    <td>{t.frequency ? humanize(t.frequency) : DASH}</td>
                    <td>{t.due ?? DASH}</td>
                    <td className="mono muted-sm">{t.as_reported}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {s.statements.map((st) => (
        <section className="card" key={st.statement}>
          <div className="card-header">
            <h2 className="card-title">{st.title}</h2>
            <span className="muted-sm">
              {[st.currency, st.reported_in && `reported in ${st.reported_in}`, st.page && `page ${st.page}`]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </div>
          <div className="summary-scroll">
            <table>
              <thead>
                <tr>
                  <th>Line item</th>
                  {st.columns.map((c) => (
                    <th key={c} className="summary-num">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {st.lines.map((line) => (
                  <tr key={line.label}>
                    <td>{line.label}</td>
                    {st.columns.map((c) => (
                      <td key={c} className="summary-num" title={line.values[c]?.as_reported}>
                        {line.values[c]?.display ?? DASH}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      {!s.key_figures.length && !s.terms.length && !s.statements.length && (
        <section className="card">
          <div className="card-body">
            <p className="muted-sm">No validated financial values were extracted from this document.</p>
          </div>
        </section>
      )}
    </div>
  );
}
