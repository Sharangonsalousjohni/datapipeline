import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Badge } from "../components/Badge";
import { ErrorBanner } from "../components/ErrorBanner";
import { useAction, useApi } from "../hooks/useApi";
import { DASH, fmtDate, humanize } from "../utils/format";

export function FeedbackPage() {
  const feedbackState = useApi((signal) => api.listFeedback(signal), []);
  const action = useAction();
  const items = feedbackState.data ?? [];

  const handleExport = async () => {
    await action.run(() => api.exportFeedback());
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Reviewer Feedback Log</h1>
          <p className="page-subtitle">
            Human-in-the-loop chunk classification corrections stored for retraining and dataset curation
          </p>
        </div>
        <div className="header-actions">
          <button type="button" className="btn btn-secondary" onClick={feedbackState.reload}>
            Refresh
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={action.busy}
            onClick={() => void handleExport()}
          >
            Export JSONL
          </button>
        </div>
      </header>

      <ErrorBanner error={action.error ?? feedbackState.error} onRetry={feedbackState.reload} />

      <section className="card">
        <div className="card-body">
          {feedbackState.loading && !feedbackState.data ? (
            <p className="muted">Loading reviewer feedback…</p>
          ) : items.length === 0 ? (
            <div className="empty-state">
              <p>No reviewer corrections have been submitted yet.</p>
              <p className="muted-sm">
                Open any document’s Chunk Explorer and click <strong>Correct Classification</strong> on a
                chunk to record feedback.
              </p>
            </div>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Document</th>
                    <th>Chunk</th>
                    <th>Original</th>
                    <th>Corrected Decision</th>
                    <th>Corrected Category</th>
                    <th>Comment</th>
                    <th>Reviewer</th>
                    <th>Submitted</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((fb) => (
                    <tr key={fb.id}>
                      <td className="mono">#{fb.id}</td>
                      <td>
                        {fb.document_id ? (
                          <Link to={`/documents/${fb.document_id}`} className="table-link mono">
                            {fb.document_id.slice(0, 10)}…
                          </Link>
                        ) : (
                          DASH
                        )}
                      </td>
                      <td>
                        {fb.document_id && fb.chunk_id ? (
                          <Link
                            to={`/documents/${fb.document_id}/chunks`}
                            className="table-link mono"
                          >
                            {fb.chunk_id.slice(0, 10)}…
                          </Link>
                        ) : (
                          <span className="mono">{fb.chunk_id ?? DASH}</span>
                        )}
                      </td>
                      <td>
                        <Badge value={fb.original_decision} size="sm" />
                      </td>
                      <td>
                        <Badge value={fb.corrected_decision} size="sm" />
                      </td>
                      <td>{fb.corrected_category ? humanize(fb.corrected_category) : DASH}</td>
                      <td>{fb.comment || DASH}</td>
                      <td>{fb.reviewer || DASH}</td>
                      <td>{fmtDate(fb.created_at)}</td>
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
