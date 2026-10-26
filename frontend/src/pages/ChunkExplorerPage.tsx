import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { CHUNK_DECISIONS, type ChunkDecision, type ChunkOut } from "../api/types";
import { Badge } from "../components/Badge";
import { BarChart } from "../components/BarChart";
import { ErrorBanner } from "../components/ErrorBanner";
import { HighlightedText } from "../components/HighlightedText";
import { LocationView } from "../components/LocationView";
import { useAction, useApi } from "../hooks/useApi";
import { DASH, fmtDate, fmtDecimal, fmtNumber, fmtPercent, humanize } from "../utils/format";

const TABS: { label: string; value: ChunkDecision | "" }[] = [
  { label: "All Chunks", value: "" },
  { label: "Relevant", value: "RELEVANT" },
  { label: "Uncertain", value: "UNCERTAIN" },
  { label: "Irrelevant", value: "IRRELEVANT" },
];

interface ChunkCardProps {
  chunk: ChunkOut;
  onFeedbackSaved: () => void;
}

function ChunkCard({ chunk, onFeedbackSaved }: ChunkCardProps) {
  const cls = chunk.classification;
  const [openForm, setOpenForm] = useState(false);
  const [decision, setDecision] = useState<ChunkDecision>(cls?.decision ?? "RELEVANT");
  const [category, setCategory] = useState<string>(cls?.top_category ?? "");
  const [comment, setComment] = useState("");
  const [reviewer, setReviewer] = useState("");
  const action = useAction();

  const categoryItems = cls
    ? Object.entries(cls.category_scores ?? {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(([k, v]) => ({ label: k, value: v, tone: "default" as const }))
    : [];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const res = await action.run(() =>
      api.submitFeedback(chunk.id, {
        corrected_decision: decision,
        corrected_category: category.trim() || null,
        comment: comment.trim() || null,
        reviewer: reviewer.trim() || null,
      }),
    );
    if (res) {
      setOpenForm(false);
      setComment("");
      onFeedbackSaved();
    }
  };

  return (
    <article className="card chunk-card">
      <div className="card-header chunk-card-header">
        <div className="badge-row">
          <span className="mono muted-sm">#{chunk.chunk_index}</span>
          <Badge value={cls?.decision ?? "UNCERTAIN"} />
          <Badge tone="neutral" size="sm">
            {chunk.chunk_type}
          </Badge>
          {cls?.top_category && (
            <Badge tone="info" size="sm">
              {humanize(cls.top_category)}
            </Badge>
          )}
          <span className="mono muted-sm">
            method: {chunk.extraction_method}
            {chunk.ocr_confidence !== null ? ` (OCR ${fmtPercent(chunk.ocr_confidence, 0)})` : ""}
          </span>
        </div>
        <div className="badge-row">
          <span className="score-pill">
            Score: <strong>{fmtDecimal(cls?.score, 2)}</strong>
          </span>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => setOpenForm((v) => !v)}
          >
            {openForm ? "Cancel" : "Correct Classification"}
          </button>
        </div>
      </div>

      <div className="card-body stack">
        {/* Source Location */}
        <div className="chunk-meta-row">
          <span className="meta-label">Source Location:</span>
          <LocationView location={chunk.location} extraLocations={chunk.extra_locations} />
        </div>

        {chunk.context && (
          <div className="chunk-meta-row">
            <span className="meta-label">Context:</span>
            <span className="muted-sm">{chunk.context}</span>
          </div>
        )}

        {/* Highlighted Chunk Text */}
        <HighlightedText
          text={chunk.text}
          chunkType={chunk.chunk_type}
          signals={cls?.signals}
          negativeSignals={cls?.negative_signals}
        />

        {/* Score Breakdown & Category Scores */}
        {cls && (
          <div className="grid-2 chunk-diagnostics">
            <div>
              <h3 className="section-sub">Score Breakdown</h3>
              <dl className="kv-grid kv-grid-compact">
                <div>
                  <dt>Composite Score</dt>
                  <dd className="mono">{fmtDecimal(cls.score, 3)}</dd>
                </div>
                <div>
                  <dt>Lexical/Structural Signal Score</dt>
                  <dd className="mono">{fmtDecimal(cls.signal_score, 3)}</dd>
                </div>
                <div>
                  <dt>Semantic Net Score</dt>
                  <dd className="mono">{fmtDecimal(cls.semantic_score, 3)}</dd>
                </div>
                <div>
                  <dt>Max Positive Similarity</dt>
                  <dd className="mono">{fmtDecimal(cls.positive_similarity, 3)}</dd>
                </div>
                <div>
                  <dt>Max Negative Similarity</dt>
                  <dd className="mono">{fmtDecimal(cls.negative_similarity, 3)}</dd>
                </div>
                <div>
                  <dt>Decided By</dt>
                  <dd className="mono">
                    {cls.decided_by}
                    {cls.llm_used ? " (LLM tie-breaker)" : ""}
                  </dd>
                </div>
              </dl>
              {cls.llm_rationale && (
                <p className="muted-sm" style={{ marginTop: "0.4rem" }}>
                  <strong>LLM Rationale:</strong> {cls.llm_rationale}
                </p>
              )}
            </div>

            <div>
              <h3 className="section-sub">Category Scores</h3>
              <BarChart items={categoryItems} formatValue="score" maxValue={1} emptyText={DASH} />
            </div>
          </div>
        )}

        {/* Matched Reference Examples */}
        {cls && cls.matched_references && cls.matched_references.length > 0 && (
          <div>
            <h3 className="section-sub">Top Matched Reference Concepts</h3>
            <div className="matched-refs">
              {cls.matched_references.map((ref, idx) => (
                <div key={`${ref.reference_id}-${idx}`} className="matched-ref-item">
                  <div className="badge-row">
                    <Badge value={ref.label} size="sm" />
                    <strong>{ref.concept}</strong>
                    <span className="muted-sm">({humanize(ref.category)})</span>
                    <span className="mono muted-sm">sim={fmtDecimal(ref.similarity, 3)}</span>
                  </div>
                  <div className="muted-sm">“{ref.matched_text}”</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Existing Reviewer Feedback */}
        {chunk.feedback && chunk.feedback.length > 0 && (
          <div className="feedback-history">
            <h3 className="section-sub">Reviewer Corrections ({chunk.feedback.length})</h3>
            {chunk.feedback.map((fb) => (
              <div key={fb.id} className="feedback-pill">
                <Badge value={fb.original_decision} size="sm" /> →{" "}
                <Badge value={fb.corrected_decision} size="sm" />
                {fb.corrected_category && (
                  <span className="mono muted-sm"> [{humanize(fb.corrected_category)}]</span>
                )}
                {fb.comment && <span> — “{fb.comment}”</span>}
                <span className="muted-sm">
                  {" "}
                  ({fb.reviewer || "anonymous"}, {fmtDate(fb.created_at)})
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Inline Feedback Form */}
        {openForm && (
          <form className="feedback-form" onSubmit={(e) => void handleSubmit(e)}>
            <h4 className="section-sub">Submit Reviewer Correction</h4>
            <ErrorBanner error={action.error} />
            <div className="form-row">
              <div className="form-field">
                <label htmlFor={`decision-${chunk.id}`}>Corrected Decision</label>
                <select
                  id={`decision-${chunk.id}`}
                  className="select"
                  value={decision}
                  onChange={(e) => setDecision(e.target.value as ChunkDecision)}
                >
                  {CHUNK_DECISIONS.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-field">
                <label htmlFor={`cat-${chunk.id}`}>Corrected Category (optional)</label>
                <input
                  id={`cat-${chunk.id}`}
                  type="text"
                  className="input"
                  placeholder="e.g. financial_statements"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                />
              </div>
              <div className="form-field">
                <label htmlFor={`rev-${chunk.id}`}>Reviewer (optional)</label>
                <input
                  id={`rev-${chunk.id}`}
                  type="text"
                  className="input"
                  placeholder="Your name or handle"
                  value={reviewer}
                  onChange={(e) => setReviewer(e.target.value)}
                />
              </div>
            </div>
            <div className="form-field">
              <label htmlFor={`comment-${chunk.id}`}>Comment / Rationale</label>
              <textarea
                id={`comment-${chunk.id}`}
                className="textarea"
                rows={2}
                placeholder="Why should this chunk have a different classification?"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-sm btn-primary" disabled={action.busy}>
                {action.busy ? "Saving…" : "Save Correction"}
              </button>
            </div>
          </form>
        )}
      </div>
    </article>
  );
}

export function ChunkExplorerPage() {
  const { id = "" } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const decisionParam = (searchParams.get("decision") ?? "") as ChunkDecision | "";

  const docState = useApi((signal) => api.getDocument(id, signal), [id]);
  const chunksState = useApi(
    (signal) => api.listChunks(id, { decision: decisionParam, limit: 200, offset: 0 }, signal),
    [id, decisionParam],
  );

  const doc = docState.data;
  const chunks = chunksState.data?.items ?? [];

  const selectTab = (val: ChunkDecision | "") => {
    const next = new URLSearchParams(searchParams);
    if (val) next.set("decision", val);
    else next.delete("decision");
    setSearchParams(next);
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <div className="breadcrumb">
            <Link to="/documents">Documents</Link> /{" "}
            <Link to={`/documents/${id}`}>{doc?.filename ?? id}</Link> / <span>Chunks</span>
          </div>
          <h1 className="page-title">Chunk Explorer</h1>
          <p className="page-subtitle">
            Inspect extracted text, highlighted financial signals, source coordinates, and reference matches
          </p>
        </div>
        <div className="header-actions">
          <Link to={`/documents/${id}`} className="btn btn-secondary">
            Back to Document Summary
          </Link>
        </div>
      </header>

      <div className="tabs" role="tablist" aria-label="Filter chunks by decision">
        {TABS.map((t) => {
          const active = decisionParam === t.value;
          const count =
            t.value === ""
              ? doc?.chunk_counts.total
              : t.value === "RELEVANT"
                ? doc?.chunk_counts.relevant
                : t.value === "UNCERTAIN"
                  ? doc?.chunk_counts.uncertain
                  : doc?.chunk_counts.irrelevant;
          return (
            <button
              key={t.label}
              type="button"
              role="tab"
              aria-selected={active}
              className={`tab${active ? " tab-active" : ""}`}
              onClick={() => selectTab(t.value)}
            >
              {t.label} {count !== undefined ? `(${fmtNumber(count)})` : ""}
            </button>
          );
        })}
      </div>

      <ErrorBanner error={chunksState.error ?? docState.error} onRetry={chunksState.reload} />

      {chunksState.loading && !chunksState.data ? (
        <p className="muted">Loading chunks…</p>
      ) : chunks.length === 0 ? (
        <div className="card">
          <div className="card-body empty-state">
            <p>No chunks found for this filter.</p>
          </div>
        </div>
      ) : (
        <div className="stack">
          {chunks.map((chunk) => (
            <ChunkCard key={chunk.id} chunk={chunk} onFeedbackSaved={chunksState.reload} />
          ))}
        </div>
      )}
    </div>
  );
}
