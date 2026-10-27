import { ApiError } from "../api/client";

interface ErrorBannerProps {
  error: ApiError | Error | string | null | undefined;
  onRetry?: () => void;
  onDismiss?: () => void;
  title?: string;
}

function renderDetails(details: unknown): string | null {
  if (details === null || details === undefined) return null;
  if (typeof details === "string") return details;
  try {
    return JSON.stringify(details, null, 2);
  } catch {
    return String(details);
  }
}

/** Displays structured backend errors: {"error": code, "message": ..., "details": ...}. */
export function ErrorBanner({ error, onRetry, onDismiss, title }: ErrorBannerProps) {
  if (!error) return null;
  const message = typeof error === "string" ? error : error.message;
  const code = error instanceof ApiError ? error.code : null;
  const status = error instanceof ApiError && error.status ? error.status : null;
  const details = error instanceof ApiError ? renderDetails(error.details) : null;
  return (
    <div className="banner banner-error" role="alert">
      <div className="banner-body">
        <strong>{title ?? "Request failed"}</strong>
        {(code || status) && (
          <span className="banner-code">
            {status ? `HTTP ${status}` : ""}
            {status && code ? " · " : ""}
            {code}
          </span>
        )}
        <div>{message}</div>
        {details && (
          <details className="banner-details">
            <summary>Details</summary>
            <pre>{details}</pre>
          </details>
        )}
      </div>
      <div className="banner-actions">
        {onRetry && (
          <button type="button" className="btn btn-sm" onClick={onRetry}>
            Retry
          </button>
        )}
        {onDismiss && (
          <button type="button" className="btn btn-sm btn-ghost" onClick={onDismiss} aria-label="Dismiss">
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
