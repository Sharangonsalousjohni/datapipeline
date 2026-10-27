import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toApiError, uploadWithProgress, type ApiError } from "../api/client";
import type { UploadResult } from "../api/types";
import { Badge } from "../components/Badge";
import { ErrorBanner } from "../components/ErrorBanner";
import { ProgressBar } from "../components/ProgressBar";
import { fmtBytes } from "../utils/format";
import { ACCEPT_ATTR, ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, validateUploadFile } from "../utils/upload";

interface QueueItem {
  localId: string;
  file: File;
  status: "ready" | "invalid" | "uploading" | "done" | "error";
  progress: number;
  validationError?: string;
  apiError?: ApiError;
  result?: UploadResult;
  abort?: () => void;
}

export function UploadPage() {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const enqueueFiles = (fileList: FileList | File[]) => {
    const arr = Array.from(fileList);
    if (arr.length === 0) return;

    const added: QueueItem[] = arr.map((file, idx) => {
      const check = validateUploadFile(file);
      return {
        localId: `${Date.now()}-${idx}-${file.name}`,
        file,
        status: check.ok ? "ready" : "invalid",
        progress: 0,
        validationError: check.error,
      };
    });

    setItems((prev) => [...prev, ...added]);

    for (const item of added) {
      if (item.status === "ready") {
        void startUpload(item.localId, item.file);
      }
    }
  };

  const startUpload = async (localId: string, file: File) => {
    const handle = uploadWithProgress(file, (fraction) => {
      setItems((prev) =>
        prev.map((it) => (it.localId === localId ? { ...it, progress: fraction } : it)),
      );
    });

    setItems((prev) =>
      prev.map((it) =>
        it.localId === localId
          ? { ...it, status: "uploading", progress: 0.02, apiError: undefined, abort: handle.abort }
          : it,
      ),
    );

    try {
      const { result } = await handle.promise;
      setItems((prev) =>
        prev.map((it) =>
          it.localId === localId ? { ...it, status: "done", progress: 1, result, abort: undefined } : it,
        ),
      );
    } catch (err) {
      setItems((prev) =>
        prev.map((it) =>
          it.localId === localId
            ? { ...it, status: "error", apiError: toApiError(err), abort: undefined }
            : it,
        ),
      );
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) {
      enqueueFiles(e.dataTransfer.files);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Upload Documents</h1>
          <p className="page-subtitle">
            Upload financial or general documents for automated extraction, chunking, and relevance classification.
          </p>
        </div>
        <div className="header-actions">
          <Link to="/documents" className="btn btn-secondary">
            Browse Documents
          </Link>
        </div>
      </header>

      <section
        className={`dropzone${dragging ? " dropzone-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTR}
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files) enqueueFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <div className="dropzone-title">Drop files here or click to select</div>
        <div className="dropzone-hint">
          Allowed formats: {ALLOWED_EXTENSIONS.join(", ")} · Max {Math.round(MAX_UPLOAD_BYTES / (1024 * 1024))} MB per
          file
        </div>
        <button
          type="button"
          className="btn btn-primary dropzone-btn"
          onClick={(e) => {
            e.stopPropagation();
            inputRef.current?.click();
          }}
        >
          Browse Documents
        </button>
      </section>

      {items.length > 0 && (
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">Upload Queue ({items.length})</h2>
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={() => setItems((prev) => prev.filter((i) => i.status === "uploading"))}
            >
              Clear Finished
            </button>
          </div>
          <div className="card-body stack">
            {items.map((it) => (
              <div key={it.localId} className="upload-item">
                <div className="upload-item-header">
                  <div>
                    <strong>{it.file.name}</strong>{" "}
                    <span className="muted-sm">({fmtBytes(it.file.size)})</span>
                  </div>
                  <div className="upload-item-badges">
                    {it.result?.duplicate && (
                      <Badge tone="info" size="sm" title="Identical SHA-256 content already existed">
                        Duplicate (deduplicated)
                      </Badge>
                    )}
                    {it.status === "invalid" && (
                      <Badge tone="failed" size="sm">
                        Rejected
                      </Badge>
                    )}
                    {it.status === "uploading" && (
                      <Badge tone="running" size="sm">
                        Uploading
                      </Badge>
                    )}
                    {it.status === "done" && it.result && (
                      <>
                        <Badge value={it.result.document.status} size="sm" />
                        {it.result.document.relevance_status && (
                          <Badge value={it.result.document.relevance_status} size="sm" />
                        )}
                      </>
                    )}
                    {it.status === "error" && (
                      <Badge tone="failed" size="sm">
                        Upload Failed
                      </Badge>
                    )}
                  </div>
                </div>

                {it.status === "uploading" && (
                  <div className="upload-progress-row">
                    <ProgressBar value={it.progress} label="Uploading…" />
                    {it.abort && (
                      <button type="button" className="btn btn-sm btn-ghost" onClick={it.abort}>
                        Cancel
                      </button>
                    )}
                  </div>
                )}

                {it.validationError && <ErrorBanner error={it.validationError} title="Invalid file" />}

                {it.apiError && (
                  <ErrorBanner
                    error={it.apiError}
                    onRetry={() => void startUpload(it.localId, it.file)}
                  />
                )}

                {it.result && (
                  <div className="upload-result-links">
                    <Link to={`/documents/${it.result.document.id}`} className="btn btn-sm btn-primary">
                      View Document Status & Results
                    </Link>
                    <Link to={`/documents/${it.result.document.id}/chunks`} className="btn btn-sm btn-secondary">
                      Explore Chunks
                    </Link>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
