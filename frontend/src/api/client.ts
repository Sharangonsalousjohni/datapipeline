import type {
  BatchUploadResult,
  ChunkDecision,
  ChunkList,
  ClassifyRequest,
  DashboardStats,
  DatasetVersionOut,
  DocumentDetail,
  DocumentList,
  DocumentReport,
  EvaluationReport,
  FeedbackIn,
  FeedbackOut,
  HealthResponse,
  ImportResult,
  JobOut,
  ReferenceEntryIn,
  ReferenceEntryList,
  ReferenceEntryOut,
  ReferenceStats,
  RelevantChunksResponse,
  StructuredExtractionReport,
  UploadResult,
  ValidationSummary,
} from "./types";

export const API_KEY_STORAGE = "fri.apiKey";

export function getApiBase(): string {
  const base = import.meta.env.VITE_API_BASE ?? "";
  return base.replace(/\/+$/, "");
}

export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE, key);
    else localStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* storage unavailable */
  }
}

export function apiUrl(path: string): string {
  return `${getApiBase()}${path}`;
}

export function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

/** Structured API error built from backend `{"error","message","details"}` bodies. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/** Parse a response body (already read as text) into an ApiError. Handles FastAPI `detail` too. */
export function parseErrorBody(status: number, statusText: string, bodyText: string): ApiError {
  let parsed: unknown = null;
  if (bodyText) {
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      parsed = null;
    }
  }
  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>;
    if (typeof obj.error === "string" || typeof obj.message === "string") {
      return new ApiError(
        status,
        typeof obj.error === "string" ? obj.error : `http_${status}`,
        typeof obj.message === "string" ? obj.message : statusText || `HTTP ${status}`,
        obj.details,
      );
    }
    if ("detail" in obj) {
      const detail = obj.detail;
      const message = typeof detail === "string" ? detail : "Request validation failed";
      return new ApiError(status, `http_${status}`, message, typeof detail === "string" ? undefined : detail);
    }
  }
  const text = bodyText && bodyText.length < 300 ? bodyText : "";
  return new ApiError(status, `http_${status}`, text || statusText || `HTTP ${status}`);
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (err instanceof Error) return new ApiError(0, "network_error", err.message || "Network error");
  return new ApiError(0, "unknown_error", String(err));
}

type Query = Record<string, string | number | boolean | null | undefined>;

export function buildQuery(params?: Query): string {
  if (!params) return "";
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    usp.set(k, String(v));
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

interface RequestOptions {
  method?: string;
  query?: Query;
  json?: unknown;
  body?: BodyInit;
  signal?: AbortSignal;
}

async function rawRequest(path: string, opts: RequestOptions = {}): Promise<Response> {
  const headers: Record<string, string> = { Accept: "application/json", ...authHeaders() };
  let body: BodyInit | undefined = opts.body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  let res: Response;
  try {
    res = await fetch(apiUrl(path) + buildQuery(opts.query), {
      method: opts.method ?? "GET",
      headers,
      body,
      signal: opts.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "network_error", "Could not reach the API server. Is the backend running?");
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw parseErrorBody(res.status, res.statusText, text);
  }
  return res;
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await rawRequest(path, opts);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!text) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(res.status, "invalid_response", "The server returned a non-JSON response");
  }
}

/** Fetch a binary resource with auth headers and return it as a Blob + suggested filename. */
export async function fetchBlob(path: string, query?: Query): Promise<{ blob: Blob; filename: string | null }> {
  const res = await rawRequest(path, { query });
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") ?? "";
  let filename: string | null = null;
  const star = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(cd);
  const plain = /filename="?([^";]+)"?/i.exec(cd);
  if (star) {
    try {
      filename = decodeURIComponent(star[1].trim().replace(/^"|"$/g, ""));
    } catch {
      filename = star[1].trim();
    }
  } else if (plain) {
    filename = plain[1].trim();
  }
  return { blob, filename };
}

/** Trigger a browser download for a Blob. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export async function downloadFile(path: string, fallbackName: string, query?: Query): Promise<void> {
  const { blob, filename } = await fetchBlob(path, query);
  saveBlob(blob, filename || fallbackName);
}

export function downloadJson(data: unknown, filename: string): void {
  saveBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), filename);
}

// ---------------------------------------------------------------- upload with progress (XHR)
export interface UploadHandle {
  promise: Promise<{ status: number; result: UploadResult }>;
  abort: () => void;
}

export function uploadWithProgress(file: File, onProgress: (fraction: number) => void): UploadHandle {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{ status: number; result: UploadResult }>((resolve, reject) => {
    xhr.open("POST", apiUrl("/api/documents/upload"));
    xhr.setRequestHeader("Accept", "application/json");
    const key = getApiKey();
    if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable && ev.total > 0) onProgress(ev.loaded / ev.total);
    };
    xhr.onload = () => {
      const text = xhr.responseText ?? "";
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          onProgress(1);
          resolve({ status: xhr.status, result: JSON.parse(text) as UploadResult });
        } catch {
          reject(new ApiError(xhr.status, "invalid_response", "The server returned a non-JSON response"));
        }
      } else {
        reject(parseErrorBody(xhr.status, xhr.statusText, text));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network_error", "Network error during upload"));
    xhr.onabort = () => reject(new ApiError(0, "aborted", "Upload cancelled"));
    const fd = new FormData();
    fd.append("file", file, file.name);
    xhr.send(fd);
  });
  return { promise, abort: () => xhr.abort() };
}

// ---------------------------------------------------------------- typed endpoints
export interface DocumentQuery {
  relevance?: string;
  status?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export interface EntryQuery {
  category?: string;
  label?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export interface ImportParams {
  file: File;
  name: string;
  version: string;
  source: string;
  license: string;
  attribution?: string;
  source_url?: string;
  rebuild_index: boolean;
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/api/health", { signal }),
  stats: (signal?: AbortSignal) => request<DashboardStats>("/api/stats", { signal }),

  listDocuments: (q: DocumentQuery = {}, signal?: AbortSignal) =>
    request<DocumentList>("/api/documents", { query: { limit: 50, offset: 0, ...q }, signal }),
  batchUpload: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return request<BatchUploadResult>("/api/documents/batch-upload", { method: "POST", body: fd });
  },
  getDocument: (id: string, signal?: AbortSignal) =>
    request<DocumentDetail>(`/api/documents/${encodeURIComponent(id)}`, { signal }),
  classify: (id: string, body: ClassifyRequest) =>
    request<JobOut>(`/api/documents/${encodeURIComponent(id)}/classify`, { method: "POST", json: body }),
  listChunks: (
    id: string,
    q: { decision?: ChunkDecision | ""; limit?: number; offset?: number } = {},
    signal?: AbortSignal,
  ) => request<ChunkList>(`/api/documents/${encodeURIComponent(id)}/chunks`, { query: q, signal }),
  relevantChunks: (id: string, includeUncertain = true) =>
    request<RelevantChunksResponse>(`/api/documents/${encodeURIComponent(id)}/relevant-chunks`, {
      query: { include_uncertain: includeUncertain },
    }),
  report: (id: string) => request<DocumentReport>(`/api/documents/${encodeURIComponent(id)}/report`),
  structuredExtraction: (id: string, signal?: AbortSignal) =>
    request<StructuredExtractionReport>(`/api/documents/${encodeURIComponent(id)}/structured-extraction`, { signal }),
  qualityGate: (id: string, signal?: AbortSignal) =>
    request<ValidationSummary>(`/api/documents/${encodeURIComponent(id)}/quality-gate`, { signal }),
  downloadOriginal: (id: string, fallbackName: string) =>
    downloadFile(`/api/documents/${encodeURIComponent(id)}/file`, fallbackName),

  getJob: (jobId: string, signal?: AbortSignal) =>
    request<JobOut>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal }),
  retryJob: (jobId: string) => request<JobOut>(`/api/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" }),

  submitFeedback: (chunkId: string, body: FeedbackIn) =>
    request<FeedbackOut>(`/api/chunks/${encodeURIComponent(chunkId)}/feedback`, { method: "POST", json: body }),
  listFeedback: (signal?: AbortSignal) => request<FeedbackOut[]>("/api/feedback", { signal }),
  exportFeedback: () => downloadFile("/api/feedback/export", "feedback.jsonl"),

  referenceStats: (signal?: AbortSignal) => request<ReferenceStats>("/api/reference-dataset/stats", { signal }),
  listEntries: (q: EntryQuery = {}, signal?: AbortSignal) =>
    request<ReferenceEntryList>("/api/reference-dataset/entries", { query: { limit: 50, offset: 0, ...q }, signal }),
  createEntry: (body: ReferenceEntryIn) =>
    request<ReferenceEntryOut>("/api/reference-dataset/entries", { method: "POST", json: body }),
  updateEntry: (id: string, body: ReferenceEntryIn) =>
    request<ReferenceEntryOut>(`/api/reference-dataset/entries/${encodeURIComponent(id)}`, {
      method: "PUT",
      json: body,
    }),
  deactivateEntry: (id: string) =>
    request<unknown>(`/api/reference-dataset/entries/${encodeURIComponent(id)}`, { method: "DELETE" }),
  importDataset: (p: ImportParams) => {
    const fd = new FormData();
    fd.append("file", p.file, p.file.name);
    fd.append("name", p.name);
    fd.append("version", p.version);
    fd.append("source", p.source);
    fd.append("license", p.license);
    if (p.attribution) fd.append("attribution", p.attribution);
    if (p.source_url) fd.append("source_url", p.source_url);
    fd.append("rebuild_index", p.rebuild_index ? "true" : "false");
    return request<ImportResult>("/api/reference-dataset/import", { method: "POST", body: fd });
  },
  rebuildEmbeddings: () => request<JobOut>("/api/reference-dataset/rebuild-embeddings", { method: "POST" }),
  datasetVersions: (signal?: AbortSignal) =>
    request<DatasetVersionOut[]>("/api/reference-dataset/versions", { signal }),

  latestEvaluation: (signal?: AbortSignal) => request<EvaluationReport>("/api/evaluation/latest", { signal }),
};
