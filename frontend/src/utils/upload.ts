/** Client-side upload validation (mirrors backend limits; the backend re-validates by content). */

export const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".txt", ".csv"];
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
export const ACCEPT_ATTR = ALLOWED_EXTENSIONS.join(",");

export interface ValidationResult {
  ok: boolean;
  error?: string;
}

export function getExtension(filename: string): string {
  const idx = filename.lastIndexOf(".");
  if (idx < 0) return "";
  return filename.slice(idx).toLowerCase();
}

export function validateUploadFile(
  file: { name: string; size: number },
  maxBytes: number = MAX_UPLOAD_BYTES,
): ValidationResult {
  const ext = getExtension(file.name);
  if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
    return {
      ok: false,
      error: `Unsupported file type${ext ? ` "${ext}"` : ""}. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`,
    };
  }
  if (file.size <= 0) {
    return { ok: false, error: "File is empty" };
  }
  if (file.size > maxBytes) {
    return { ok: false, error: `File exceeds the ${Math.round(maxBytes / (1024 * 1024))} MB limit` };
  }
  return { ok: true };
}
