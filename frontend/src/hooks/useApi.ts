import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, toApiError } from "../api/client";

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
  setData: (d: T | null) => void;
}

/**
 * Run an async loader whenever `deps` change. Aborts stale requests.
 * `pollMs` (optional): re-run silently on an interval while `shouldPoll(data)` returns true.
 */
export function useApi<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  opts: { pollMs?: number; shouldPoll?: (data: T | null) => boolean; enabled?: boolean } = {},
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(opts.enabled !== false);
  const [tick, setTick] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const shouldPollRef = useRef(opts.shouldPoll);
  shouldPollRef.current = opts.shouldPoll;

  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (opts.enabled === false) {
      setLoading(false);
      return;
    }
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let cancelled = false;

    const run = async (silent: boolean) => {
      if (!silent) setLoading(true);
      try {
        const result = await loaderRef.current(ctrl.signal);
        if (cancelled) return;
        setData(result);
        setError(null);
        if (opts.pollMs && shouldPollRef.current?.(result)) {
          timer = setTimeout(() => void run(true), opts.pollMs);
        }
      } catch (e) {
        if (cancelled || (e instanceof DOMException && e.name === "AbortError")) return;
        setError(toApiError(e));
      } finally {
        if (!cancelled && !silent) setLoading(false);
      }
    };
    void run(false);
    return () => {
      cancelled = true;
      ctrl.abort();
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, opts.enabled, opts.pollMs]);

  return { data, error, loading, reload, setData };
}

/** Wrap an async action with busy/error state. */
export function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const run = useCallback(async <R,>(fn: () => Promise<R>): Promise<R | undefined> => {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(toApiError(e));
      return undefined;
    } finally {
      setBusy(false);
    }
  }, []);
  return { busy, error, setError, run };
}
