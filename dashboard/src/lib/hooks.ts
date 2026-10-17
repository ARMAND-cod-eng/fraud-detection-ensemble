import { useCallback, useEffect, useRef, useState } from "react";
import type { TxnSummary } from "./api";

/** Poll an async source. Keeps the previous data while refetching (no flashing). */
export function usePoll<T>(fn: () => Promise<T>, intervalMs: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setData(await fnRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const tick = async () => {
      await refresh();
      if (alive && intervalMs > 0) timer = window.setTimeout(tick, document.hidden ? intervalMs * 4 : intervalMs);
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, refresh, ...deps]);

  return { data, error, loading, refresh, setData };
}

export interface StreamState {
  events: TxnSummary[];
  connected: boolean;
  dropped: number;
  received: number;
}

/** Subscribe to server-sent transaction events; keeps the latest `limit` rows. */
export function useTransactionStream(limit = 200, paused = false): StreamState {
  const [state, setState] = useState<StreamState>({ events: [], connected: false, dropped: 0, received: 0 });
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const buffer = useRef<TxnSummary[]>([]);

  useEffect(() => {
    const es = new EventSource("/api/v1/stream");
    es.onopen = () => setState((s) => ({ ...s, connected: true }));
    es.onerror = () => setState((s) => ({ ...s, connected: false }));
    es.addEventListener("transaction", (ev) => {
      buffer.current.push(JSON.parse((ev as MessageEvent).data));
    });
    es.addEventListener("stats", (ev) => {
      const d = JSON.parse((ev as MessageEvent).data);
      setState((s) => ({ ...s, dropped: d.dropped ?? s.dropped }));
    });
    // Flush in frames, not per event, so 200 tps does not mean 200 renders per second.
    const flush = window.setInterval(() => {
      if (!buffer.current.length) return;
      const incoming = buffer.current.reverse();
      buffer.current = [];
      setState((s) => ({
        ...s,
        received: s.received + incoming.length,
        events: pausedRef.current ? s.events : [...incoming, ...s.events].slice(0, limit),
      }));
    }, 400);
    return () => {
      es.close();
      window.clearInterval(flush);
    };
  }, [limit]);

  return state;
}

export function useHashRoute(): [string, string[]] {
  const read = () => (window.location.hash.replace(/^#\/?/, "") || "overview").split("/");
  const [parts, setParts] = useState<string[]>(read);
  useEffect(() => {
    const on = () => setParts(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [parts[0], parts.slice(1)];
}

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}
