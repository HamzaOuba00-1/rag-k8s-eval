import { useEffect, useState } from "react";

export type Loadable<T> = { data?: T; error?: string; loading: boolean };

/** Loads once on mount (or when `deps` change); aborts on unmount. */
export function useApi<T>(load: (signal: AbortSignal) => Promise<T>, deps: unknown[] = []): Loadable<T> {
  const [state, setState] = useState<Loadable<T>>({ loading: true });
  useEffect(() => {
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true, error: undefined }));
    load(controller.signal)
      .then((data) => setState({ data, loading: false }))
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setState({ loading: false, error: e instanceof Error ? e.message : "Something went wrong" });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
