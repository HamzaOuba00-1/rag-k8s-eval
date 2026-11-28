import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, type AskResponse, type Quota } from "../api/client";

/** One question and its answer in the thread. */
export type Turn = {
  id: number;
  question: string;
  status: "loading" | "done" | "error";
  result?: AskResponse;
  error?: string;
};

type Chat = {
  turns: Turn[];
  busy: boolean;
  quota?: Quota;
  ask: (question: string) => void;
  retry: (id: number) => void;
  reset: () => void;
};

// Older turns are dropped past this length: the thread lives in memory only.
export const MAX_TURNS = 30;

const ChatContext = createContext<Chat | null>(null);

/**
 * Thread state for the Ask page, kept above the routes so it survives navigation.
 * Nothing is persisted (no localStorage, nothing sent to the server beyond each
 * question), and every question is answered on its own: earlier turns are not sent.
 */
export function ChatProvider({ children }: { children: ReactNode }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [quota, setQuota] = useState<Quota>();
  const busy = useRef(false);
  const [busyState, setBusyState] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const turnsRef = useRef(turns);
  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);

  useEffect(() => {
    const c = new AbortController();
    api.quota(c.signal).then(setQuota, () => undefined);
    return () => c.abort();
  }, []);

  const update = (id: number, patch: Partial<Turn>) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));

  const run = useCallback(async (id: number, question: string) => {
    busy.current = true;
    setBusyState(true);
    const c = new AbortController();
    controller.current = c;
    try {
      const result = await api.ask(question, c.signal);
      update(id, { status: "done", result, error: undefined });
      setQuota(result.quota);
    } catch (e) {
      if (!c.signal.aborted) update(id, { status: "error", error: e instanceof Error ? e.message : "Something went wrong" });
    } finally {
      if (controller.current === c) {
        busy.current = false;
        setBusyState(false);
      }
    }
  }, []);

  const ask = useCallback(
    (question: string) => {
      const q = question.trim();
      if (busy.current || q.length < 3) return;
      const id = nextId.current++;
      setTurns((ts) => [...ts, { id, question: q, status: "loading" as const }].slice(-MAX_TURNS));
      void run(id, q);
    },
    [run],
  );

  const retry = useCallback(
    (id: number) => {
      const turn = turnsRef.current.find((t) => t.id === id);
      if (busy.current || !turn) return;
      update(id, { status: "loading", error: undefined });
      void run(id, turn.question);
    },
    [run],
  );

  const reset = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    busy.current = false;
    setBusyState(false);
    setTurns([]);
  }, []);

  const value = useMemo(() => ({ turns, busy: busyState, quota, ask, retry, reset }), [turns, busyState, quota, ask, retry, reset]);
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): Chat {
  const chat = useContext(ChatContext);
  if (!chat) throw new Error("useChat must be used inside <ChatProvider>");
  return chat;
}
