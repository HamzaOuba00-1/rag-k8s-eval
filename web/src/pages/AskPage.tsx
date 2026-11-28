import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { api } from "../api/client";
import { AnswerText, SourceChips, SourceList } from "../components/Answer";
import { Logo } from "../components/Brand";
import { ErrorNote } from "../components/Cards";
import { Icon } from "../components/Icon";
import { type Turn, useChat } from "../lib/chat";
import { configLabel } from "../lib/format";
import { useApi } from "../lib/useApi";

const MAX_CHARS = 500;

function Avatar() {
  return (
    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-surface ring-1 ring-line-strong">
      <Logo size={22} />
    </span>
  );
}

function TurnView({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  const [tab, setTab] = useState(0);
  const [copied, setCopied] = useState(false);
  const [showExcerpts, setShowExcerpts] = useState(false);
  const result = turn.result;
  const answer = result?.answers[tab];
  const live = result?.answers.some((a) => a.origin === "live");

  return (
    <article className="space-y-4" aria-label={`Question: ${turn.question}`}>
      <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-line text-fg ring-1 ring-line-strong">
        {turn.question}
      </div>

      <div className="flex gap-3">
        <Avatar />
        <div className="min-w-0 flex-1 space-y-3">
          <p className="flex h-8 flex-wrap items-center gap-x-2 text-xs text-muted">
            <span className="font-medium text-fg">DocTruth</span>
            {result && (
              <>
                <span aria-hidden="true">·</span>
                <span>{configLabel(result.config)}</span>
                <span aria-hidden="true">·</span>
                <span>{live ? "generated now" : result.answers.length ? "evaluation run" : "sources only"}</span>
              </>
            )}
          </p>

          {turn.status === "loading" && (
            <div role="status" className="flex items-center gap-2 text-sm text-muted">
              <span className="flex gap-1" aria-hidden="true">
                {[0, 150, 300].map((d) => (
                  <span key={d} className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" style={{ animationDelay: `${d}ms` }} />
                ))}
              </span>
              Retrieving and answering…
            </div>
          )}

          {turn.status === "error" && (
            <div className="space-y-2">
              <ErrorNote message={turn.error ?? "Something went wrong"} />
              <button type="button" onClick={onRetry}
                className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs text-muted ring-1 ring-line-strong hover:text-fg">
                <Icon name="refresh" className="h-3.5 w-3.5" /> Retry
              </button>
            </div>
          )}

          {result && turn.status === "done" && (
            <>
              {result.message && (
                <p className={`rounded-xl px-4 py-2.5 text-sm ${result.demo_mode ? "bg-warn/10 text-warn ring-1 ring-warn/25" : "bg-accent/10 text-muted ring-1 ring-accent/20"}`}>
                  {result.message}
                </p>
              )}

              {result.answers.length > 1 && (
                <div role="tablist" aria-label="Model" className="flex flex-wrap gap-2">
                  {result.answers.map((a, i) => (
                    <button key={a.model} role="tab" type="button" aria-selected={tab === i} onClick={() => setTab(i)}
                      className={`rounded-full px-3 py-1.5 text-xs transition-colors ${tab === i ? "bg-accent text-ink" : "text-muted ring-1 ring-line-strong hover:text-fg"}`}>
                      {a.model} · {a.origin}
                    </button>
                  ))}
                </div>
              )}

              {answer && (
                <div className="panel rounded-tl-md p-5">
                  <AnswerText text={answer.text} sources={result.sources} />
                  <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line pt-3 text-xs text-muted">
                    <button type="button" className="flex items-center gap-1.5 hover:text-fg"
                      onClick={() => void navigator.clipboard?.writeText(answer.text).then(() => setCopied(true))}>
                      <Icon name={copied ? "check" : "copy"} className="h-4 w-4" /> {copied ? "Copied" : "Copy"}
                    </button>
                    <Link className="flex items-center gap-1.5 hover:text-fg" to={`/compare?q=${encodeURIComponent(result.question)}`}>
                      <Icon name="compare" className="h-4 w-4" /> Compare retrieval
                    </Link>
                    <span className="ml-auto">{answer.model}</span>
                  </div>
                </div>
              )}

              {result.sources.length > 0 && (
                <div className="space-y-2.5">
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <Icon name="book" className="h-3.5 w-3.5" />
                    <span>
                      {result.sources.length} sources
                      {answer ? `, ${answer.cited.length} cited` : ""}
                    </span>
                    <button type="button" aria-expanded={showExcerpts} onClick={() => setShowExcerpts((v) => !v)}
                      className="ml-auto flex items-center gap-1 hover:text-fg">
                      {showExcerpts ? "Hide excerpts" : "Show excerpts"}
                      <Icon name="chevron" className={`h-3.5 w-3.5 transition-transform ${showExcerpts ? "rotate-180" : ""}`} />
                    </button>
                  </div>
                  {showExcerpts ? (
                    <SourceList sources={result.sources} cited={answer?.cited ?? []} />
                  ) : (
                    <SourceChips sources={result.sources} cited={answer?.cited ?? []} />
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </article>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const examples = useApi((s) => api.examples(s));
  const factual = examples.data?.filter((e) => e.category === "factual").slice(0, 3) ?? [];
  const impossible = examples.data?.find((e) => e.category !== "factual");
  const picks = [...factual, ...(impossible ? [impossible] : [])];

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 py-10 text-center">
      <Logo size={80} className="drop-shadow-[0_0_28px_rgb(193_154_107/0.25)]" />
      <h1 className="mt-2 font-display text-3xl md:text-4xl">Ask the Kubernetes docs</h1>
      <p className="max-w-md text-sm text-muted">Answers cite the pages they come from, and say so when the docs don't cover it.</p>
      {picks.length > 0 && (
        <div className="mt-5 grid w-full max-w-2xl gap-3 sm:grid-cols-2">
          {picks.map((e) => (
            <button key={e.id} type="button" onClick={() => onPick(e.question)}
              className="panel group flex items-start gap-3 p-4 text-left text-sm transition-colors hover:border-line-strong hover:bg-raised">
              <span className="flex-1 leading-relaxed">{e.question}</span>
              {e.category !== "factual" ? (
                <span className="rounded-full bg-warn/10 px-2 py-0.5 text-[11px] whitespace-nowrap text-warn ring-1 ring-warn/20">not in the docs</span>
              ) : (
                <Icon name="arrow" className="mt-0.5 h-4 w-4 text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
              )}
            </button>
          ))}
        </div>
      )}
      <Link to="/compare" className="mt-2 text-xs text-muted underline decoration-line-strong underline-offset-4 hover:text-fg">
        Or compare what each chunking strategy retrieves
      </Link>
    </div>
  );
}

export function AskPage() {
  const { turns, busy, quota, ask, retry, reset } = useChat();
  const [params, setParams] = useSearchParams();
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string>();
  const handled = useRef<string | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  // /ask?q=... (links from Home and Compare): ask once, then clear the URL.
  useEffect(() => {
    const q = params.get("q");
    if (!q || handled.current === q) return;
    handled.current = q;
    setParams({}, { replace: true });
    if (q.trim().length >= 3) ask(q);
  }, [params, setParams, ask]);

  const last = turns[turns.length - 1];
  useEffect(() => {
    end.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns.length, last?.status]);

  // Grow the text box with its content, up to a few lines.
  useEffect(() => {
    const el = input.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [draft]);

  function send(text: string) {
    const q = text.trim();
    if (q.length < 3) {
      setError("Enter a question of at least 3 characters.");
      return;
    }
    if (busy) return;
    ask(q);
    setDraft("");
    setError(undefined);
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    send(draft);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send(draft);
    }
  }

  return (
    <div className="mx-auto flex min-h-[calc(100vh-12rem)] max-w-3xl flex-col">
      {turns.length > 0 && (
        <div className="mb-6 flex items-center justify-between gap-3 border-b border-line pb-3 text-xs text-muted">
          <span>{turns.length === 1 ? "1 question" : `${turns.length} questions`} · this thread is not saved</span>
          <button type="button" onClick={() => { reset(); input.current?.focus(); }}
            className="flex items-center gap-1.5 rounded-full px-3 py-1.5 ring-1 ring-line-strong transition-colors hover:text-fg">
            <Icon name="newChat" className="h-3.5 w-3.5" /> New chat
          </button>
        </div>
      )}

      {turns.length === 0 ? (
        <EmptyState onPick={send} />
      ) : (
        <div className="flex-1 space-y-10" aria-live="polite">
          {turns.map((t) => <TurnView key={t.id} turn={t} onRetry={() => retry(t.id)} />)}
        </div>
      )}
      <div ref={end} />

      <div className="sticky bottom-24 mt-8 md:bottom-6">
        {error && <div className="mb-2"><ErrorNote message={error} /></div>}
        <form onSubmit={submit}
          className="flex items-end gap-2 rounded-3xl border border-line-strong bg-surface/90 p-2 pl-5 float-shadow backdrop-blur-xl focus-within:border-accent/60">
          <label htmlFor="question" className="sr-only">Question</label>
          <textarea id="question" ref={input} rows={1} value={draft} maxLength={MAX_CHARS} onKeyDown={onKeyDown}
            onChange={(e) => { setDraft(e.target.value); setError(undefined); }}
            placeholder={turns.length ? "Ask another question" : "How do I drain a node?"}
            className="max-h-40 flex-1 resize-none self-center bg-transparent py-1.5 text-[15px] leading-relaxed text-fg outline-none placeholder:text-faint" />
          <button type="submit" aria-label="Ask" disabled={busy}
            className="rounded-full bg-accent p-2.5 text-ink transition-colors hover:bg-accent-strong disabled:opacity-60">
            <Icon name="send" className="h-4 w-4" />
          </button>
        </form>
        <p className="mt-2 flex flex-wrap justify-between gap-x-4 px-3 text-[11px] text-faint">
          <span>Each question is answered on its own: earlier messages are not sent with it.</span>
          <span className="tabular-nums">
            {draft.length > MAX_CHARS - 100 && `${draft.length}/${MAX_CHARS} · `}
            {quota ? `${quota.remaining_today} live answers left today` : ""}
          </span>
        </p>
      </div>
    </div>
  );
}
