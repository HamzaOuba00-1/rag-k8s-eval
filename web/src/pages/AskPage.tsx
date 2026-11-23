import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { api, type AskResponse } from "../api/client";
import { AnswerText, SourceList } from "../components/Answer";
import { Logo } from "../components/Brand";
import { Card, ErrorNote } from "../components/Cards";
import { Icon } from "../components/Icon";
import { configLabel } from "../lib/format";
import { useApi } from "../lib/useApi";

function plural(n: number, word: string): string {
  return `${n.toLocaleString("en")} ${word}${n === 1 ? "" : "s"}`;
}

export function AskPage() {
  const [params, setParams] = useSearchParams();
  const asked = params.get("q") ?? "";
  const [draft, setDraft] = useState(asked);
  const [result, setResult] = useState<AskResponse>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState(0);
  const [copied, setCopied] = useState(false);
  const quota = useApi((s) => api.quota(s), [result]);
  const examples = useApi((s) => api.examples(s));
  const controller = useRef<AbortController | null>(null);

  const run = useCallback(async (question: string) => {
    controller.current?.abort();
    const c = new AbortController();
    controller.current = c;
    setLoading(true);
    setError(undefined);
    setTab(0);
    try {
      setResult(await api.ask(question, c.signal));
    } catch (e) {
      if (!c.signal.aborted) setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      if (!c.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    setDraft(asked);
    if (asked.trim().length >= 3) void run(asked);
    else setResult(undefined);
    return () => controller.current?.abort();
  }, [asked, run]);

  function submit(e: FormEvent) {
    e.preventDefault();
    const q = draft.trim();
    if (q.length < 3) {
      setError("Enter a question of at least 3 characters.");
      return;
    }
    setParams({ q });
  }

  const answer = result?.answers[tab];
  const remaining = quota.data?.remaining_today;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_280px]">
      <div className="flex min-h-[60vh] flex-col gap-4">
        <div className="flex items-center justify-between text-xs text-muted">
          <span>{result ? configLabel(result.config) : "Live: fixed · hybrid"}</span>
          {remaining !== undefined && <span>{remaining} live answers left today</span>}
        </div>

        {!asked && (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 py-10 text-center">
            <Logo size={88} className="drop-shadow-[0_0_28px_rgb(193_154_107/0.25)]" />
            <h1 className="font-display text-3xl md:text-4xl">Ask the Kubernetes docs</h1>
            <p className="text-sm text-muted">Answers cite the pages they come from.</p>
            <div className="mt-4 grid w-full max-w-lg gap-3 sm:grid-cols-2">
              {examples.data?.[0] && (
                <Link to={`/ask?q=${encodeURIComponent(examples.data[0].question)}`}
                  className="panel p-4 text-left text-sm transition-colors hover:border-line-strong hover:bg-raised">
                  <span className="eyebrow">Example question</span>
                  <span className="mt-2 block leading-relaxed">{examples.data[0].question}</span>
                </Link>
              )}
              <Link to="/compare" className="panel p-4 text-left text-sm transition-colors hover:border-line-strong hover:bg-raised">
                <span className="eyebrow">Compare retrieval</span>
                <span className="mt-2 block leading-relaxed">See what each chunking strategy finds</span>
              </Link>
            </div>
          </div>
        )}

        {asked && (
          <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 text-[15px] text-fg ring-1 ring-line-strong">{asked}</div>
        )}

        {loading && <p className="text-sm text-muted" role="status">Retrieving and answering…</p>}
        {error && <ErrorNote message={error} />}

        {result && !loading && (
          <>
            <div className="ml-[42px] space-y-3 empty:hidden">
            {result.message && (
              <p className={`rounded-xl px-5 py-3 text-sm ${result.demo_mode ? "bg-warn/10 text-warn ring-1 ring-warn/25" : "bg-accent/10 text-muted ring-1 ring-accent/20"}`}>
                {result.message}
              </p>
            )}
            {result.answers.length > 1 && (
              <div role="tablist" className="flex gap-2">
                {result.answers.map((a, i) => (
                  <button key={a.model} role="tab" aria-selected={tab === i} onClick={() => setTab(i)}
                    className={`rounded-full px-3 py-1.5 text-xs transition-colors ${tab === i ? "bg-accent text-ink" : "text-muted ring-1 ring-line-strong hover:text-fg"}`}>
                    {a.model} · {a.origin}
                  </button>
                ))}
              </div>
            )}
            </div>
            {answer && (
              <div className="flex gap-3">
                <Logo size={30} className="mt-1 self-start" />
                <Card className="flex-1 rounded-tl-md">
                  <AnswerText text={answer.text} sources={result.sources} />
                  <div className="mt-4 flex items-center gap-4 border-t border-line pt-3 text-xs text-muted">
                    <button className="flex items-center gap-1 hover:text-fg"
                      onClick={() => void navigator.clipboard.writeText(answer.text).then(() => setCopied(true))}>
                      <Icon name="copy" className="h-4 w-4" /> {copied ? "Copied" : "Copy"}
                    </button>
                    <Link className="flex items-center gap-1 hover:text-fg" to={`/compare?q=${encodeURIComponent(result.question)}`}>
                      <Icon name="compare" className="h-4 w-4" /> Compare retrieval
                    </Link>
                    <span className="ml-auto">{answer.model}</span>
                  </div>
                </Card>
              </div>
            )}
          </>
        )}

        <form onSubmit={submit} className="sticky bottom-24 mt-auto flex items-center gap-2 rounded-full border border-line-strong bg-surface/90 p-1.5 pl-5 float-shadow backdrop-blur-xl focus-within:border-accent/60 md:bottom-6">
          <label htmlFor="question" className="sr-only">Question</label>
          <input id="question" value={draft} maxLength={500} onChange={(e) => { setDraft(e.target.value); setError(undefined); }}
            placeholder="How do I drain a node?" className="flex-1 bg-transparent text-[15px] text-fg outline-none placeholder:text-faint" />
          <button type="submit" aria-label="Ask" disabled={loading}
            className="rounded-full bg-accent p-2.5 text-ink transition-colors hover:bg-accent-strong disabled:opacity-60">
            <Icon name="send" className="h-4 w-4" />
          </button>
        </form>
      </div>

      <aside className="space-y-3">
        <h2 className="eyebrow">Sources</h2>
        {result ? (
          <>
            <SourceList sources={result.sources} cited={answer?.cited ?? []} />
            <p className="text-xs text-muted">
              {plural(result.sources.reduce((n, s) => n + s.text.length, 0), "character")} of context ·{" "}
              {plural(result.sources.length, "excerpt")}
            </p>
          </>
        ) : (
          <p className="text-sm text-muted">The excerpts given to the model appear here.</p>
        )}
      </aside>
    </div>
  );
}
