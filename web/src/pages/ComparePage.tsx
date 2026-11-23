import { type FormEvent, useState } from "react";
import { useSearchParams } from "react-router";
import { api, METHODS, type Method, type SearchCell, STRATEGIES } from "../api/client";
import { Card, ErrorNote } from "../components/Cards";
import { LABELS } from "../lib/format";
import { useApi } from "../lib/useApi";

const BEST = { strategy: "fixed", method: "hybrid_rerank" };

function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return "–";
  return `${value < 10 ? value.toFixed(1) : value.toFixed(0)} ms`;
}

function CellView({ cell, sourceUrl }: { cell: SearchCell; sourceUrl?: string | null }) {
  const best = cell.strategy === BEST.strategy && cell.method === BEST.method;
  if (!cell.available) {
    return (
      <div className="rounded-xl border border-dashed border-line p-3 text-xs leading-relaxed text-faint">
        Reranking runs offline: available for the evaluation questions only.
      </div>
    );
  }
  return (
    <div className={`rounded-xl border p-3 text-xs ${best ? "border-accent/70 bg-accent/5 shadow-[0_0_24px_rgb(193_154_107/0.12)]" : "border-line bg-surface/60"}`}>
      <ol className="space-y-1">
        {cell.sources.slice(0, 3).map((s) => (
          <li key={s.chunk_id} className={`truncate ${sourceUrl && s.url === sourceUrl ? "font-medium text-ok" : ""}`}>
            {s.n}. {s.title}
          </li>
        ))}
      </ol>
      <p className="mt-2 text-muted">
        {cell.origin === "live" ? ms(cell.milliseconds) : "evaluation run"}
        {sourceUrl !== undefined && sourceUrl !== null &&
          ` · ${cell.source_rank ? `source page at ${cell.source_rank}` : "source page not in context"}`}
      </p>
    </div>
  );
}

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const [draft, setDraft] = useState(query);
  const [methods, setMethods] = useState<Method[]>(["dense", "hybrid", "hybrid_rerank"]);
  const [inputError, setInputError] = useState<string>();
  const examples = useApi((s) => api.examples(s));
  const result = useApi(
    (s) => (query ? api.search({ query, strategies: STRATEGIES, methods }, s) : Promise.resolve(undefined)),
    [query, methods.join()],
  );

  function submit(e: FormEvent) {
    e.preventDefault();
    if (draft.trim().length < 3) {
      setInputError("Enter a question of at least 3 characters.");
      return;
    }
    setParams({ q: draft.trim() });
  }

  function toggle(m: Method) {
    setMethods((cur) => (cur.includes(m) ? (cur.length > 1 ? cur.filter((x) => x !== m) : cur) : METHODS.filter((x) => cur.includes(x) || x === m)));
  }

  const data = result.data;
  return (
    <div className="space-y-4">
      <div>
        <p className="eyebrow">Retrieval lab</p>
        <h1 className="mt-3 font-display text-3xl md:text-[44px]">Compare retrieval</h1>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
          Three chunking strategies, four retrieval methods: see which excerpts each one puts in the 5,000-character
          context, and where the source page lands.
        </p>
      </div>
      <form onSubmit={submit} className="flex gap-2">
        <label htmlFor="query" className="sr-only">Question</label>
        <input id="query" value={draft} onChange={(e) => { setDraft(e.target.value); setInputError(undefined); }}
          placeholder={examples.data?.[0]?.question ?? "How do I drain a node?"}
          className="flex-1 rounded-full border border-line-strong bg-surface px-5 py-2.5 text-[15px] text-fg outline-none placeholder:text-faint focus:border-accent/60" />
        <button className="btn-primary">Compare</button>
      </form>
      {inputError && <ErrorNote message={inputError} />}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="eyebrow mr-1">Methods</span>
        {METHODS.map((m) => (
          <button key={m} type="button" aria-pressed={methods.includes(m)} onClick={() => toggle(m)}
            className={`rounded-full px-3 py-1.5 transition-colors ${methods.includes(m) ? "bg-accent/10 text-accent ring-1 ring-accent/40" : "text-muted ring-1 ring-line-strong hover:text-fg"}`}>
            {LABELS[m]}
          </button>
        ))}
      </div>

      {result.error && <ErrorNote message={result.error} />}
      {query && result.loading && <p className="text-sm text-muted" role="status">Searching…</p>}
      {data && !result.loading && (
        <Card>
          <div className="space-y-4 md:hidden">
            {STRATEGIES.map((s) => (
              <section key={s}>
                <h2 className="eyebrow mb-2">{s}</h2>
                <div className="space-y-2">
                  {methods.map((m) => {
                    const cell = data.cells.find((c) => c.strategy === s && c.method === m);
                    return cell ? (
                      <div key={m}>
                        <p className="mb-1 text-[11px] text-muted">{LABELS[m]}</p>
                        <CellView cell={cell} sourceUrl={data.evaluation_question ? data.source_url : undefined} />
                      </div>
                    ) : null;
                  })}
                </div>
              </section>
            ))}
          </div>
          <div className="hidden overflow-x-auto md:block">
            <div className="grid min-w-[640px] gap-2 p-0.5" style={{ gridTemplateColumns: `90px repeat(${methods.length}, minmax(0, 1fr))` }}>
              <div />
              {methods.map((m) => <div key={m} className="eyebrow">{LABELS[m]}</div>)}
              {STRATEGIES.map((s) => (
                <div key={s} className="contents">
                  <div className="eyebrow self-center">{s}</div>
                  {methods.map((m) => {
                    const cell = data.cells.find((c) => c.strategy === s && c.method === m);
                    return cell ? <CellView key={m} cell={cell} sourceUrl={data.evaluation_question ? data.source_url : undefined} /> : <div key={m} />;
                  })}
                </div>
              ))}
            </div>
          </div>
          <p className="mt-3 text-xs text-muted">
            {data.evaluation_question
              ? "Evaluation question: green is the page it was written from. "
              : "Free question: no source page to check against. "}
            The framed cell is the best configuration of the evaluation. Live cells are timed on the server; reranked
            cells come from the evaluation run.
          </p>
        </Card>
      )}
      {!query && (
        <p className="text-sm text-muted">
          Type a question, or pick an evaluation question to see where its source page lands in each context.
        </p>
      )}
    </div>
  );
}
