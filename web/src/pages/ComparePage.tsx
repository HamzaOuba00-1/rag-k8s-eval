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
      <div className="rounded-xl bg-slate-50 p-3 text-xs text-slate-400 ring-1 ring-slate-200">
        Reranking runs offline: available for the evaluation questions only.
      </div>
    );
  }
  return (
    <div className={`rounded-xl bg-white p-3 text-xs ${best ? "ring-2 ring-brand-400" : "ring-1 ring-slate-200"}`}>
      <ol className="space-y-1">
        {cell.sources.slice(0, 3).map((s) => (
          <li key={s.chunk_id} className={`truncate ${sourceUrl && s.url === sourceUrl ? "font-medium text-ok" : ""}`}>
            {s.n}. {s.title}
          </li>
        ))}
      </ol>
      <p className="mt-2 text-slate-500">
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
      <h1 className="text-xl font-semibold">Compare retrieval</h1>
      <form onSubmit={submit} className="flex gap-2">
        <label htmlFor="query" className="sr-only">Question</label>
        <input id="query" value={draft} onChange={(e) => { setDraft(e.target.value); setInputError(undefined); }}
          placeholder={examples.data?.[0]?.question ?? "How do I drain a node?"}
          className="flex-1 rounded-full bg-white px-4 py-2 text-sm shadow-sm ring-1 ring-slate-300 outline-none" />
        <button className="rounded-full bg-brand-600 px-4 py-2 text-sm text-white">Compare</button>
      </form>
      {inputError && <ErrorNote message={inputError} />}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-500">Methods</span>
        {METHODS.map((m) => (
          <button key={m} type="button" aria-pressed={methods.includes(m)} onClick={() => toggle(m)}
            className={`rounded-full px-3 py-1 ${methods.includes(m) ? "bg-brand-50 text-brand-800 ring-1 ring-brand-200" : "ring-1 ring-slate-300 text-slate-500"}`}>
            {LABELS[m]}
          </button>
        ))}
      </div>

      {result.error && <ErrorNote message={result.error} />}
      {query && result.loading && <p className="text-sm text-slate-500" role="status">Searching…</p>}
      {data && !result.loading && (
        <Card>
          <div className="space-y-4 md:hidden">
            {STRATEGIES.map((s) => (
              <section key={s}>
                <h2 className="mb-2 text-xs font-semibold text-slate-500">{s}</h2>
                <div className="space-y-2">
                  {methods.map((m) => {
                    const cell = data.cells.find((c) => c.strategy === s && c.method === m);
                    return cell ? (
                      <div key={m}>
                        <p className="mb-1 text-[11px] text-slate-500">{LABELS[m]}</p>
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
              {methods.map((m) => <div key={m} className="text-xs font-semibold text-slate-500">{LABELS[m]}</div>)}
              {STRATEGIES.map((s) => (
                <div key={s} className="contents">
                  <div className="self-center text-xs font-semibold text-slate-500">{s}</div>
                  {methods.map((m) => {
                    const cell = data.cells.find((c) => c.strategy === s && c.method === m);
                    return cell ? <CellView key={m} cell={cell} sourceUrl={data.evaluation_question ? data.source_url : undefined} /> : <div key={m} />;
                  })}
                </div>
              ))}
            </div>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            {data.evaluation_question
              ? "Evaluation question: green is the page it was written from. "
              : "Free question: no source page to check against. "}
            The framed cell is the best configuration of the evaluation. Live cells are timed on the server; reranked
            cells come from the evaluation run.
          </p>
        </Card>
      )}
      {!query && (
        <p className="text-sm text-slate-500">
          Type a question, or pick an evaluation question to see where its source page lands in each context.
        </p>
      )}
    </div>
  );
}
