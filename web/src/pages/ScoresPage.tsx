import { api, type GenerationRow } from "../api/client";
import { Card, ErrorNote } from "../components/Cards";
import { ci, configLabel, num, signed, significant } from "../lib/format";
import { useApi } from "../lib/useApi";

function GenRows({ rows }: { rows: (GenerationRow | null | undefined)[] }) {
  const cols = rows.filter((r): r is GenerationRow => !!r);
  const lines: [string, (r: GenerationRow) => string][] = [
    ["Citation coverage", (r) => ci(r.citation_coverage)],
    ["Correct refusals", (r) => (r.refusal_rate ? `${Math.round(r.refusal_rate.mean * r.n_impossible)}/${r.n_impossible}` : "pending")],
    ["False refusals", (r) => (r.false_refusal_rate ? `${Math.round(r.false_refusal_rate.mean * r.n_factual)}/${r.n_factual}` : "pending")],
    ["Faithfulness", (r) => ci(r.faithfulness)],
    ["Answer relevancy", (r) => ci(r.answer_relevancy)],
    ["Seconds per answer", (r) => num(r.seconds_mean)],
  ];
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-[11px] tracking-wide text-faint uppercase">
          <th className="py-1 font-normal" />
          {cols.map((r) => <th key={r.model} className="py-1 font-normal">{r.model} <span className="text-faint">({r.provider})</span></th>)}
        </tr>
      </thead>
      <tbody>
        {lines.map(([label, f]) => (
          <tr key={label} className="border-t border-line">
            <td className="py-1.5">{label}</td>
            {cols.map((r) => <td key={r.model} className={`py-1.5 tabular-nums ${f(r) === "pending" ? "text-faint" : ""}`}>{f(r)}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ScoresPage() {
  const { data, error, loading } = useApi((s) => api.scores(s));
  if (error) return <ErrorNote message={error} />;
  if (loading || !data) return <p className="text-sm text-muted" role="status">Loading scores…</p>;

  const rows = data.retrieval.filter((r) => r.subset === "all").sort((a, b) => (b.mrr?.mean ?? 0) - (a.mrr?.mean ?? 0));
  const gen = data.generation;
  const bestApp = gen?.configs.find((c) => c.config === data.best_config);
  const fp32 = (data.onnx_parity as { configs?: Record<string, Record<string, { identical_context?: number }>> } | null)
    ?.configs?.fixed_hybrid?.fp32?.identical_context;

  return (
    <div className="space-y-6">
      <div>
        <p className="eyebrow">Evaluation</p>
        <h1 className="mt-3 font-display text-4xl tracking-tight md:text-5xl">Scores</h1>
        <p className="text-sm text-muted">
          Measured results only, with 95% bootstrap intervals. Bundle built {data.measured_on?.slice(0, 10) ?? "–"}.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <Card>
          <h2 className="font-display text-2xl">Retrieval</h2>
          <p className="text-sm text-muted">MRR of the source page</p>
          <p className="mb-5 text-xs text-faint">{rows[0]?.n ?? 0} answerable questions, 5,000-character context budget</p>
          <ul className="space-y-2.5">
            {rows.map((r) => {
              const isBest = r.config === data.best_config;
              const isLive = r.config === data.live_config;
              return (
                <li key={r.config} className="grid grid-cols-[150px_minmax(0,1fr)_44px] items-center gap-2 text-xs">
                  <span className="truncate">
                    {configLabel(r.config)}
                    {isLive && <span className="ml-1 rounded bg-accent/10 px-1 text-[10px] text-accent">live</span>}
                  </span>
                  <span className="h-1.5 rounded-full bg-line" title={`${ci(r.mrr)}${r.mrr_diff_vs_best ? `; vs best ${signed(r.mrr_diff_vs_best)}` : ""}`}>
                    <span className={`block h-1.5 rounded-full ${isBest ? "bg-gradient-to-r from-accent-deep to-accent-strong" : "bg-accent/35"}`} style={{ width: `${(r.mrr?.mean ?? 0) * 100}%` }} />
                  </span>
                  <span className="tabular-nums">{num(r.mrr?.mean)}</span>
                </li>
              );
            })}
          </ul>
          <dl className="mt-4 space-y-1 text-xs text-muted">
            {rows.filter((r) => r.config === data.best_config || r.config === data.live_config).map((r) => (
              <div key={r.config}>
                <dt className="inline font-medium">{configLabel(r.config)}: </dt>
                <dd className="inline">
                  MRR {ci(r.mrr)}
                  {r.mrr_diff_vs_best &&
                    `, vs best ${signed(r.mrr_diff_vs_best)} (${significant(r.mrr_diff_vs_best) ? "significant" : "not significant"})`}
                </dd>
              </div>
            ))}
          </dl>
        </Card>

        <div className="space-y-4">
          <Card>
            <h2 className="font-display text-2xl">Generation</h2>
            <p className="mb-3 text-sm text-muted">{configLabel(data.best_config)}</p>
            {gen && bestApp ? <GenRows rows={[bestApp, gen.reference]} /> : <p className="text-sm text-faint">pending</p>}
            <p className="mt-2 text-xs text-muted">
              Times come from different hardware (Groq vs a Kaggle T4): not a model speed comparison.
            </p>
          </Card>
          <Card>
            <h2 className="mb-2 font-display text-2xl">LLM judge</h2>
            <p className="text-sm text-muted">
              {gen?.judge_complete
                ? `Judge vs human ratings: ${JSON.stringify(gen.judge_agreement)}`
                : "pending: the judge run and its check against human ratings are not complete."}
            </p>
          </Card>
          <Card>
            <h2 className="mb-2 font-display text-2xl">This deployment</h2>
            <p className="text-sm text-muted">
              Live answers use {configLabel(data.live_config)}: the reranker takes seconds per query on one vCPU. The
              ONNX embedder gives the same context as the evaluation on{" "}
              {fp32 === undefined ? "–" : `${Math.round(fp32 * 100)}%`} of the questions.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
