import type { ReactNode } from "react";
import { api, type Estimate, type GenerationRow, type JudgeAgreement } from "../api/client";
import { Card, ErrorNote } from "../components/Cards";
import { ci, configLabel, num, signed, significant } from "../lib/format";
import { useApi } from "../lib/useApi";

function CardHead({ title, sub, note }: { title: string; sub?: ReactNode; note?: ReactNode }) {
  return (
    <header className="mb-5">
      <h2 className="font-display text-xl">{title}</h2>
      {sub && <p className="mt-1 text-sm text-muted">{sub}</p>}
      {note && <p className="mt-0.5 text-xs text-faint">{note}</p>}
    </header>
  );
}

/** Mean on the first line, its 95% interval below in small type: columns stay aligned. */
function Est({ e }: { e: Estimate | null | undefined }) {
  if (!e) return <span className="text-faint">pending</span>;
  return (
    <>
      <span className="block">{num(e.mean)}</span>
      <span className="block text-[11px] text-faint">[{num(e.low)}, {num(e.high)}]</span>
    </>
  );
}

function Count({ rate, n }: { rate: Estimate | null | undefined; n: number }) {
  return rate ? <>{Math.round(rate.mean * n)}/{n}</> : <span className="text-faint">pending</span>;
}

const TH = "py-2 align-bottom text-[11px] font-normal tracking-wide text-faint uppercase";

function GenRows({ rows }: { rows: (GenerationRow | null | undefined)[] }) {
  const cols = rows.filter((r): r is GenerationRow => !!r);
  const lines: { label: string; sub?: string; cell: (r: GenerationRow) => ReactNode }[] = [
    { label: "Citation coverage", cell: (r) => <Est e={r.citation_coverage} /> },
    { label: "Correct refusals", cell: (r) => <Count rate={r.refusal_rate} n={r.n_impossible} /> },
    { label: "False refusals", cell: (r) => <Count rate={r.false_refusal_rate} n={r.n_factual} /> },
    { label: "Faithfulness", sub: "LLM judge", cell: (r) => <Est e={r.faithfulness} /> },
    { label: "Answer relevancy", sub: "LLM judge", cell: (r) => <Est e={r.answer_relevancy} /> },
    { label: "Seconds per answer", cell: (r) => num(r.seconds_mean) },
  ];
  return (
    <table className="w-full table-fixed text-sm">
      <colgroup>
        <col className="w-[42%]" />
        {cols.map((r) => <col key={r.model} />)}
      </colgroup>
      <thead>
        <tr>
          <th className={`${TH} text-left`}>Metric</th>
          {cols.map((r) => (
            <th key={r.model} className={`${TH} pl-3 text-right`}>
              <span className="block break-words">{r.model}</span>
              <span className="block normal-case">{r.provider}</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {lines.map(({ label, sub, cell }) => (
          <tr key={label} className="border-t border-line">
            <td className="py-2.5 align-top">
              {label}
              {sub && <span className="block text-[11px] text-faint">{sub}</span>}
            </td>
            {cols.map((r) => (
              <td key={r.model} className="py-2.5 pl-3 text-right align-top tabular-nums">{cell(r)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Agreement({ a }: { a: JudgeAgreement }) {
  const human = a.annotator === "human";
  return (
    <>
      <table className="w-full table-fixed text-xs">
        <colgroup>
          <col className="w-[30%]" />
          <col className="w-[24%]" />
          <col className="w-[13%]" />
          <col className="w-[13%]" />
          <col className="w-[20%]" />
        </colgroup>
        <thead>
          <tr>
            <th className={`${TH} text-left`}>Metric</th>
            <th className={`${TH} text-right`}>Spearman</th>
            <th className={`${TH} text-right`}>MAE</th>
            <th className={`${TH} text-right`}>Bias</th>
            <th className={`${TH} text-right`}>Means</th>
          </tr>
        </thead>
        <tbody>
          {a.metrics.map((m) => (
            <tr key={m.metric} className="border-t border-line">
              <td className="py-2 align-top">{m.metric.replace("_", " ")}</td>
              <td className="py-2 text-right align-top tabular-nums">
                <span className="block">{num(m.spearman)}</span>
                {m.spearman_ci?.length === 2 && (
                  <span className="block text-[11px] text-faint">[{num(m.spearman_ci[0])}, {num(m.spearman_ci[1])}]</span>
                )}
              </td>
              <td className="py-2 text-right align-top tabular-nums">{num(m.mae)}</td>
              <td className="py-2 text-right align-top tabular-nums">
                {m.bias === null ? "–" : `${m.bias >= 0 ? "+" : "−"}${Math.abs(m.bias).toFixed(2)}`}
              </td>
              <td className="py-2 text-right align-top tabular-nums">{num(m.annotator_mean)} / {num(m.judge_mean)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!human && <p className="mt-3 text-xs text-warn">Agreement between two models, not a human validation.</p>}
      <p className="mt-2 text-xs leading-relaxed text-muted">
        Means: annotator / judge. Bias = judge minus annotator. Levels agree, but almost every answer is rated good
        (ceiling effect), so whether the judge can rank answers is not tested here.
      </p>
    </>
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
  const agreement = gen?.judge_complete ? gen.judge_agreement : null;

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Evaluation</p>
        <h1 className="mt-3 font-display text-3xl md:text-[44px]">Scores</h1>
        <p className="mt-3 text-sm text-muted">
          Measured results only, with 95% bootstrap intervals. Bundle built {data.measured_on?.slice(0, 10) ?? "–"}.
        </p>
      </div>

      {/* Two independent columns, aligned at the top: no card is stretched to its neighbour's height. */}
      <div className="grid items-start gap-4 lg:grid-cols-[1.15fr_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHead title="Retrieval" sub="MRR of the source page"
              note={`${rows[0]?.n ?? 0} answerable questions, 5,000-character context budget`} />
            <ul className="space-y-2.5">
              {rows.map((r) => {
                const isBest = r.config === data.best_config;
                const isLive = r.config === data.live_config;
                return (
                  <li key={r.config} className="grid grid-cols-[160px_minmax(0,1fr)_36px] items-center gap-3 text-xs">
                    <span className="flex items-center gap-1.5 truncate">
                      {configLabel(r.config)}
                      {isLive && <span className="rounded bg-accent/10 px-1 text-[10px] text-accent">live</span>}
                    </span>
                    <span className="h-1.5 rounded-full bg-line" title={`${ci(r.mrr)}${r.mrr_diff_vs_best ? `; vs best ${signed(r.mrr_diff_vs_best)}` : ""}`}>
                      <span className={`block h-1.5 rounded-full ${isBest ? "bg-gradient-to-r from-accent-deep to-accent-strong" : "bg-accent/35"}`} style={{ width: `${(r.mrr?.mean ?? 0) * 100}%` }} />
                    </span>
                    <span className="text-right tabular-nums">{num(r.mrr?.mean)}</span>
                  </li>
                );
              })}
            </ul>
            <dl className="mt-5 space-y-1 border-t border-line pt-4 text-xs text-muted">
              {rows.filter((r) => r.config === data.best_config || r.config === data.live_config).map((r) => (
                <div key={r.config}>
                  <dt className="inline font-medium text-fg">{configLabel(r.config)}: </dt>
                  <dd className="inline">
                    MRR {ci(r.mrr)}
                    {r.mrr_diff_vs_best &&
                      `, vs best ${signed(r.mrr_diff_vs_best)} (${significant(r.mrr_diff_vs_best) ? "significant" : "not significant"})`}
                  </dd>
                </div>
              ))}
            </dl>
          </Card>

          <Card>
            <CardHead title="This deployment" />
            <p className="-mt-2 text-sm leading-relaxed text-muted">
              Live answers use {configLabel(data.live_config)}: the reranker takes seconds per query on one vCPU. The
              ONNX embedder gives the same context as the evaluation on{" "}
              {fp32 === undefined ? "–" : `${Math.round(fp32 * 100)}%`} of the questions.
            </p>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHead title="Generation" sub={configLabel(data.best_config)} />
            {gen && bestApp ? <GenRows rows={[bestApp, gen.reference]} /> : <p className="text-sm text-faint">pending</p>}
            <p className="mt-3 text-xs text-muted">
              Times come from different hardware (Groq vs a Kaggle T4): not a model speed comparison.
            </p>
          </Card>

          <Card>
            <CardHead title="LLM judge"
              sub={agreement ? (
                <>
                  {agreement.n_items} answers labelled blind (0 / 0.5 / 1) by{" "}
                  {agreement.annotator === "human" ? "a human" : <code className="text-xs">{agreement.annotator}</code>}.
                </>
              ) : undefined} />
            {agreement ? (
              <Agreement a={agreement} />
            ) : (
              <p className="-mt-2 text-sm text-muted">pending: the judge run and its agreement check are not complete.</p>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
