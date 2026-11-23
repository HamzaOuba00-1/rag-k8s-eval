import { Link } from "react-router";
import { api } from "../api/client";
import { Card, ErrorNote, QuotaCard, ScoreRing } from "../components/Cards";
import { Icon } from "../components/Icon";
import { useApi } from "../lib/useApi";

export function HomePage() {
  const quota = useApi((s) => api.quota(s));
  const scores = useApi((s) => api.scores(s));
  const examples = useApi((s) => api.examples(s));

  const sc = scores.data;
  const best = sc?.retrieval.find((r) => r.subset === "all" && r.config === sc.best_config);
  const gen = sc?.generation?.configs.find((c) => c.config === sc.best_config) ?? null;
  const refusals = gen?.refusal_rate ? Math.round(gen.refusal_rate.mean * gen.n_impossible) : null;

  const factual = examples.data?.filter((e) => e.category === "factual").slice(0, 3) ?? [];
  const impossible = examples.data?.find((e) => e.category !== "factual");

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-[1.4fr_1fr]">
        <Card className="bg-brand-50 ring-brand-100">
          <h1 className="text-xl font-semibold text-brand-900">Ask the Kubernetes docs</h1>
          <p className="mt-1 text-sm text-brand-800">
            Answers cite the pages they come from, and say so when the docs don't cover it. Every score on this site
            was measured on 53 evaluation questions.
          </p>
          <Link to="/ask" className="mt-4 inline-flex items-center gap-2 rounded-full bg-brand-600 px-4 py-2 text-sm text-white">
            Ask a question <Icon name="arrow" className="h-4 w-4" />
          </Link>
        </Card>
        <QuotaCard quota={quota.data} />
      </div>

      <section>
        <h2 className="mb-3 text-sm font-semibold">Measured scores</h2>
        {scores.error && <ErrorNote message={scores.error} />}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <ScoreRing label="Retrieval MRR" value={best?.mrr?.mean ?? null} display={best?.mrr?.mean.toFixed(2)}
            sub={`${best?.n ?? 0} questions`} />
          <ScoreRing label="Correct refusals" value={gen?.refusal_rate?.mean ?? null}
            display={`${refusals}/${gen?.n_impossible}`} sub="unanswerable questions" />
          <ScoreRing label="Faithfulness" value={gen?.faithfulness?.mean ?? null}
            display={gen?.faithfulness?.mean.toFixed(2)} sub="LLM judge" />
          <ScoreRing label="Answer relevancy" value={gen?.answer_relevancy?.mean ?? null}
            display={gen?.answer_relevancy?.mean.toFixed(2)} sub="LLM judge" />
        </div>
      </section>

      <section>
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-sm font-semibold">Try an example question</h2>
          <Link to="/scores" className="text-xs text-brand-800 underline">How these are scored</Link>
        </div>
        {examples.error && <ErrorNote message={examples.error} />}
        <div className="grid gap-3 md:grid-cols-2">
          {[...factual, ...(impossible ? [impossible] : [])].map((e) => (
            <Link key={e.id} to={`/ask?q=${encodeURIComponent(e.question)}`}
              className="flex items-center gap-3 rounded-2xl bg-white p-4 text-sm shadow-sm ring-1 ring-slate-200/70 hover:ring-brand-200">
              <span className="flex-1">{e.question}</span>
              {e.category !== "factual" && <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[11px] text-amber-800">not in the docs</span>}
              <Icon name="arrow" className="h-4 w-4 text-slate-400" />
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
