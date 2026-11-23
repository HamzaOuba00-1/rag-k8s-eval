import { Link } from "react-router";
import { api } from "../api/client";
import { ErrorNote, QuotaCard, ScoreRing } from "../components/Cards";
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
    <div className="space-y-14">
      <section className="grid items-end gap-6 md:grid-cols-[1.5fr_1fr]">
        <div>
          <p className="eyebrow">Kubernetes documentation · grounded answers</p>
          <h1 className="mt-4 font-display text-4xl leading-[1.05] md:text-[56px]">
            Answers you can <span className="text-accent">check</span>.
          </h1>
          <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-muted">
            Every answer cites the pages it comes from and says so when the docs don't cover it. Every score on this
            site was measured on 53 evaluation questions, with confidence intervals.
          </p>
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link to="/ask" className="btn-primary">
              Ask a question <Icon name="arrow" className="h-4 w-4" />
            </Link>
            <Link to="/scores" className="rounded-full px-4 py-2 text-sm text-muted ring-1 ring-line-strong transition-colors hover:text-fg">
              See the evaluation
            </Link>
          </div>
        </div>
        <QuotaCard quota={quota.data} />
      </section>

      <section>
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-xl">Measured scores</h2>
          <span className="eyebrow">best configuration</span>
        </div>
        {scores.error && <ErrorNote message={scores.error} />}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <ScoreRing label="Retrieval MRR" value={best?.mrr?.mean ?? null} display={best?.mrr?.mean.toFixed(2)}
            sub={`${best?.n ?? 0} answerable questions`} />
          <ScoreRing label="Correct refusals" value={gen?.refusal_rate?.mean ?? null}
            display={`${refusals}/${gen?.n_impossible}`} sub="questions the docs can't answer" />
          <ScoreRing label="Faithfulness" value={gen?.faithfulness?.mean ?? null}
            display={gen?.faithfulness?.mean.toFixed(2)} sub="LLM judge" />
          <ScoreRing label="Answer relevancy" value={gen?.answer_relevancy?.mean ?? null}
            display={gen?.answer_relevancy?.mean.toFixed(2)} sub="LLM judge" />
        </div>
      </section>

      <section>
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-xl">Try an example question</h2>
          <Link to="/scores" className="text-xs text-muted underline decoration-line-strong underline-offset-4 hover:text-fg">
            How these are scored
          </Link>
        </div>
        {examples.error && <ErrorNote message={examples.error} />}
        <div className="grid gap-3 md:grid-cols-2">
          {[...factual, ...(impossible ? [impossible] : [])].map((e) => (
            <Link key={e.id} to={`/ask?q=${encodeURIComponent(e.question)}`}
              className="panel group flex items-center gap-4 p-4 text-sm transition-colors hover:border-line-strong hover:bg-raised">
              <span className="flex-1 leading-relaxed">{e.question}</span>
              {e.category !== "factual" && (
                <span className="rounded-full bg-warn/10 px-2 py-0.5 text-[11px] text-warn ring-1 ring-warn/20">not in the docs</span>
              )}
              <Icon name="arrow" className="h-4 w-4 text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
