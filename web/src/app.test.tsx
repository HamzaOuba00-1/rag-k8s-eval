import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import type { AskResponse, Quota, ScoresResponse, SearchResponse } from "./api/client";
import { routes } from "./App";

// Fake API data for the tests only.
const quota: Quota = {
  live_enabled: true, remaining_today: 62, daily_limit: 80, visitor_remaining_this_hour: 5,
  resets_at: "2026-11-24T00:00:00+00:00", reason: null,
};
const est = (mean: number) => ({ mean, low: mean - 0.1, high: mean + 0.05, n: 45 });
const scores: ScoresResponse = {
  live_config: "fixed_hybrid", best_config: "fixed_hybrid_rerank", measured_on: "2026-11-25T10:00:00+00:00",
  generation: null, onnx_parity: null,
  retrieval: [
    { subset: "all", config: "fixed_hybrid_rerank", chunking: "fixed", retrieval: "hybrid_rerank", n: 45, mrr: est(0.84),
      hit_at_1: est(0.76), hit_at_3: est(0.9), hit_at_ctx: est(0.93), context_chars: 4879, mrr_diff_vs_best: null },
    { subset: "all", config: "fixed_hybrid", chunking: "fixed", retrieval: "hybrid", n: 45, mrr: est(0.775),
      hit_at_1: est(0.7), hit_at_3: est(0.85), hit_at_ctx: est(0.9), context_chars: 4900,
      mrr_diff_vs_best: { mean: -0.066, low: -0.172, high: 0.036, n: 45 } },
    { subset: "page", config: "fixed_dense", chunking: "fixed", retrieval: "dense", n: 18, mrr: est(0.5),
      hit_at_1: null, hit_at_3: null, hit_at_ctx: null, context_chars: 4800, mrr_diff_vs_best: null },
  ],
};
const source = (n: number, title: string) => ({
  n, title, url: `https://kubernetes.io/docs/${n}/`, doc_id: `d${n}`, chunk_id: `c${n}`, text: "excerpt", score: 1,
});
const precomputed: AskResponse = {
  question: "How do I drain a node?", config: "fixed_hybrid_rerank", demo_mode: false,
  message: "Evaluation question: answers from the evaluation run, nothing generated now.",
  sources: [source(1, "Safely drain a node"), source(2, "Disruptions")],
  answers: [
    { model: "qwen3.8-27b", text: "Run `kubectl drain` [1].", is_refusal: false, cited: [1], origin: "precomputed", generated_at: null },
    { model: "qwen2.5-7b-instruct", text: "Use drain [2].", is_refusal: false, cited: [2], origin: "precomputed", generated_at: null },
  ],
  quota,
};
const search: SearchResponse = {
  query: "How do I drain a node?", budget_chars: 5000, evaluation_question: true, source_url: "https://kubernetes.io/docs/1/",
  cells: ["fixed", "structure", "semantic"].flatMap((strategy) =>
    (["dense", "hybrid", "hybrid_rerank"] as const).map((method) => ({
      strategy: strategy as "fixed", method, available: !(strategy === "semantic" && method === "hybrid_rerank"),
      origin: method === "hybrid_rerank" ? ("precomputed" as const) : ("live" as const),
      sources: [source(2, "Disruptions"), source(1, "Safely drain a node")], context_chars: 14,
      milliseconds: method === "hybrid_rerank" ? null : 12.4, source_rank: 2,
    })),
  ),
};

let askBody: AskResponse = precomputed;
beforeEach(() => {
  askBody = precomputed;
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const path = url.replace("/api", "");
    const body = { "/quota": quota, "/scores": scores, "/examples": [], "/ask": askBody, "/search": search }[path];
    return new Response(JSON.stringify(body), { status: 200 });
  }));
});

function renderAt(path: string) {
  return render(<RouterProvider router={createMemoryRouter(routes, { initialEntries: [path] })} />);
}

test("home shows measured MRR and pending generation scores", async () => {
  renderAt("/");
  expect(await screen.findByRole("img", { name: "Retrieval MRR: 0.84" })).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Citation coverage: pending" })).toBeInTheDocument();
  expect(screen.queryByRole("img", { name: /Faithfulness/ })).not.toBeInTheDocument();
  expect(await screen.findByText("of 80 left")).toBeInTheDocument();
});

test("ask shows precomputed answers per model with citation links", async () => {
  const user = userEvent.setup();
  renderAt("/ask?q=How%20do%20I%20drain%20a%20node%3F");
  expect(await screen.findByText(/answers from the evaluation run/)).toBeInTheDocument();
  const link = screen.getAllByRole("link", { name: "[1]" })[0]!;
  expect(link).toHaveAttribute("href", "https://kubernetes.io/docs/1/");
  expect(screen.getByText("kubectl drain").tagName).toBe("CODE");
  await user.click(screen.getByRole("tab", { name: /qwen2.5-7b-instruct/ }));
  expect(screen.getByText(/Use drain/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Show excerpts/ }));
  expect(screen.getByText("Retrieved, not cited")).toBeInTheDocument();
});

test("ask keeps a thread of questions and a new chat clears it", async () => {
  const user = userEvent.setup();
  renderAt("/ask");
  await user.type(screen.getByLabelText("Question"), "How do I drain a node?{Enter}");
  expect(await screen.findByText(/answers from the evaluation run/)).toBeInTheDocument();
  await user.type(screen.getByLabelText("Question"), "What is a Pod?{Enter}");
  await waitFor(() => expect(screen.getAllByRole("article")).toHaveLength(2));
  await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  expect(screen.getByText("2 questions · this thread is not saved")).toBeInTheDocument();
  await user.click(screen.getAllByRole("button", { name: /New chat/ })[0]!);
  expect(screen.queryAllByRole("article")).toHaveLength(0);
});

test("citations to a non-https URL are not rendered as links", async () => {
  askBody = { ...precomputed, sources: [{ ...source(1, "Bad"), url: "javascript:alert(1)" }] };
  renderAt("/ask?q=How%20do%20I%20drain%20a%20node%3F");
  expect(await screen.findByText(/answers from the evaluation run/)).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "[1]" })).toBeNull();
  expect(screen.getAllByText("[1]", { selector: "span" }).length).toBeGreaterThan(0);
});

test("ask shows the demo message and keeps the live sources", async () => {
  askBody = { ...precomputed, config: "fixed_hybrid", demo_mode: true, answers: [],
    message: "Today's live answers are used up. The sources below were retrieved live." };
  renderAt("/ask?q=What%20is%20a%20Pod%3F");
  expect(await screen.findByText(/used up/)).toBeInTheDocument();
  expect(screen.getByText("fixed · hybrid")).toBeInTheDocument();
  expect(screen.getAllByRole("link", { name: /Safely drain a node/ }).length).toBeGreaterThan(0);
});

test("ask validates a too short question", async () => {
  const user = userEvent.setup();
  renderAt("/ask");
  await user.type(screen.getByLabelText("Question"), "ab");
  await user.click(screen.getByRole("button", { name: "Ask" }));
  expect(screen.getByRole("alert")).toHaveTextContent("at least 3 characters");
});

test("compare marks the source page and unavailable rerank cells", async () => {
  renderAt("/compare?q=How%20do%20I%20drain%20a%20node%3F");
  // jsdom applies no CSS, so the mobile list and the desktop grid are both in the DOM: counts are doubled.
  expect((await screen.findAllByText(/source page at 2/)).length).toBe(16);
  expect(screen.getAllByText(/available for the evaluation questions only/)).toHaveLength(2);
  expect(screen.getAllByText(/^12 ms/).length).toBe(12);
  expect(screen.queryByText(/^0 ms/)).toBeNull();
  expect(screen.getAllByText(/^evaluation run/).length).toBe(4);
});

test("scores list only the main subset and flag the live config", async () => {
  renderAt("/scores");
  const list = await screen.findByRole("list");
  expect(within(list).getAllByRole("listitem")).toHaveLength(2);
  expect(within(list).getByText("live")).toBeInTheDocument();
  expect(screen.getByText(/not significant/)).toBeInTheDocument();
  expect(screen.getAllByText(/pending/).length).toBeGreaterThan(0);
});

test("scores name an LLM annotator and say it is not a human validation", async () => {
  const row = {
    model: "qwen3.8-27b", config: "fixed_hybrid_rerank", chunking: "fixed" as const, retrieval: "hybrid_rerank" as const,
    n_factual: 45, n_impossible: 8, refusal_rate: est(1), false_refusal_rate: est(0.02), citation_coverage: est(0.91),
    answers_with_invalid_citation: 0, tokens_mean: 1400, seconds_mean: 0.47, provider: "groq", generated_at: null,
    judge_complete: true, faithfulness: est(0.98), answer_relevancy: est(0.88), context_precision: est(0.89),
    context_recall: est(0.97),
  };
  const withJudge: ScoresResponse = {
    ...scores,
    generation: {
      app_model: "qwen3.8-27b", reference_model: "qwen2.5-7b-instruct", best_config: "fixed_hybrid_rerank",
      judge: null, judge_complete: true, configs: [row], reference: { ...row, model: "qwen2.5-7b-instruct" },
      generated_at: null,
      judge_agreement: {
        annotator: "claude-opus-5-5", n_items: 40,
        metrics: [{ metric: "faithfulness", n: 40, spearman: -0.04, spearman_ci: [-0.09, -0.03], mae: 0.04,
          bias: -0.01, bias_ci: [-0.05, 0.03], kappa: null, annotator_mean: 0.99, judge_mean: 0.97 }],
      },
    },
  };
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(withJudge), { status: 200 })));
  renderAt("/scores");
  expect(await screen.findByText("claude-opus-5-5")).toBeInTheDocument();
  expect(screen.getByText(/not a human validation/)).toBeInTheDocument();
  expect(screen.getByText("−0.01")).toBeInTheDocument();
});

test("the theme toggle switches between dark and light", async () => {
  const user = userEvent.setup();
  renderAt("/scores");
  const [toggle] = await screen.findAllByRole("button", { name: /Switch to (light|dark) mode/ });
  const before = document.documentElement.dataset.theme;
  await user.click(toggle!);
  expect(document.documentElement.dataset.theme).not.toBe(before);
  expect(["light", "dark"]).toContain(document.documentElement.dataset.theme);
});
