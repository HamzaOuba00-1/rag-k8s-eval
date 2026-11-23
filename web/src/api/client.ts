import type { components } from "./schema";

type S = components["schemas"];
export type Source = S["Source"];
export type Quota = S["Quota"];
export type Answer = S["Answer"];
export type AskResponse = S["AskResponse"];
export type ExampleQuestion = S["ExampleQuestion"];
export type SearchRequest = S["SearchRequest"];
export type SearchResponse = S["SearchResponse"];
export type SearchCell = S["SearchCell"];
export type ScoresResponse = S["ScoresResponse"];
export type RetrievalRow = S["RetrievalRow"];
export type GenerationRow = S["GenerationRow"];
export type Estimate = S["Estimate"];
export type Strategy = SearchCell["strategy"];
export type Method = SearchCell["method"];

export const STRATEGIES: Strategy[] = ["fixed", "structure", "semantic"];
export const METHODS: Method[] = ["bm25", "dense", "hybrid", "hybrid_rerank"];

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`;
    try {
      const body = (await resp.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep the generic message
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

export const api = {
  quota: (signal?: AbortSignal) => request<Quota>("/quota", { signal }),
  examples: (signal?: AbortSignal) => request<ExampleQuestion[]>("/examples", { signal }),
  scores: (signal?: AbortSignal) => request<ScoresResponse>("/scores", { signal }),
  ask: (question: string, signal?: AbortSignal) =>
    request<AskResponse>("/ask", { method: "POST", body: JSON.stringify({ question }), signal }),
  search: (body: SearchRequest, signal?: AbortSignal) =>
    request<SearchResponse>("/search", { method: "POST", body: JSON.stringify(body), signal }),
};
