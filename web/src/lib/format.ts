import type { Estimate } from "../api/client";

export const LABELS: Record<string, string> = {
  fixed: "fixed",
  structure: "structure",
  semantic: "semantic",
  bm25: "BM25",
  dense: "dense",
  hybrid: "hybrid",
  hybrid_rerank: "hybrid + rerank",
};

export function configLabel(config: string): string {
  const [strategy, ...rest] = config.split("_");
  return `${LABELS[strategy ?? ""] ?? strategy} · ${LABELS[rest.join("_")] ?? rest.join("_")}`;
}

export function num(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? "–" : value.toFixed(digits);
}

export function ci(e: Estimate | null | undefined, digits = 2): string {
  if (!e) return "pending";
  return `${e.mean.toFixed(digits)} [${e.low.toFixed(digits)}, ${e.high.toFixed(digits)}]`;
}

export function signed(e: Estimate | null | undefined, digits = 3): string {
  if (!e) return "";
  const f = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
  return `${f(e.mean)} [${f(e.low)}, ${f(e.high)}]`;
}

/** True when a 95% interval of a difference excludes zero. */
export function significant(e: Estimate | null | undefined): boolean {
  return !!e && (e.low > 0 || e.high < 0);
}
