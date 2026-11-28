import type { ReactNode } from "react";
import type { Source } from "../api/client";
import { Icon } from "./Icon";

/** Only https links are rendered as links (defence in depth: URLs come from the corpus). */
export function safeHref(url: string): string | undefined {
  try {
    return new URL(url).protocol === "https:" ? url : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Renders answer text: paragraphs, `code`, and [n] citations linked to their source page.
 * The model's text is only ever rendered as React text nodes, never as HTML.
 */
export function AnswerText({ text, sources }: { text: string; sources: Source[] }) {
  const byN = new Map(sources.map((s) => [s.n, s]));
  return (
    <div className="space-y-3 text-[15px] leading-[1.75] text-fg/95">
      {text.split(/\n{2,}/).map((para, i) => (
        <p key={i} className="whitespace-pre-line">{inline(para, byN)}</p>
      ))}
    </div>
  );
}

function inline(text: string, byN: Map<number, Source>): ReactNode[] {
  return text.split(/(\[\d+\]|`[^`]+`)/g).map((part, i) => {
    const cite = /^\[(\d+)\]$/.exec(part);
    if (cite) {
      const source = byN.get(Number(cite[1]));
      const href = source && safeHref(source.url);
      if (!source || !href) return <span key={i}>{part}</span>;
      return (
        <a key={i} href={href} target="_blank" rel="noreferrer" title={source.title}
          className="mx-0.5 rounded-md bg-accent/10 px-1.5 py-px align-[1px] text-xs font-medium text-accent ring-1 ring-accent/25 transition-colors hover:bg-accent/20">
          [{source.n}]
        </a>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i} className="rounded-md bg-ink px-1.5 py-0.5 font-mono text-[13px] text-accent-strong ring-1 ring-line">{part.slice(1, -1)}</code>;
    }
    return <span key={i}>{part}</span>;
  });
}

/** One line per source: cited ones first and highlighted. */
export function SourceChips({ sources, cited }: { sources: Source[]; cited: number[] }) {
  const used = new Set(cited);
  const ordered = [...sources].sort((a, b) => Number(used.has(b.n)) - Number(used.has(a.n)) || a.n - b.n);
  return (
    <ul className="flex flex-wrap gap-2">
      {ordered.map((s) => (
        <li key={s.chunk_id}>
          <a href={safeHref(s.url)} target="_blank" rel="noreferrer" title={s.url}
            className={`flex max-w-[260px] items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ring-1 transition-colors ${
              used.has(s.n) ? "bg-accent/10 text-accent ring-accent/30 hover:bg-accent/15" : "text-muted ring-line hover:text-fg hover:ring-line-strong"}`}>
            <span className="tabular-nums">[{s.n}]</span>
            <span className="truncate">{s.title}</span>
          </a>
        </li>
      ))}
    </ul>
  );
}

export function SourceList({ sources, cited }: { sources: Source[]; cited: number[] }) {
  const used = new Set(cited);
  return (
    <ol className="space-y-2">
      {sources.map((s) => (
        <li key={s.chunk_id} className={`rounded-xl border p-3 transition-colors ${used.has(s.n) ? "border-accent/35 bg-surface" : "border-line bg-surface/40"}`}>
          <a href={safeHref(s.url)} target="_blank" rel="noreferrer"
            className={`flex items-center gap-1 text-sm font-medium ${used.has(s.n) ? "text-accent" : "text-muted"}`}>
            [{s.n}] {s.title}
            <Icon name="external" className="h-3.5 w-3.5" />
          </a>
          <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-muted">{s.text}</p>
          {!used.has(s.n) && <p className="mt-1 text-[11px] text-faint">Retrieved, not cited</p>}
        </li>
      ))}
    </ol>
  );
}
