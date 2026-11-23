import type { ReactNode } from "react";
import type { Source } from "../api/client";
import { Icon } from "./Icon";

/** Renders answer text: paragraphs, `code`, and [n] citations linked to their source page. */
export function AnswerText({ text, sources }: { text: string; sources: Source[] }) {
  const byN = new Map(sources.map((s) => [s.n, s]));
  return (
    <div className="space-y-3 text-[15px] leading-relaxed">
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
      if (!source) return <span key={i}>{part}</span>;
      return (
        <a key={i} href={source.url} target="_blank" rel="noreferrer" title={source.title}
          className="mx-0.5 rounded bg-brand-50 px-1 text-sm font-medium text-brand-800 hover:bg-brand-100">
          [{source.n}]
        </a>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i} className="rounded bg-slate-100 px-1 font-mono text-[13px]">{part.slice(1, -1)}</code>;
    }
    return <span key={i}>{part}</span>;
  });
}

export function SourceList({ sources, cited }: { sources: Source[]; cited: number[] }) {
  const used = new Set(cited);
  return (
    <ol className="space-y-2">
      {sources.map((s) => (
        <li key={s.chunk_id} className={`rounded-xl p-3 ring-1 ${used.has(s.n) ? "bg-white ring-brand-200" : "bg-slate-50 ring-slate-200"}`}>
          <a href={s.url} target="_blank" rel="noreferrer"
            className={`flex items-center gap-1 text-sm font-medium ${used.has(s.n) ? "text-brand-800" : "text-slate-600"}`}>
            [{s.n}] {s.title}
            <Icon name="external" className="h-3.5 w-3.5" />
          </a>
          <p className="mt-1 line-clamp-3 text-xs text-slate-500">{s.text}</p>
          {!used.has(s.n) && <p className="mt-1 text-[11px] text-slate-400">Retrieved, not cited</p>}
        </li>
      ))}
    </ol>
  );
}
