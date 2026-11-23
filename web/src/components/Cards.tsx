import type { ReactNode } from "react";
import type { Quota } from "../api/client";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-2xl bg-white p-4 shadow-sm ring-1 ring-slate-200/70 ${className}`}>{children}</section>;
}

export function QuotaCard({ quota }: { quota?: Quota }) {
  if (!quota) return <Card className="animate-pulse text-sm text-slate-400">Live answers…</Card>;
  const used = quota.daily_limit - quota.remaining_today;
  const pct = quota.daily_limit ? Math.round((100 * used) / quota.daily_limit) : 100;
  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <h2 className="text-sm font-semibold">Live answers today</h2>
        <span className="text-sm tabular-nums">
          {quota.remaining_today}/{quota.daily_limit} left
        </span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-slate-100" role="progressbar" aria-valuenow={used} aria-valuemin={0}
        aria-valuemax={quota.daily_limit} aria-label="Live answers used today">
        <div className="h-2 rounded-full bg-brand-600" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-xs text-slate-500">
        {quota.live_enabled
          ? "Resets at 00:00 UTC. Example questions don't use the quota."
          : `${quota.reason ?? "Live answers are paused."} Example questions still show full answers.`}
      </p>
    </Card>
  );
}

/** Ring for a 0–1 score; `value` null means not measured yet. */
export function ScoreRing({ value, display, label, sub }: { value: number | null; display?: string; label: string; sub: string }) {
  const r = 20;
  const c = 2 * Math.PI * r;
  return (
    <Card className="flex items-center gap-3">
      <svg viewBox="0 0 48 48" className="h-14 w-14 shrink-0" role="img" aria-label={`${label}: ${value === null ? "pending" : display}`}>
        <circle cx="24" cy="24" r={r} fill="none" stroke="#e2e8f0" strokeWidth="5" />
        {value !== null && (
          <circle cx="24" cy="24" r={r} fill="none" stroke="var(--color-brand-600)" strokeWidth="5" strokeLinecap="round"
            strokeDasharray={`${c * Math.max(0, Math.min(1, value))} ${c}`} transform="rotate(-90 24 24)" />
        )}
        <text x="24" y="28" textAnchor="middle" className="fill-slate-800 text-[11px] font-semibold">
          {value === null ? "…" : display}
        </text>
      </svg>
      <div>
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs text-slate-500">{value === null ? "pending" : sub}</p>
      </div>
    </Card>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return <p role="alert" className="rounded-xl bg-red-50 p-3 text-sm text-red-800">{message}</p>;
}
