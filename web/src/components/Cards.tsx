import type { ReactNode } from "react";
import type { Quota } from "../api/client";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`panel p-5 ${className}`}>{children}</section>;
}

export function QuotaCard({ quota }: { quota?: Quota }) {
  if (!quota) return <Card className="animate-pulse text-sm text-faint">Live answers…</Card>;
  const used = quota.daily_limit - quota.remaining_today;
  const pct = quota.daily_limit ? Math.round((100 * used) / quota.daily_limit) : 100;
  return (
    <Card>
      <p className="eyebrow">Live answers today</p>
      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="text-4xl leading-none tabular-nums">{quota.remaining_today}</span>
        <span className="text-sm text-muted">of {quota.daily_limit} left</span>
      </div>
      <div className="mt-4 h-1 rounded-full bg-line" role="progressbar" aria-valuenow={used} aria-valuemin={0}
        aria-valuemax={quota.daily_limit} aria-label="Live answers used today">
        <div className="h-1 rounded-full bg-gradient-to-r from-accent-deep to-accent" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-3 text-xs leading-relaxed text-muted">
        {quota.live_enabled
          ? "Resets at 00:00 UTC. Example questions don't use the quota."
          : `${quota.reason ?? "Live answers are paused."} Example questions still show full answers.`}
      </p>
    </Card>
  );
}

/** Ring for a 0–1 score; `value` null means not measured yet. */
export function ScoreRing({ value, display, label, sub }: { value: number | null; display?: string; label: string; sub: string }) {
  const r = 21;
  const c = 2 * Math.PI * r;
  return (
    <Card className="flex flex-col gap-4">
      <svg viewBox="0 0 52 52" className="h-16 w-16" role="img" aria-label={`${label}: ${value === null ? "pending" : display}`}>
        <circle cx="26" cy="26" r={r} fill="none" stroke="var(--color-line)" strokeWidth="3" />
        {value !== null && (
          <circle cx="26" cy="26" r={r} fill="none" stroke="var(--color-accent)" strokeWidth="3" strokeLinecap="round"
            strokeDasharray={`${c * Math.max(0, Math.min(1, value))} ${c}`} transform="rotate(-90 26 26)" />
        )}
        <text x="26" y="30.5" textAnchor="middle" className={`text-[12px] font-medium ${value === null ? "fill-faint" : "fill-fg"}`}>
          {value === null ? "—" : display}
        </text>
      </svg>
      <div>
        <p className="text-sm font-medium">{label}</p>
        <p className={`mt-0.5 text-xs ${value === null ? "text-faint italic" : "text-muted"}`}>{value === null ? "pending" : sub}</p>
      </div>
    </Card>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return <p role="alert" className="rounded-xl bg-danger/10 p-3 text-sm text-danger ring-1 ring-danger/25">{message}</p>;
}
