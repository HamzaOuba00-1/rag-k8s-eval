export function Orb({ size = 24 }: { size?: number }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block shrink-0 rounded-full bg-brand-400 ring-brand-50"
      style={{ width: size, height: size, boxShadow: `inset 0 0 0 ${Math.round(size / 4)}px var(--color-brand-200)` }}
    />
  );
}

export function Brand() {
  return (
    <span className="flex items-center gap-2">
      <Orb />
      <span className="text-base font-semibold tracking-tight">DocTruth</span>
    </span>
  );
}
