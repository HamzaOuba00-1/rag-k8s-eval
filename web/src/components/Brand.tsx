/** Brand mark: a matte bronze sphere with a soft highlight. */
export function Orb({ size = 24 }: { size?: number }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block shrink-0 rounded-full"
      style={{
        width: size,
        height: size,
        background:
          "radial-gradient(circle at 32% 28%, #f1d9b8 0%, #c19a6b 28%, #6b4f33 68%, #2a1f15 100%)",
        boxShadow: `0 0 ${Math.round(size / 2)}px rgb(193 154 107 / 0.25), inset 0 -${Math.max(1, Math.round(size / 10))}px ${Math.round(size / 5)}px rgb(0 0 0 / 0.45)`,
      }}
    />
  );
}

export function Brand() {
  return (
    <span className="flex items-center gap-2.5">
      <Orb size={22} />
      <span className="font-display text-[22px] leading-none tracking-tight text-fg">DocTruth</span>
    </span>
  );
}
