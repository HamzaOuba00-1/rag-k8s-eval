/** DocTruth mark: the bronze-edged mask emblem (public/logo-*.png). */
export function Logo({ size = 24, className = "" }: { size?: number; className?: string }) {
  const src = size > 96 ? "/logo-512.png" : size > 32 ? "/logo-192.png" : "/logo-64.png";
  return (
    <img src={src} width={size} height={size} alt="" aria-hidden="true" draggable={false}
      style={{ width: size, height: size }}
      className={`logo-mark shrink-0 object-contain select-none ${className}`} />
  );
}

export function Brand() {
  return (
    <span className="flex items-center gap-2.5">
      <Logo size={30} />
      <span className="text-[17px] font-semibold tracking-[-0.03em] text-fg">DocTruth</span>
    </span>
  );
}
