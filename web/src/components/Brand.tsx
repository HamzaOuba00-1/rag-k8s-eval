/** DocTruth mark: the bronze-edged mask emblem (public/logo-*.png, content centred in the square). */
export function Logo({ size = 24, className = "" }: { size?: number; className?: string }) {
  const src = size > 96 ? "/logo-512.png" : size > 32 ? "/logo-192.png" : "/logo-64.png";
  return (
    <img src={src} width={size} height={size} alt="" aria-hidden="true" draggable={false}
      style={{ width: size, height: size }}
      className={`logo-mark block shrink-0 object-contain select-none ${className}`} />
  );
}

/**
 * Logo in a 32 px box, the same box as the sidebar icons, so the mark, the icons and
 * the labels line up on the same axes.
 */
export function Brand() {
  return (
    <span className="flex items-center gap-3">
      <span className="grid h-8 w-8 shrink-0 place-items-center">
        <Logo size={30} />
      </span>
      <span className="text-[17px] leading-none font-semibold tracking-[-0.03em] text-fg">DocTruth</span>
    </span>
  );
}
