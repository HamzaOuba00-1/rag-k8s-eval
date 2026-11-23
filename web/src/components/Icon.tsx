// A few stroke icons, drawn inline to keep the bundle free of an icon library.
const PATHS = {
  home: "M3 11l9-8 9 8M5 10v10h14V10",
  ask: "M4 5h16v11H8l-4 4z",
  compare: "M4 4h7v16H4zM13 4h7v16h-7z",
  scores: "M5 20V10M12 20V4M19 20v-7",
  send: "M12 19V5M5 12l7-7 7 7",
  copy: "M8 8h12v12H8zM4 16V4h12",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  arrow: "M5 12h14M13 6l6 6-6 6",
  info: "M12 8h.01M11 12h1v5h1M12 3a9 9 0 100 18 9 9 0 000-18z",
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, className = "h-5 w-5" }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"
      strokeLinejoin="round" className={className} aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
