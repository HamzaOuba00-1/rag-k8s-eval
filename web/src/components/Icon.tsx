// Stroke icons on a 24 px grid, drawn inline so the bundle needs no icon library.
// One stroke width and round caps everywhere, so the set reads as one family.
const PATHS = {
  home: ["M3.5 10.5 12 3.5l8.5 7", "M5.5 9v10.5a1 1 0 0 0 1 1H10v-5.5h4v5.5h3.5a1 1 0 0 0 1-1V9"],
  ask: ["M20.5 11.5a8 8 0 0 1-11.8 7.1L3.5 20l1.4-4.7A8 8 0 1 1 20.5 11.5z", "M8.5 10h7", "M8.5 13.5h4.5"],
  compare: [
    "M9.5 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h4.5",
    "M14.5 4H19a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-4.5",
    "M12 2.5v19",
  ],
  scores: ["M4 3.5V20h16.5", "M8 16v-4", "M12.5 16V8", "M17 16v-6.5"],
  send: ["M12 19V5.5", "m5.5 11.5 6.5-6.5 6.5 6.5"],
  copy: [
    "M10 8.5h9a1 1 0 0 1 1 1v9.5a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1V9.5a1 1 0 0 1 1-1z",
    "M5.5 15.5H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h9.5a1 1 0 0 1 1 1v.5",
  ],
  check: ["m5 12.5 4.5 4.5L19 7.5"],
  external: ["M14 4h6v6", "M20 4l-8.5 8.5", "M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"],
  arrow: ["M5 12h14", "m13 6 6 6-6 6"],
  info: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z", "M12 11v5.5", "M12 7.75v.01"],
  sun: [
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z",
    "M12 2.5v1.5", "M12 20v1.5", "M4.9 4.9 6 6", "M18 18l1.1 1.1", "M2.5 12H4", "M20 12h1.5", "M4.9 19.1 6 18", "M18 6l1.1-1.1",
  ],
  moon: ["M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"],
  newChat: ["M12 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-6", "M17.6 3.4a2 2 0 0 1 2.9 2.9L13 13.8l-3.5.9.9-3.5z"],
  code: ["m8 8-4 4 4 4", "m16 8 4 4-4 4", "m13.5 5-3 14"],
  refresh: ["M20 11.5a8 8 0 1 0-2.4 5.9", "M20 5v6.5h-6.5"],
  book: ["M5 4.5A1.5 1.5 0 0 1 6.5 3H19v15H6.5A1.5 1.5 0 0 0 5 19.5z", "M5 19.5A1.5 1.5 0 0 0 6.5 21H19"],
  chevron: ["m6 9 6 6 6-6"],
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, className = "h-5 w-5" }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round"
      strokeLinejoin="round" className={className} aria-hidden="true">
      {PATHS[name].map((d) => <path key={d} d={d} />)}
    </svg>
  );
}
