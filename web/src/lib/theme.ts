export type Theme = "dark" | "light";

const KEY = "doctruth-theme";

/** Saved choice, else the system preference. Storage can be unavailable: then nothing is remembered. */
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // private mode or blocked storage
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function applyTheme(theme: Theme, remember = true): void {
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "light" ? "#f5efe6" : "#0e0c0a");
  if (!remember) return;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // not remembered, still applied
  }
}
