import { useEffect, useState } from "react";
import { applyTheme, initialTheme, type Theme } from "../lib/theme";
import { Icon } from "./Icon";

export function ThemeToggle({ className = "" }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => applyTheme(theme, false), [theme]);
  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button type="button" onClick={() => { setTheme(next); applyTheme(next); }}
      aria-label={`Switch to ${next} mode`} title={`Switch to ${next} mode`}
      className={`rounded-full p-2 text-muted ring-1 ring-line transition-colors hover:text-fg hover:ring-line-strong ${className}`}>
      <Icon name={theme === "dark" ? "sun" : "moon"} className="h-4 w-4" />
    </button>
  );
}
