import { NavLink, Outlet } from "react-router";
import { Brand } from "./Brand";
import { Icon, type IconName } from "./Icon";

const NAV: { to: string; label: string; icon: IconName }[] = [
  { to: "/", label: "Home", icon: "home" },
  { to: "/ask", label: "Ask", icon: "ask" },
  { to: "/compare", label: "Compare", icon: "compare" },
  { to: "/scores", label: "Scores", icon: "scores" },
];

function navClass({ isActive }: { isActive: boolean }) {
  return `group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors ${
    isActive ? "bg-raised text-fg ring-1 ring-line-strong" : "text-muted hover:bg-raised/60 hover:text-fg"
  }`;
}

export function Footer() {
  return (
    <footer className="mx-auto w-full max-w-5xl border-t border-line px-4 pb-28 pt-6 text-[11px] leading-relaxed text-faint md:px-10 md:pb-8">
      <p>AI-generated answers: check the cited pages. Questions are sent to Groq to generate answers and are not stored.</p>
      <p className="mt-1">
        Docs: The Kubernetes Authors,{" "}
        <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
        Unofficial project, not affiliated with the Linux Foundation or the Kubernetes project.{" "}
        <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://github.com/HamzaOuba00-1/rag-k8s-eval">Source code and evaluation</a>
      </p>
    </footer>
  );
}

export function Layout() {
  return (
    <div className="min-h-screen md:grid md:grid-cols-[232px_minmax(0,1fr)]">
      <aside className="sticky top-0 hidden h-screen border-r border-line bg-surface/40 px-4 py-6 backdrop-blur md:flex md:flex-col">
        <div className="mb-10 px-2">
          <Brand />
        </div>
        <p className="eyebrow mb-2 px-3">Explore</p>
        <nav className="flex flex-col gap-1" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"} className={navClass}>
              <Icon name={n.icon} className="h-[18px] w-[18px]" />
              {n.label}
            </NavLink>
          ))}
        </nav>
        <p className="mt-auto px-3 text-[11px] leading-relaxed text-faint">
          Retrieval-augmented answers over the Kubernetes documentation, with every score measured.
        </p>
      </aside>

      <div className="flex min-h-screen flex-col">
        <header className="flex items-center px-5 pt-5 md:hidden">
          <Brand />
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 md:px-10 md:py-12">
          <Outlet />
        </main>
        <Footer />
      </div>

      <nav aria-label="Main" className="fixed inset-x-5 bottom-5 z-10 flex justify-around rounded-full border border-line bg-surface/85 p-1.5 shadow-2xl shadow-black/60 backdrop-blur-xl md:hidden">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === "/"} aria-label={n.label}
            className={({ isActive }) => `rounded-full p-3 transition-colors ${isActive ? "bg-accent text-ink" : "text-muted"}`}>
            <Icon name={n.icon} />
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
