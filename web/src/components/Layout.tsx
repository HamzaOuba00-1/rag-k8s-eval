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
  return `flex items-center gap-3 rounded-full px-4 py-2 text-sm transition-colors ${
    isActive ? "bg-brand-600 text-white" : "text-slate-600 hover:bg-brand-50"
  }`;
}

export function Footer() {
  return (
    <footer className="px-4 pb-24 pt-6 text-center text-xs leading-relaxed text-slate-500 md:pb-6">
      <p>AI-generated answers: check the cited pages. Questions are sent to Groq to generate answers and are not stored.</p>
      <p>
        Docs: The Kubernetes Authors,{" "}
        <a className="underline" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Unofficial
        project, not affiliated with the Linux Foundation or the Kubernetes project.{" "}
        <a className="underline" href="https://github.com/HamzaOuba00-1/rag-k8s-eval">Source code and evaluation</a>
      </p>
    </footer>
  );
}

export function Layout() {
  return (
    <div className="min-h-screen md:grid md:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="hidden border-r border-slate-200 bg-white p-4 md:flex md:flex-col md:gap-2">
        <div className="mb-6 px-2">
          <Brand />
        </div>
        <nav className="flex flex-col gap-1" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"} className={navClass}>
              <Icon name={n.icon} />
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="flex min-h-screen flex-col">
        <header className="flex items-center px-4 pt-4 md:hidden">
          <Brand />
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-4 md:px-8 md:py-8">
          <Outlet />
        </main>
        <Footer />
      </div>

      <nav aria-label="Main" className="fixed inset-x-4 bottom-4 z-10 flex justify-around rounded-full bg-white/95 p-2 shadow-lg ring-1 ring-slate-200 md:hidden">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === "/"} aria-label={n.label}
            className={({ isActive }) => `rounded-full p-3 ${isActive ? "bg-brand-600 text-white" : "text-slate-500"}`}>
            <Icon name={n.icon} />
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
