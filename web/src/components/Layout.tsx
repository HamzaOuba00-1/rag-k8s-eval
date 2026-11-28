import { NavLink, Outlet, useNavigate } from "react-router";
import { ChatProvider, useChat } from "../lib/chat";
import { Brand } from "./Brand";
import { Icon, type IconName } from "./Icon";
import { ThemeToggle } from "./ThemeToggle";

const NAV: { to: string; label: string; icon: IconName }[] = [
  { to: "/", label: "Home", icon: "home" },
  { to: "/ask", label: "Ask", icon: "ask" },
  { to: "/compare", label: "Compare", icon: "compare" },
  { to: "/scores", label: "Scores", icon: "scores" },
];

const REPO = "https://github.com/HamzaOuba00-1/rag-k8s-eval";

// Every sidebar row is a 32 px icon box + label, the same box as the logo in <Brand>.
const ROW = "group flex items-center gap-3 rounded-xl p-1 pr-3 text-sm transition-colors";

function navClass({ isActive }: { isActive: boolean }) {
  return `${ROW} ${isActive ? "bg-raised text-fg ring-1 ring-line-strong" : "text-muted hover:bg-raised/60 hover:text-fg"}`;
}

function IconBox({ name, active = false }: { name: IconName; active?: boolean }) {
  return (
    <span className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg transition-colors ${
      active ? "bg-accent/12 text-accent" : "text-faint group-hover:text-fg"}`}>
      <Icon name={name} className="h-[18px] w-[18px]" />
    </span>
  );
}

export function Footer() {
  return (
    <footer className="mx-auto w-full max-w-5xl px-4 pb-28 text-[11px] leading-relaxed text-faint md:px-10 md:pb-8">
      <div className="border-t border-line pt-6">
        <p>AI-generated answers: check the cited pages. Questions are sent to Groq to generate answers and are not stored.</p>
        <p className="mt-1">
          Docs: The Kubernetes Authors,{" "}
          <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://creativecommons.org/licenses/by/4.0/"
            target="_blank" rel="noreferrer">CC BY 4.0</a>.
          Unofficial project, not affiliated with the Linux Foundation or the Kubernetes project.{" "}
          <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href={REPO} target="_blank" rel="noreferrer">
            Source code and evaluation
          </a>
        </p>
      </div>
    </footer>
  );
}

function LiveStatus() {
  const { quota } = useChat();
  if (!quota) return null;
  const on = quota.live_enabled && quota.remaining_today > 0 && !quota.reason;
  return (
    <div className="rounded-xl border border-line bg-surface/60 p-3">
      <p className="flex items-center gap-2 text-xs font-medium text-fg">
        <span className={`h-1.5 w-1.5 rounded-full ${on ? "bg-ok" : "bg-warn"}`} aria-hidden="true" />
        {on ? "Live answers on" : "Live answers paused"}
      </p>
      <p className="mt-1 text-[11px] text-muted tabular-nums">
        {quota.remaining_today} of {quota.daily_limit} left today
      </p>
    </div>
  );
}

function Shell() {
  const { reset } = useChat();
  const navigate = useNavigate();
  const newChat = () => {
    reset();
    void navigate("/ask");
  };

  return (
    <div className="min-h-screen md:grid md:grid-cols-[240px_minmax(0,1fr)]">
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-line bg-surface/40 px-3 pt-6 pb-4 backdrop-blur md:flex">
        <div className="mb-6 px-1">
          <Brand />
        </div>

        <button type="button" onClick={newChat}
          className={`${ROW} mb-6 text-fg ring-1 ring-line-strong hover:bg-raised`}>
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-accent text-ink">
            <Icon name="newChat" className="h-4 w-4" />
          </span>
          New chat
        </button>

        <p className="eyebrow mb-2 px-1">Explore</p>
        <nav className="flex flex-col gap-1" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"} className={navClass}>
              {({ isActive }) => (
                <>
                  <IconBox name={n.icon} active={isActive} />
                  {n.label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto space-y-3">
          <LiveStatus />
          <div className="flex items-center justify-between gap-2 pl-1">
            <a href={REPO} target="_blank" rel="noreferrer" className={`${ROW} flex-1 text-muted hover:text-fg`}>
              <IconBox name="code" />
              Source code
            </a>
            <ThemeToggle />
          </div>
        </div>
      </aside>

      <div className="flex min-h-screen flex-col">
        <header className="flex items-center justify-between px-4 pt-5 md:hidden">
          <Brand />
          <div className="flex items-center gap-2">
            <button type="button" onClick={newChat} aria-label="New chat" title="New chat"
              className="rounded-full p-2 text-muted ring-1 ring-line transition-colors hover:text-fg hover:ring-line-strong">
              <Icon name="newChat" className="h-4 w-4" />
            </button>
            <ThemeToggle />
          </div>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 pt-6 pb-10 md:px-10 md:pt-10 md:pb-14">
          <Outlet />
        </main>
        <Footer />
      </div>

      <nav aria-label="Main" className="fixed inset-x-5 bottom-5 z-10 flex justify-around rounded-full border border-line bg-surface/85 p-1.5 float-shadow backdrop-blur-xl md:hidden">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === "/"} aria-label={n.label}
            className={({ isActive }) => `grid h-11 w-11 place-items-center rounded-full transition-colors ${isActive ? "bg-accent text-ink" : "text-muted"}`}>
            <Icon name={n.icon} />
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

export function Layout() {
  return (
    <ChatProvider>
      <Shell />
    </ChatProvider>
  );
}
