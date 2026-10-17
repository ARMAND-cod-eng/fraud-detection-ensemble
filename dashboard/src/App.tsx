import { useEffect, useState } from "react";
import { api } from "./lib/api";
import { usePoll, useHashRoute } from "./lib/hooks";
import { Icon, Shield } from "./components/ui";
import Overview from "./pages/Overview";
import Live from "./pages/Live";
import Cases from "./pages/Cases";
import Performance from "./pages/Performance";
import Policy from "./pages/Policy";
import Monitoring from "./pages/Monitoring";

const NAV = [
  { key: "overview", label: "Overview", icon: Icon.overview },
  { key: "live", label: "Live monitor", icon: Icon.live },
  { key: "cases", label: "Investigations", icon: Icon.cases },
  { key: "performance", label: "Model performance", icon: Icon.model },
  { key: "policy", label: "Decision policy", icon: Icon.policy },
  { key: "monitoring", label: "Monitoring", icon: Icon.monitor },
];

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem("fs-theme") ?? ""; } catch { return ""; }
  });
  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem("fs-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  const dark = theme === "dark" || (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  return [dark ? "dark" : "light", () => setTheme(dark ? "light" : "dark")];
}

export default function App() {
  const [page, params] = useHashRoute();
  const [theme, toggle] = useTheme();
  const ready = usePoll(() => fetch("/ready").then((r) => r.json()), 10000);
  const open = usePoll(() => api.transactions({ review_status: "open", alerts_only: true, limit: 1 }), 5000);
  const ok = ready.data?.status === "ready";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand"><Shield /><div className="brand-name">FraudShield <span>AI</span></div></div>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => (
            <a key={n.key} href={`#/${n.key}`} className={page === n.key ? "active" : ""} aria-current={page === n.key ? "page" : undefined}>
              <n.icon />{n.label}
              {n.key === "cases" && open.data?.total ? <span className="count">{open.data.total}</span> : null}
            </a>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className={`badge ${ok ? "ok" : "alert"}`}><span className="dot" />{ok ? "Service ready" : "Service not ready"}</span>
          {ready.data?.model_version && <span className="mono muted" style={{ fontSize: 11, wordBreak: "break-all" }}>{ready.data.model_version}</span>}
          <span className="small muted">Local demo · public ULB dataset</span>
          <button className="btn sm" onClick={toggle} aria-label="Toggle colour theme">{theme === "dark" ? <Icon.sun /> : <Icon.moon />}{theme === "dark" ? "Light" : "Dark"} mode</button>
        </div>
      </aside>
      <main className="main">
        {page === "overview" && <Overview />}
        {page === "live" && <Live />}
        {page === "cases" && <Cases id={params[0] ? decodeURIComponent(params[0]) : undefined} />}
        {page === "performance" && <Performance />}
        {page === "policy" && <Policy />}
        {page === "monitoring" && <Monitoring />}
        {!NAV.some((n) => n.key === page) && <div className="empty">Page not found. <a href="#/overview">Go to overview</a></div>}
      </main>
    </div>
  );
}
