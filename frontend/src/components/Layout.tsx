import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { api, getApiKey, setApiKey } from "../api/client";
import type { HealthResponse } from "../api/types";
import { Badge } from "./Badge";

interface LayoutProps {
  children: ReactNode;
}

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/upload", label: "Upload" },
  { to: "/documents", label: "Documents" },
  { to: "/reference", label: "Reference Dataset" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/feedback", label: "Feedback" },
];

export function Layout({ children }: LayoutProps) {
  const [apiKeyInput, setApiKeyInput] = useState<string>(() => getApiKey());
  const [savedNotice, setSavedNotice] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    api
      .health(ctrl.signal)
      .then((h) => {
        setHealth(h);
        setHealthErr(null);
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setHealthErr(e instanceof Error ? e.message : "Offline");
      });
    return () => ctrl.abort();
  }, [savedNotice]);

  const handleSaveKey = (e: React.FormEvent) => {
    e.preventDefault();
    setApiKey(apiKeyInput.trim());
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 1800);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="sidebar-brand">
          <div className="brand-title">FRI Engine</div>
          <div className="brand-subtitle">Financial Relevance Intelligence</div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? " nav-link-active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-section">
            <div className="sidebar-label">System Status</div>
            {health ? (
              <div className="health-summary">
                <div className="health-row">
                  <span>API</span>
                  <Badge tone={health.status === "ok" ? "relevant" : "uncertain"} size="sm">
                    {health.status.toUpperCase()}
                  </Badge>
                </div>
                <div className="health-row">
                  <span>Worker</span>
                  <Badge tone={health.worker_running ? "relevant" : "failed"} size="sm">
                    {health.worker_running ? "Running" : "Stopped"}
                  </Badge>
                </div>
                <div className="health-row">
                  <span>Vectors</span>
                  <span className="mono">{health.index_vectors.toLocaleString()}</span>
                </div>
                <div className="health-row">
                  <span>Mode</span>
                  <span className="mono">{health.classifier_mode}</span>
                </div>
              </div>
            ) : healthErr ? (
              <div className="health-row">
                <span>API</span>
                <Badge tone="failed" size="sm" title={healthErr}>
                  Unreachable
                </Badge>
              </div>
            ) : (
              <div className="muted-sm">Checking status…</div>
            )}
          </div>

          <form className="sidebar-section" onSubmit={handleSaveKey}>
            <label htmlFor="sidebar-api-key" className="sidebar-label">
              API Key {health?.auth_enabled ? "(required)" : "(optional)"}
            </label>
            <div className="api-key-row">
              <input
                id="sidebar-api-key"
                type="password"
                className="input input-sm"
                placeholder="X-API-Key…"
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                autoComplete="off"
              />
              <button type="submit" className="btn btn-sm">
                {savedNotice ? "Saved" : "Set"}
              </button>
            </div>
          </form>
        </div>
      </aside>

      <main className="main-content">{children}</main>
    </div>
  );
}
