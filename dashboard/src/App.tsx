import { useState } from "react";
import LiveMonitor from "./LiveMonitor";
import HarnessApp from "./HarnessApp";
import AgentApp from "./AgentApp";
import "./App.css";

type Tab = "agent" | "live" | "harness";

export default function App() {
  const [tab, setTab] = useState<Tab>("agent");

  return (
    <div className="shell">
      <header className="header shell-header">
        <h1>
          <span>Leash</span> — Assistant Axis Monitor
        </h1>
        <nav className="tab-nav">
          <button
            type="button"
            className={`tab-btn ${tab === "agent" ? "active" : ""}`}
            onClick={() => setTab("agent")}
          >
            Agent
          </button>
          <button
            type="button"
            className={`tab-btn ${tab === "live" ? "active" : ""}`}
            onClick={() => setTab("live")}
          >
            Live
          </button>
          <button
            type="button"
            className={`tab-btn ${tab === "harness" ? "active" : ""}`}
            onClick={() => setTab("harness")}
          >
            Traces
          </button>
        </nav>
        <div className="header-meta">
          {tab === "agent"
            ? "Live agent + projections · backend :8787"
            : tab === "live"
              ? "Higher projection → more Assistant-like"
              : "Replay saved sessions"}
        </div>
      </header>
      {tab === "agent" && <AgentApp />}
      {tab === "live" && <LiveMonitor />}
      {tab === "harness" && <HarnessApp />}
    </div>
  );
}
