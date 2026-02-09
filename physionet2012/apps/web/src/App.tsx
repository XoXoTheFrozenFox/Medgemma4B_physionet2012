import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import type { Preset } from "./api";
import { analyze, health } from "./api";

const SAMPLE_NOTE =
  "55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.";

function cleanReply(s: string): string {
  if (!s) return "";
  return s
    .replace(/<\/?(Answer|BL)\s*>/gi, "")
    .replace(/<unused\d+>\s*thought/gi, "")
    .replace(/<\/?end_of_turn\s*>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export default function App() {
  const [preset, setPreset] = useState<Preset>("quick");
  const [note, setNote] = useState<string>(SAMPLE_NOTE);
  const [reply, setReply] = useState<string>("");
  const [meta, setMeta] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [healthInfo, setHealthInfo] = useState<any>(null);
  const [error, setError] = useState<string>("");

  const presetHint = useMemo(() => {
    if (preset === "quick") return "Fast triage bullets (brief).";
    if (preset === "normal") return "SOAP + tasks + red flags + patient summary.";
    return "Full analysis with workup + management considerations.";
  }, [preset]);

  async function refreshHealth() {
    try {
      const h = await health();
      setHealthInfo(h);
    } catch {
      setHealthInfo(null);
    }
  }

  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 15_000);
    return () => clearInterval(t);
  }, []);

  async function runAnalyze() {
    setError("");
    setReply("");
    setMeta(null);
    setLoading(true);

    const res = await analyze({
      preset,
      note,
      debug: true,
    });

    setLoading(false);

    if (!res.ok) {
      setError(res.error || "Unknown error");
      return;
    }

    setReply(cleanReply(res.data?.reply ?? ""));
    setMeta(res.data?.meta ?? null);
  }

  return (
    <div className="app">
      <div className="header">
        <div className="brand">
          <div className="logo">🩺</div>
          <div className="title">
            <h1>MedGemma Clinical Console</h1>
            <span>Local inference • Quick / Normal / Detailed • Neon red theme 🫀</span>
          </div>
        </div>

        <div className="status">
          <span className={"pill " + (healthInfo?.ok ? "ok" : "bad")}>
            {healthInfo?.ok ? "API: Ready" : "API: Offline"}
          </span>
          <span className="pill">{healthInfo?.gpu ? `GPU: ${healthInfo.gpu}` : "GPU: n/a"}</span>
          <span className="pill">{healthInfo?.is_loaded_in_4bit ? "4-bit: Yes" : "4-bit: ?"}</span>
        </div>
      </div>

      <div className="grid">
        {/* LEFT */}
        <div className="card">
          <div className="cardHead">
            <div className="cardTitle">
              <span className="emoji">🧾</span> Clinical note input
            </div>
            <div className="controls">
              <button className="btn small" onClick={() => setNote(SAMPLE_NOTE)} title="Load sample note">
                Load sample
              </button>
              <button
                className="btn small"
                onClick={() => {
                  setNote("");
                  setReply("");
                  setMeta(null);
                  setError("");
                }}
                title="Clear"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="body">
            <textarea
              className="textarea"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Paste or type the clinical note / case details here..."
            />

            <div className="metaRow">
              <span>
                Preset: <strong>{preset}</strong>
              </span>
              <span>•</span>
              <span>{presetHint}</span>
              <span>•</span>
              <span>Tip: keep vitals + labs in the first lines 🧪</span>
            </div>

            <div className="btnRow">
              <button className={"btn " + (preset === "quick" ? "btnActive" : "")} onClick={() => setPreset("quick")}>
                ⚡ Quick
              </button>
              <button className={"btn " + (preset === "normal" ? "btnActive" : "")} onClick={() => setPreset("normal")}>
                📋 Normal
              </button>
              <button
                className={"btn " + (preset === "detailed" ? "btnActive" : "")}
                onClick={() => setPreset("detailed")}
              >
                🧠 Detailed
              </button>

              <div style={{ flex: 1 }} />

              <motion.button
                className="btn primary"
                onClick={runAnalyze}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                disabled={loading}
              >
                {loading ? (
                  <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                    <span className="spinner" /> Analyzing…
                  </span>
                ) : (
                  "▶ Analyze"
                )}
              </motion.button>
            </div>

            {!!error && <div className="toast">⚠ {error}</div>}

            <div className="footerNote">Note: This is a reference UI for your local model. Always validate clinically. 💉🩻</div>
          </div>
        </div>

        {/* RIGHT */}
        <div className="card">
          <div className="cardHead">
            <div className="cardTitle">
              <span className="emoji">🫀</span> Model reply
            </div>
            <div className="controls">
              <button
                className="btn small"
                onClick={() => navigator.clipboard.writeText(reply || "")}
                disabled={!reply}
                title="Copy reply"
              >
                Copy
              </button>
              <button
                className="btn small"
                onClick={() => navigator.clipboard.writeText(JSON.stringify(meta ?? {}, null, 2))}
                disabled={!meta}
                title="Copy meta"
              >
                Copy meta
              </button>
            </div>
          </div>

          <div className="body">
            <motion.div
              className="output"
              initial={{ opacity: 0.0, y: 6 }}
              animate={{ opacity: 1.0, y: 0 }}
              transition={{ duration: 0.25 }}
            >
              {reply ? reply : "No reply yet. Click Analyze. 🩺"}
            </motion.div>

            <div className="metaRow">
              <span>Meta:</span>
              <span className="pill" style={{ borderColor: "rgba(255,255,255,0.14)" }}>
                {meta?.latency_ms ? `latency ${meta.latency_ms} ms` : "latency n/a"}
              </span>
              <span className="pill" style={{ borderColor: "rgba(255,255,255,0.14)" }}>
                {meta?.passes != null ? `passes ${meta.passes}` : "passes n/a"}
              </span>
              <span className="pill" style={{ borderColor: "rgba(255,255,255,0.14)" }}>
                {meta?.usage?.total_tokens != null ? `tokens ${meta.usage.total_tokens}` : "tokens n/a"}
              </span>
            </div>

            <div className="output" style={{ maxHeight: 180, fontSize: 12, color: "rgba(255,255,255,0.75)" }}>
              {meta ? JSON.stringify(meta, null, 2) : "Meta will appear here when debug=true."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
