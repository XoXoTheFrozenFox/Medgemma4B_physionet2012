import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Preset = "quick" | "normal" | "detailed";

const SAMPLE_NOTE =
  "55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function safeStr(x: unknown): string {
  if (typeof x === "string") return x;
  if (x == null) return "";
  try { return JSON.stringify(x, null, 2); } catch { return String(x); }
}

async function httpGetJson(url: string, timeoutMs = 15000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally {
    clearTimeout(id);
  }
}

async function httpPostJson(url: string, body: any, timeoutMs = 240000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const text = await r.text();
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${text}`);
    return JSON.parse(text);
  } finally {
    clearTimeout(id);
  }
}

export default function App() {
  const [preset, setPreset] = useState<Preset>("quick");
  const [note, setNote] = useState<string>(SAMPLE_NOTE);
  const [reply, setReply] = useState<string>("");
  const [meta, setMeta] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState<string>("");

  const outputRef = useRef<HTMLDivElement | null>(null);

  const presetHint = useMemo(() => {
    if (preset === "quick") return "⚡ Fast triage bullets (brief).";
    if (preset === "normal") return "📋 SOAP + tasks + red flags + patient summary.";
    return "🧠 Full analysis with workup + management considerations.";
  }, [preset]);

  const ipcAvailable = !!window.medapi?.health && !!window.medapi?.analyze;

  async function refreshHealth() {
    try {
      // ✅ Prefer IPC, fallback to direct HTTP so UI still works.
      const h = ipcAvailable
        ? await window.medapi!.health()
        : await httpGetJson(`${DEFAULT_API_BASE}/health`, 15000);
      setHealth(h);
    } catch {
      setHealth(null);
    }
  }

  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 12000);
    return () => clearInterval(t);
  }, [ipcAvailable]);

  useEffect(() => {
    if (outputRef.current) outputRef.current.scrollTop = 0;
  }, [reply]);

  async function runAnalyze() {
    setError("");
    setReply("");
    setMeta(null);

    if (!note.trim()) {
      setError("Please paste a clinical note first.");
      return;
    }

    setLoading(true);
    try {
      const payload = { preset, note, debug: true };
      const res = ipcAvailable
        ? await window.medapi!.analyze(payload)
        : { ok: true, data: await httpPostJson(`${DEFAULT_API_BASE}/v1/analyze`, payload, 240000) };

      if (!res?.ok) {
        setError(res?.error || "Unknown error.");
        return;
      }

      setReply(String(res.data?.reply ?? ""));
      setMeta(res.data?.meta ?? null);
    } catch (e: any) {
      setError(e?.name === "AbortError" ? "Request timed out." : (e?.message || String(e)));
    } finally {
      setLoading(false);
    }
  }

  async function copy(kind: "reply" | "meta") {
    const text = kind === "reply" ? (reply || "") : safeStr(meta ?? {});
    try {
      if (window.medapi?.copyText) {
        await window.medapi.copyText(text);
      } else {
        await navigator.clipboard.writeText(text);
      }
    } catch {}
  }

  const apiReady = !!health?.ok;

  return (
    <div className="app">
      <div className="topGlow" />

      <div className="header">
        <div className="brand">
          <div className="logoPulse">
            <div className="logo">🩺</div>
          </div>
          <div className="title">
            <h1>MedGemma Clinical Console</h1>
            <span>
              Local inference • Quick / Normal / Detailed • Neon Red Theme 🫀{" "}
              <span style={{ opacity: 0.7 }}>
                {ipcAvailable ? "• IPC OK" : "• IPC off (HTTP fallback)"}
              </span>
            </span>
          </div>
        </div>

        <div className="status">
          <span className={"pill " + (apiReady ? "ok" : "bad")}>
            {apiReady ? "API: Ready" : "API: Offline"}
          </span>
          <span className="pill">{health?.gpu ? `GPU: ${health.gpu}` : "GPU: n/a"}</span>
          <span className="pill">
            {health?.is_loaded_in_4bit === true ? "4-bit: Yes" : health?.is_loaded_in_4bit === false ? "4-bit: No" : "4-bit: ?"}
          </span>
        </div>
      </div>

      <div className="grid">
        {/* LEFT */}
        <div className="card">
          <div className="cardHead">
            <div className="cardTitle"><span className="emoji">🧾</span> Clinical note input</div>
            <div className="controls">
              <button className="btn small" onClick={() => setNote(SAMPLE_NOTE)}>Load sample</button>
              <button className="btn small" onClick={() => { setNote(""); setReply(""); setMeta(null); setError(""); }}>Clear</button>
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
              <span>Preset: <strong>{preset}</strong></span>
              <span>•</span>
              <span>{presetHint}</span>
              <span>•</span>
              <span className="muted">Tip: vitals + labs early 🧪 • imaging pending 🩻</span>
            </div>

            <div className="btnRow">
              <motion.button className={"btn " + (preset === "quick" ? "btnActive" : "")}
                onClick={() => setPreset("quick")} whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }}>
                ⚡ Quick
              </motion.button>

              <motion.button className={"btn " + (preset === "normal" ? "btnActive" : "")}
                onClick={() => setPreset("normal")} whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }}>
                📋 Normal
              </motion.button>

              <motion.button className={"btn " + (preset === "detailed" ? "btnActive" : "")}
                onClick={() => setPreset("detailed")} whileHover={{ y: -1 }} whileTap={{ scale: 0.98 }}>
                🧠 Detailed
              </motion.button>

              <div style={{ flex: 1 }} />

              <motion.button className="btn primary"
                onClick={runAnalyze} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} disabled={loading}>
                {loading ? <span className="inline"><span className="spinner" /> Analyzing…</span> : "▶ Analyze"}
              </motion.button>
            </div>

            <AnimatePresence>
              {!!error && (
                <motion.div className="toast"
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
                  transition={{ duration: 0.18 }}>
                  ⚠ {error}
                </motion.div>
              )}
            </AnimatePresence>

            <div className="footerNote">Reference UI for your local model. Validate clinically. 💉🩻</div>
          </div>
        </div>

        {/* RIGHT */}
        <div className="card">
          <div className="cardHead">
            <div className="cardTitle"><span className="emoji">🫀</span> Model reply</div>
            <div className="controls">
              <button className="btn small" onClick={() => copy("reply")} disabled={!reply}>Copy reply</button>
              <button className="btn small" onClick={() => copy("meta")} disabled={!meta}>Copy meta</button>
            </div>
          </div>

          <div className="body">
            <motion.div
              ref={outputRef}
              className="output"
              initial={{ opacity: 0.0, y: 6 }}
              animate={{ opacity: 1.0, y: 0 }}
              transition={{ duration: 0.22 }}
            >
              {reply ? reply : "No reply yet. Click Analyze. 🩺"}
            </motion.div>

            <div className="metaRow">
              <span>Meta:</span>
              <span className="pill dim">{meta?.latency_ms ? `latency ${meta.latency_ms} ms` : "latency n/a"}</span>
              <span className="pill dim">{meta?.passes != null ? `passes ${meta.passes}` : "passes n/a"}</span>
              <span className="pill dim">{meta?.usage?.total_tokens != null ? `tokens ${meta.usage.total_tokens}` : "tokens n/a"}</span>
            </div>

            <div className="output metaBox">{meta ? safeStr(meta) : "Meta will appear here when debug=true."}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
