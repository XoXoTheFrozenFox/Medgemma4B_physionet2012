export type Preset = "quick" | "normal" | "detailed";

export type AnalyzeReq = {
  preset: Preset;
  note: string;
  debug?: boolean;
  max_new_tokens?: number | null;
  temperature?: number | null;
  top_p?: number | null;
  repetition_penalty?: number | null;
};

export type AnalyzeResp = {
  preset: Preset;
  reply: string;
  meta?: any;
};

function withTimeout(ms: number) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { controller, cancel: () => clearTimeout(id) };
}

// We call Vite proxy endpoints (avoids CORS):
// GET  /api/health
// POST /api/v1/analyze
const API = "/api";

export async function health(): Promise<any> {
  const { controller, cancel } = withTimeout(15000);
  try {
    const r = await fetch(`${API}/health`, { signal: controller.signal });
    if (!r.ok) throw new Error(`Health HTTP ${r.status}`);
    return await r.json();
  } finally {
    cancel();
  }
}

export async function analyze(payload: AnalyzeReq): Promise<{ ok: boolean; data?: AnalyzeResp; error?: string }> {
  if (!payload?.note?.trim()) return { ok: false, error: "Empty note." };

  const { controller, cancel } = withTimeout(240000);
  try {
    const r = await fetch(`${API}/v1/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await r.text();
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}: ${text}` };

    const json = JSON.parse(text);
    return { ok: true, data: json };
  } catch (e: any) {
    const msg = e?.name === "AbortError" ? "Request timed out." : (e?.message || String(e));
    return { ok: false, error: msg };
  } finally {
    cancel();
  }
}
