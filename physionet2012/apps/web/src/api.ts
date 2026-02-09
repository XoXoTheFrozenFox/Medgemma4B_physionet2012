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

// We call Vite proxy endpoints (avoids CORS):
// GET  /api/health
// POST /api/v1/analyze
const API = "/api";

export async function health(): Promise<any> {
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error(`Health HTTP ${r.status}`);
    return await r.json();
  } catch (e: any) {
    // keep shape similar to other calls (optional)
    throw new Error(e?.message || String(e));
  }
}

export async function analyze(
  payload: AnalyzeReq
): Promise<{ ok: boolean; data?: AnalyzeResp; error?: string }> {
  if (!payload?.note?.trim()) return { ok: false, error: "Empty note." };

  try {
    const r = await fetch(`${API}/v1/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await r.text();
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}: ${text}` };

    const json = JSON.parse(text) as AnalyzeResp;
    return { ok: true, data: json };
  } catch (e: any) {
    return { ok: false, error: e?.message || String(e) };
  }
}
