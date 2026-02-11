// src/types/medapi.ts
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

export type AnalyzeRes = {
  ok: boolean;
  error?: string;
  data?: {
    reply?: string;
    meta?: any;
    [k: string]: any;
  };
};

export type HealthRes = {
  ok: boolean;
  error?: string;
  gpu?: string;
  is_loaded_in_4bit?: boolean;
  [k: string]: any;
};
