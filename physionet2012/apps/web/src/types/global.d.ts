// src/types/global.d.ts
export {};

import type { AnalyzeReq, AnalyzeRes, HealthRes } from "./medapi";

declare global {
  interface Window {
    medapi?: {
      health: () => Promise<HealthRes>;
      analyze: (payload: AnalyzeReq) => Promise<AnalyzeRes>;
      copyText?: (text: string) => Promise<void> | void;
    };
  }
}
