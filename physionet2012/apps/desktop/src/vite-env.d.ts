/// <reference types="vite/client" />

declare global {
  interface Window {
    medapi?: {
      health: () => Promise<any>;
      analyze: (payload: any) => Promise<{ ok: boolean; error?: string; data?: any }>;
      copyText: (text: string) => Promise<void>;
    };
  }
}

export {};
