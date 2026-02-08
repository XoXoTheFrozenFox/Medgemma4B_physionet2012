import { contextBridge, ipcRenderer, clipboard } from "electron";

contextBridge.exposeInMainWorld("medapi", {
  health: () => ipcRenderer.invoke("medapi:health"),
  analyze: (payload: any) => ipcRenderer.invoke("medapi:analyze", payload),
  copyText: async (text: string) => {
    clipboard.writeText(String(text ?? ""));
  },
});
