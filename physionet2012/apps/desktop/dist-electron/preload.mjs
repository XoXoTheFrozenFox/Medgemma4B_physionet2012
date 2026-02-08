"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("medapi", {
  health: () => electron.ipcRenderer.invoke("medapi:health"),
  analyze: (payload) => electron.ipcRenderer.invoke("medapi:analyze", payload),
  copyText: async (text) => {
    electron.clipboard.writeText(String(text ?? ""));
  }
});
