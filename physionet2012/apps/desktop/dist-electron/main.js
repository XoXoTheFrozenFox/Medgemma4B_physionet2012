import { app, BrowserWindow, ipcMain, nativeImage, shell } from "electron";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
const DEFAULT_API_BASE = "http://127.0.0.1:8000";
let mainWindow = null;
const __filename$1 = fileURLToPath(import.meta.url);
const __dirname$1 = path.dirname(__filename$1);
try {
  app.setAppUserModelId("com.medgemma.clinical.console");
} catch {
}
function firstExisting(candidates) {
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {
    }
  }
  return null;
}
function resolveAsset(relPath) {
  const appPath = app.getAppPath();
  const cwd = process.cwd();
  const candidates = [
    // run-from-project-root
    path.join(cwd, relPath),
    // when packaged or when appPath points to app.asar root
    path.join(appPath, relPath),
    // when dist-electron/main.js is deeper
    path.join(__dirname$1, "..", relPath),
    path.join(__dirname$1, "../..", relPath)
  ];
  const found = firstExisting(candidates);
  console.log(
    `[electron] resolveAsset("${relPath}") candidates:
` + candidates.map((c) => "  " + c).join("\n")
  );
  console.log(`[electron] resolveAsset("${relPath}") resolved:`, found ?? "(NOT FOUND)");
  return found ?? candidates[0];
}
function resolvePreloadPath() {
  const appPath = app.getAppPath();
  const cwd = process.cwd();
  const candidates = [
    // If built output lands next to main.js in dist-electron
    path.join(__dirname$1, "preload.js"),
    path.join(__dirname$1, "preload.cjs"),
    path.join(__dirname$1, "preload.mjs"),
    // If built output is nested
    path.join(__dirname$1, "preload", "index.js"),
    path.join(__dirname$1, "preload", "index.cjs"),
    path.join(__dirname$1, "preload", "index.mjs"),
    // Project-root guesses
    path.join(appPath, "dist-electron", "preload.js"),
    path.join(appPath, "dist-electron", "preload", "index.js"),
    path.join(cwd, "dist-electron", "preload.js"),
    path.join(cwd, "dist-electron", "preload", "index.js"),
    // Some templates emit into dist-electron/electron/*
    path.join(appPath, "dist-electron", "electron", "preload.js"),
    path.join(cwd, "dist-electron", "electron", "preload.js")
  ];
  const found = firstExisting(candidates);
  console.log(
    "[electron] preload candidates:\n" + candidates.map((c) => "  " + c).join("\n")
  );
  console.log("[electron] preload resolved:", found ?? "(NOT FOUND)");
  return found ?? candidates[0];
}
function resolveIndexHtml() {
  const appPath = app.getAppPath();
  const cwd = process.cwd();
  const candidates = [
    // Common build outputs
    path.join(__dirname$1, "../dist/index.html"),
    path.join(__dirname$1, "../dist/renderer/index.html"),
    path.join(__dirname$1, "../renderer/index.html"),
    path.join(__dirname$1, "../index.html"),
    // project-root guesses
    path.join(appPath, "dist", "index.html"),
    path.join(cwd, "dist", "index.html")
  ];
  const found = firstExisting(candidates);
  console.log("[electron] index.html resolved:", found ?? "(NOT FOUND)");
  return found ?? candidates[0];
}
function createWindow() {
  const preloadPath = resolvePreloadPath();
  const iconPath = resolveAsset(path.join("assets", "icon.ico"));
  const iconImg = nativeImage.createFromPath(iconPath);
  if (iconImg.isEmpty()) {
    console.warn(
      "[electron] icon image is empty. Check that assets/icon.ico exists and is a real .ico file."
    );
  }
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 980,
    minHeight: 680,
    backgroundColor: "#0b0b10",
    title: "MedGemma Clinical Console",
    icon: iconImg,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      // ✅ improves preload reliability in dev; harden later if you want
      sandbox: false
    }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    mainWindow.loadURL(devUrl);
  } else {
    mainWindow.loadFile(resolveIndexHtml());
  }
}
process.on("unhandledRejection", (err) => {
  console.error("[electron] unhandledRejection:", err);
});
app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
}).catch((err) => {
  console.error("[electron] app.whenReady failed:", err);
});
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
function withTimeout(ms) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { controller, cancel: () => clearTimeout(id) };
}
ipcMain.handle("medapi:health", async () => {
  const base = process.env.MEDAPI_BASE_URL || DEFAULT_API_BASE;
  const { controller, cancel } = withTimeout(15e3);
  try {
    const r = await fetch(`${base}/health`, { signal: controller.signal });
    if (!r.ok) throw new Error(`Health HTTP ${r.status}`);
    return await r.json();
  } finally {
    cancel();
  }
});
ipcMain.handle("medapi:analyze", async (_evt, payload) => {
  var _a;
  const base = process.env.MEDAPI_BASE_URL || DEFAULT_API_BASE;
  if (!((_a = payload == null ? void 0 : payload.note) == null ? void 0 : _a.trim())) {
    return { ok: false, error: "Empty note." };
  }
  const body = {
    preset: payload.preset,
    note: payload.note,
    debug: !!payload.debug,
    max_new_tokens: payload.max_new_tokens ?? void 0,
    temperature: payload.temperature ?? void 0,
    top_p: payload.top_p ?? void 0,
    repetition_penalty: payload.repetition_penalty ?? void 0
  };
  const { controller, cancel } = withTimeout(24e4);
  try {
    const r = await fetch(`${base}/v1/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal
    });
    const text = await r.text();
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}: ${text}` };
    const json = JSON.parse(text);
    return { ok: true, data: json };
  } catch (e) {
    const msg = (e == null ? void 0 : e.name) === "AbortError" ? "Request timed out." : (e == null ? void 0 : e.message) || String(e);
    return { ok: false, error: msg };
  } finally {
    cancel();
  }
});
