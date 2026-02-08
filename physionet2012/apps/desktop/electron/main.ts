// electron/main.ts
import { app, BrowserWindow, ipcMain, shell, nativeImage } from "electron";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

type Preset = "quick" | "normal" | "detailed";

type AnalyzeReq = {
  preset: Preset;
  note: string;
  debug?: boolean;
  max_new_tokens?: number | null;
  temperature?: number | null;
  top_p?: number | null;
  repetition_penalty?: number | null;
};

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

let mainWindow: BrowserWindow | null = null;

// ✅ ESM-safe __dirname / __filename
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ✅ Helps Windows taskbar grouping + identity
try {
  app.setAppUserModelId("com.medgemma.clinical.console");
} catch {}

function firstExisting(candidates: string[]): string | null {
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return null;
}

function resolveAsset(relPath: string): string {
  const appPath = app.getAppPath();
  const cwd = process.cwd();

  const candidates = [
    // run-from-project-root
    path.join(cwd, relPath),
    // when packaged or when appPath points to app.asar root
    path.join(appPath, relPath),
    // when dist-electron/main.js is deeper
    path.join(__dirname, "..", relPath),
    path.join(__dirname, "../..", relPath),
  ];

  const found = firstExisting(candidates);

  console.log(
    `[electron] resolveAsset("${relPath}") candidates:\n` +
      candidates.map((c) => "  " + c).join("\n")
  );
  console.log(`[electron] resolveAsset("${relPath}") resolved:`, found ?? "(NOT FOUND)");

  return found ?? candidates[0];
}

function resolvePreloadPath(): string {
  const appPath = app.getAppPath();
  const cwd = process.cwd();

  const candidates = [
    // If built output lands next to main.js in dist-electron
    path.join(__dirname, "preload.js"),
    path.join(__dirname, "preload.cjs"),
    path.join(__dirname, "preload.mjs"),

    // If built output is nested
    path.join(__dirname, "preload", "index.js"),
    path.join(__dirname, "preload", "index.cjs"),
    path.join(__dirname, "preload", "index.mjs"),

    // Project-root guesses
    path.join(appPath, "dist-electron", "preload.js"),
    path.join(appPath, "dist-electron", "preload", "index.js"),
    path.join(cwd, "dist-electron", "preload.js"),
    path.join(cwd, "dist-electron", "preload", "index.js"),

    // Some templates emit into dist-electron/electron/*
    path.join(appPath, "dist-electron", "electron", "preload.js"),
    path.join(cwd, "dist-electron", "electron", "preload.js"),
  ];

  const found = firstExisting(candidates);

  console.log(
    "[electron] preload candidates:\n" + candidates.map((c) => "  " + c).join("\n")
  );
  console.log("[electron] preload resolved:", found ?? "(NOT FOUND)");

  return found ?? candidates[0];
}

function resolveIndexHtml(): string {
  const appPath = app.getAppPath();
  const cwd = process.cwd();

  const candidates = [
    // Common build outputs
    path.join(__dirname, "../dist/index.html"),
    path.join(__dirname, "../dist/renderer/index.html"),
    path.join(__dirname, "../renderer/index.html"),
    path.join(__dirname, "../index.html"),

    // project-root guesses
    path.join(appPath, "dist", "index.html"),
    path.join(cwd, "dist", "index.html"),
  ];

  const found = firstExisting(candidates);
  console.log("[electron] index.html resolved:", found ?? "(NOT FOUND)");
  return found ?? candidates[0];
}

function createWindow() {
  const preloadPath = resolvePreloadPath();

  // ✅ Icon: resolve + nativeImage (more reliable than raw string)
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
      sandbox: false,
    },
  });

  // Open external links in the default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    mainWindow.loadURL(devUrl);
    // mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(resolveIndexHtml());
  }
}

// ✅ prevent “UnhandledPromiseRejectionWarning”
process.on("unhandledRejection", (err) => {
  console.error("[electron] unhandledRejection:", err);
});

app
  .whenReady()
  .then(() => {
    createWindow();

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  })
  .catch((err) => {
    console.error("[electron] app.whenReady failed:", err);
  });

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

function withTimeout(ms: number) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { controller, cancel: () => clearTimeout(id) };
}

ipcMain.handle("medapi:health", async () => {
  const base = process.env.MEDAPI_BASE_URL || DEFAULT_API_BASE;
  const { controller, cancel } = withTimeout(15_000);
  try {
    const r = await fetch(`${base}/health`, { signal: controller.signal });
    if (!r.ok) throw new Error(`Health HTTP ${r.status}`);
    return await r.json();
  } finally {
    cancel();
  }
});

ipcMain.handle("medapi:analyze", async (_evt, payload: AnalyzeReq) => {
  const base = process.env.MEDAPI_BASE_URL || DEFAULT_API_BASE;

  if (!payload?.note?.trim()) {
    return { ok: false, error: "Empty note." };
  }

  const body = {
    preset: payload.preset,
    note: payload.note,
    debug: !!payload.debug,
    max_new_tokens: payload.max_new_tokens ?? undefined,
    temperature: payload.temperature ?? undefined,
    top_p: payload.top_p ?? undefined,
    repetition_penalty: payload.repetition_penalty ?? undefined,
  };

  const { controller, cancel } = withTimeout(240_000);
  try {
    const r = await fetch(`${base}/v1/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const text = await r.text();
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}: ${text}` };

    const json = JSON.parse(text);
    return { ok: true, data: json };
  } catch (e: any) {
    const msg =
      e?.name === "AbortError" ? "Request timed out." : (e?.message || String(e));
    return { ok: false, error: msg };
  } finally {
    cancel();
  }
});
