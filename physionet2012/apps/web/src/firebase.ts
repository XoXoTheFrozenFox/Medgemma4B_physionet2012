// src/firebase.ts
import { initializeApp, getApps } from "firebase/app";
import { getAuth, setPersistence, browserLocalPersistence } from "firebase/auth";

function must(name: string, v: string | undefined): string {
  if (!v || !String(v).trim()) throw new Error(`Missing env ${name}. Check .env.local and restart dev server.`);
  return v;
}

export let firebaseInitError: string | null = null;

export const auth = (() => {
  try {
    const config = {
      apiKey: must("VITE_FIREBASE_API_KEY", import.meta.env.VITE_FIREBASE_API_KEY),
      authDomain: must("VITE_FIREBASE_AUTH_DOMAIN", import.meta.env.VITE_FIREBASE_AUTH_DOMAIN),
      projectId: must("VITE_FIREBASE_PROJECT_ID", import.meta.env.VITE_FIREBASE_PROJECT_ID),
      appId: must("VITE_FIREBASE_APP_ID", import.meta.env.VITE_FIREBASE_APP_ID),
      storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
      messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
    };

    const app = getApps().length ? getApps()[0] : initializeApp(config);
    const a = getAuth(app);

    // Keep session after refresh
    setPersistence(a, browserLocalPersistence).catch(() => {});

    return a;
  } catch (e: any) {
    firebaseInitError = e?.message || String(e);
    return null;
  }
})();
