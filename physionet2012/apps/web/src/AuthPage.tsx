// src/AuthPage.tsx
import React, { useState } from "react";
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "./firebase";

function friendly(code?: string, fallback?: string) {
  switch (code) {
    case "auth/invalid-email":
      return "Invalid email.";
    case "auth/user-not-found":
      return "No account found for that email.";
    case "auth/wrong-password":
      return "Incorrect password.";
    case "auth/weak-password":
      return "Password too weak (use 8+ chars).";
    case "auth/email-already-in-use":
      return "Email already in use. Try signing in.";
    case "auth/network-request-failed":
      return "Network error.";
    case "auth/too-many-requests":
      return "Too many attempts. Try again later.";
    default:
      return fallback || "Auth error.";
  }
}

export default function AuthPage() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function submit() {
    setErr("");
    if (!auth) return setErr("Firebase not initialized. Check .env.local.");

    const e = email.trim();
    if (!e) return setErr("Enter your email.");
    if (!pw) return setErr("Enter your password.");

    if (mode === "signup") {
      if (pw.length < 8) return setErr("Use at least 8 characters.");
      if (pw !== pw2) return setErr("Passwords do not match.");
    }

    setBusy(true);
    try {
      if (mode === "signin") {
        await signInWithEmailAndPassword(auth, e, pw);
      } else {
        await createUserWithEmailAndPassword(auth, e, pw);
      }
    } catch (ex: any) {
      setErr(friendly(ex?.code, ex?.message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="authBody">
      <div className="authTop">
        <div className="authTitle">🔐 {mode === "signin" ? "Sign in" : "Create account"}</div>
        <div className="authSub">Email/Password (Firebase Auth)</div>
      </div>

      <div className="authForm">
        <label className="field">
          <span className="fieldLabel">Email</span>
          <input
            className="input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@example.com"
            autoComplete="email"
          />
        </label>

        <label className="field">
          <span className="fieldLabel">Password</span>
          <input
            className="input"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            placeholder="••••••••"
            type="password"
            autoComplete={mode === "signin" ? "current-password" : "new-password"}
          />
        </label>

        {mode === "signup" && (
          <label className="field">
            <span className="fieldLabel">Confirm password</span>
            <input
              className="input"
              value={pw2}
              onChange={(e) => setPw2(e.target.value)}
              placeholder="••••••••"
              type="password"
              autoComplete="new-password"
            />
          </label>
        )}

        <div className="authBtns">
          <button className="btn primary" onClick={submit} disabled={busy}>
            {busy ? "Working…" : mode === "signin" ? "Sign in" : "Sign up"}
          </button>

          <button
            className="btn"
            onClick={() => {
              setErr("");
              setMode(mode === "signin" ? "signup" : "signin");
            }}
            disabled={busy}
          >
            {mode === "signin" ? "Create account" : "I have an account"}
          </button>
        </div>

        {!!err && <div className="toast">⚠ {err}</div>}
      </div>
    </div>
  );
}
