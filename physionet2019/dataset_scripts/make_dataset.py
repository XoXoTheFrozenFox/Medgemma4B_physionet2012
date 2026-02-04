import argparse
import json
import math
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Variable presets
# -----------------------------

PRESET_VARS = {
    "2019": [
        "HR", "MAP", "SBP", "DBP", "RR", "SpO2", "TempC",
        "FiO2", "pH", "PaCO2", "SaO2", "HCO3", "BaseExcess",
        "Lactate", "Creatinine", "WBC", "Glucose", "K", "Platelets",
    ],
    "2012": [
        "HR", "MAP", "SBP", "DBP", "RR", "SpO2", "TempC",
        "Lactate", "Creatinine", "WBC", "pH", "PaO2", "PaCO2",
        "Na", "K", "Glucose",
    ],
}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _linear_slope(ts_hours: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    x = ts_hours - ts_hours.mean()
    y = values - values.mean()
    denom = (x * x).sum()
    if denom <= 1e-12:
        return np.nan
    return float((x * y).sum() / denom)


def featurize_window(df_win: pd.DataFrame, vars_list: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if df_win.empty:
        for v in vars_list:
            out[v] = {"last": np.nan, "min": np.nan, "max": np.nan, "mean": np.nan, "slope_hr": np.nan}
        return out

    t0 = df_win["ts"].min()
    for v in vars_list:
        d = df_win[df_win["var"] == v].sort_values("ts")
        if d.empty:
            out[v] = {"last": np.nan, "min": np.nan, "max": np.nan, "mean": np.nan, "slope_hr": np.nan}
            continue

        vals = d["value"].astype(float).to_numpy()
        ts_h = ((d["ts"] - t0).dt.total_seconds() / 3600.0).to_numpy()

        out[v] = {
            "last": float(vals[-1]),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "slope_hr": _linear_slope(ts_h, vals),
        }
    return out


def rules_label(features: Dict[str, Dict[str, float]]) -> Dict:
    drivers, checks, evidence = [], [], []

    def v(name, key="last"):
        return features.get(name, {}).get(key, np.nan)

    map_last = v("MAP")
    lact_last = v("Lactate")
    if (not math.isnan(map_last) and map_last < 65) or (not math.isnan(lact_last) and lact_last > 2.0):
        drivers.append("possible_hemodynamic_instability")
        checks += ["review_fluids_pressors", "check_lactate_trend", "assess_perfusion"]
        if not math.isnan(map_last):
            evidence.append(f"MAP last={map_last:.1f}")
        if not math.isnan(lact_last):
            evidence.append(f"Lactate last={lact_last:.2f}")

    spo2_last = v("SpO2")
    rr_last = v("RR")
    if (not math.isnan(spo2_last) and spo2_last < 92) or (not math.isnan(rr_last) and rr_last > 24):
        drivers.append("possible_respiratory_compromise")
        checks += ["verify_oxygen_delivery", "consider_ABG_if_available", "review_CXR_if_available"]
        if not math.isnan(spo2_last):
            evidence.append(f"SpO2 last={spo2_last:.1f}")
        if not math.isnan(rr_last):
            evidence.append(f"RR last={rr_last:.1f}")

    cr_slope = v("Creatinine", "slope_hr")
    cr_last = v("Creatinine")
    if (not math.isnan(cr_slope) and cr_slope > 0.02) or (not math.isnan(cr_last) and cr_last > 1.5):
        drivers.append("possible_renal_dysfunction")
        checks += ["review_urine_output_if_available", "review_nephrotoxins", "trend_creatinine"]
        if not math.isnan(cr_last):
            evidence.append(f"Creatinine last={cr_last:.2f}")
        if not math.isnan(cr_slope):
            evidence.append(f"Creatinine slope/hr={cr_slope:.3f}")

    wbc_last = v("WBC")
    temp_last = v("TempC")
    if (not math.isnan(wbc_last) and (wbc_last > 12 or wbc_last < 4)) or (not math.isnan(temp_last) and temp_last > 38.0):
        drivers.append("possible_infection_inflammation")
        checks += ["review_cultures_if_any", "review_antibiotics_if_any", "check_source_control"]
        if not math.isnan(wbc_last):
            evidence.append(f"WBC last={wbc_last:.1f}")
        if not math.isnan(temp_last):
            evidence.append(f"TempC last={temp_last:.1f}")

    status = "worsening" if drivers else "stable"

    narrative = [f"Status: {status}."]
    if drivers:
        narrative.append("Key concerns: " + ", ".join(drivers) + ".")
    else:
        narrative.append("No obvious deterioration signals from available trends.")
    if evidence:
        narrative.append("Evidence: " + "; ".join(evidence[:6]) + ".")
    if checks:
        narrative.append("What to check next: " + ", ".join(list(dict.fromkeys(checks))[:6]) + ".")

    return {
        "status": status,
        "drivers": drivers[:6],
        "what_to_check_next": list(dict.fromkeys(checks))[:10],
        "evidence": evidence[:10],
        "narrative": " ".join(narrative),
        "disclaimer": "Demo only. Not medical advice. Clinician review required.",
    }


def build_prompt(window_text: str, patient_context: str = "") -> str:
    ctx = f"PATIENT_CONTEXT:\n{patient_context}\n\n" if patient_context else ""
    return (
        "You are an ICU trend summarizer for clinicians.\n"
        "Given the last 12 hours of vitals/labs, output STRICT JSON with keys:\n"
        "status, drivers, what_to_check_next, evidence, narrative, disclaimer.\n"
        "Be cautious, state uncertainty, and do not invent missing data.\n\n"
        f"{ctx}"
        f"LAST_12H_WINDOW:\n{window_text}\n"
    )


def window_to_text(features: Dict[str, Dict[str, float]], vars_list: List[str]) -> str:
    def fmt(x):
        return "NA" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"

    lines = []
    for vname in vars_list:
        stats = features.get(vname, {})
        lines.append(
            f"{vname}: last={fmt(stats.get('last'))}, min={fmt(stats.get('min'))}, "
            f"max={fmt(stats.get('max'))}, mean={fmt(stats.get('mean'))}, "
            f"slope_hr={fmt(stats.get('slope_hr'))}"
        )
    return "\n".join(lines)


def _pick_static(d: pd.DataFrame, col: str) -> Optional[float]:
    if col not in d.columns:
        return None
    s = pd.to_numeric(d[col], errors="coerce").dropna()
    return None if s.empty else float(s.iloc[0])


# 🔧 FIXED: dict-safe lookup
def _labels_lookup(labels_map: Optional[dict], stay_id: int, iculos: int) -> Optional[int]:
    if not labels_map:
        return None
    return labels_map.get((stay_id, iculos))


def build_examples(
    df: pd.DataFrame,
    window_hours: int,
    stride_hours: int,
    vars_list: List[str],
    max_windows_per_stay: int,
    min_non_nan: int,
    labels_map: Optional[dict] = None,
) -> List[dict]:

    examples: List[dict] = []

    for stay_id, d in tqdm(df.groupby("stay_id"), desc="stays"):
        d = d.sort_values("ts")
        t_min, t_max = d["ts"].min(), d["ts"].max()
        if pd.isna(t_min) or pd.isna(t_max):
            continue

        ctx_parts = []
        for col in ["age", "gender", "unit1", "unit2"]:
            val = _pick_static(d, col)
            if val is not None:
                ctx_parts.append(f"{col.capitalize()}: {val:.0f}")
        patient_ctx = "\n".join(ctx_parts)

        iculos_min = int(pd.to_numeric(d["iculos"], errors="coerce").min()) if "iculos" in d.columns else None

        anchor = t_min + timedelta(hours=window_hours)
        windows = 0

        while anchor <= t_max:
            start = anchor - timedelta(hours=window_hours)
            win = d[(d["ts"] >= start) & (d["ts"] <= anchor)]
            feats = featurize_window(win, vars_list)

            non_nan = sum(not math.isnan(feats[v]["last"]) for v in vars_list)
            if non_nan >= min_non_nan:
                prompt = build_prompt(window_to_text(feats, vars_list), patient_ctx)
                target_obj = rules_label(feats)

                if labels_map and iculos_min is not None:
                    anchor_iculos = int(iculos_min + round((anchor - t_min).total_seconds() / 3600.0))
                    lbl = _labels_lookup(labels_map, int(stay_id), anchor_iculos)
                    if lbl is not None:
                        target_obj["sepsis_label"] = int(lbl)

                examples.append({
                    "stay_id": int(stay_id),
                    "anchor_ts": str(anchor),
                    "prompt": prompt,
                    "target": json.dumps(target_obj, ensure_ascii=False),
                })
                windows += 1

            if max_windows_per_stay > 0 and windows >= max_windows_per_stay:
                break
            anchor += timedelta(hours=stride_hours)

    return examples


def split_by_stay_id(examples: List[dict], val_frac: float, seed: int):
    stay_ids = list({e["stay_id"] for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(stay_ids)
    n_val = max(1, int(len(stay_ids) * val_frac))
    val_ids = set(stay_ids[:n_val])
    return (
        [e for e in examples if e["stay_id"] not in val_ids],
        [e for e in examples if e["stay_id"] in val_ids],
    )


def write_jsonl(path: str, recs: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_vars_list(vars_arg: str, preset: str) -> List[str]:
    return [v.strip() for v in vars_arg.split(",") if v.strip()] if vars_arg else PRESET_VARS[preset]


def _load_labels_map(labels_csv: str) -> Optional[dict]:
    if not labels_csv:
        return None

    lab = pd.read_csv(labels_csv)
    if lab.empty:
        return None

    required = ["stay_id", "iculos", "sepsis_label"]
    if not all(c in lab.columns for c in required):
        raise ValueError(
            "labels_csv must contain columns: stay_id, iculos, sepsis_label"
        )

    # Convert ONLY required columns
    lab["stay_id"] = pd.to_numeric(lab["stay_id"], errors="coerce")
    lab["iculos"] = pd.to_numeric(lab["iculos"], errors="coerce")
    lab["sepsis_label"] = pd.to_numeric(lab["sepsis_label"], errors="coerce")

    lab = lab.dropna(subset=required)
    if lab.empty:
        return None

    lab["stay_id"] = lab["stay_id"].astype(int)
    lab["iculos"] = lab["iculos"].astype(int)
    lab["sepsis_label"] = lab["sepsis_label"].astype(int)

    return {
        (int(r.stay_id), int(r.iculos)): int(r.sepsis_label)
        for r in lab.itertuples(index=False)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_csv", type=str, required=True)
    ap.add_argument("--out_train_jsonl", type=str, required=True)
    ap.add_argument("--out_val_jsonl", type=str, required=True)
    ap.add_argument("--window_hours", type=int, default=12)
    ap.add_argument("--stride_hours", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_windows_per_stay", type=int, default=10)
    ap.add_argument("--min_non_nan", type=int, default=3)
    ap.add_argument("--preset", type=str, default="2019", choices=["2019", "2012", "auto"])
    ap.add_argument("--vars", type=str, default="")
    ap.add_argument("--labels_csv", type=str, default="")
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["value"] = df["value"].apply(_safe_float)
    df = df[df["ts"].notna() & df["value"].notna()]

    preset = args.preset
    if preset == "auto":
        preset = "2012" if df["var"].isin(["PaO2", "Na"]).any() else "2019"

    vars_list = _parse_vars_list(args.vars, preset)
    df = df[df["var"].isin(vars_list)]

    labels_map = _load_labels_map(args.labels_csv)

    examples = build_examples(
        df,
        args.window_hours,
        args.stride_hours,
        vars_list,
        args.max_windows_per_stay,
        args.min_non_nan,
        labels_map,
    )

    train, val = split_by_stay_id(examples, args.val_frac, args.seed)
    os.makedirs(os.path.dirname(args.out_train_jsonl), exist_ok=True)

    write_jsonl(args.out_train_jsonl, train)
    write_jsonl(args.out_val_jsonl, val)

    print(f"[ok] preset: {preset}")
    print(f"[ok] vars: {vars_list}")
    print(f"[ok] examples total: {len(examples):,}")
    print(f"[ok] train: {len(train):,}")
    print(f"[ok] val:   {len(val):,}")


if __name__ == "__main__":
    main()
