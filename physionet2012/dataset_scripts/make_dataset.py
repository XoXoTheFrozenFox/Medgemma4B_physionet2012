import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Variables expected downstream (must match PhysioNet converter mapping)
CORE_VITALS = ["HR", "MAP", "SBP", "DBP", "RR", "SpO2", "TempC"]
CORE_LABS = ["Lactate", "Creatinine", "WBC", "pH", "PaO2", "PaCO2", "Na", "K", "Glucose"]
ALL_VARS = CORE_VITALS + CORE_LABS


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
    out = {}
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
    """Weak supervision labels (reproducible demo)."""
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


def build_prompt(window_text: str) -> str:
    return (
        "You are an ICU trend summarizer for clinicians.\n"
        "Given the last 12 hours of vitals/labs, output STRICT JSON with keys:\n"
        "status, drivers, what_to_check_next, evidence, narrative, disclaimer.\n"
        "Be cautious, state uncertainty, and do not invent missing data.\n\n"
        f"LAST_12H_WINDOW:\n{window_text}\n"
    )


def window_to_text(features: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for vname, stats in features.items():
        def fmt(x):
            return "NA" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"
        lines.append(
            f"{vname}: last={fmt(stats['last'])}, min={fmt(stats['min'])}, max={fmt(stats['max'])}, "
            f"mean={fmt(stats['mean'])}, slope_hr={fmt(stats['slope_hr'])}"
        )
    return "\n".join(lines)


def build_examples(
    df: pd.DataFrame,
    window_hours: int,
    stride_hours: int,
    vars_list: List[str],
    max_windows_per_stay: int,
) -> List[dict]:
    examples: List[dict] = []

    if df.empty:
        return examples

    for stay_id, d in tqdm(df.groupby("stay_id"), desc="stays"):
        d = d.sort_values("ts")
        t_min, t_max = d["ts"].min(), d["ts"].max()
        if pd.isna(t_min) or pd.isna(t_max):
            continue

        anchor = t_min + timedelta(hours=window_hours)

        windows = 0
        while anchor <= t_max:
            start = anchor - timedelta(hours=window_hours)
            win = d[(d["ts"] >= start) & (d["ts"] <= anchor)]
            feats = featurize_window(win, vars_list)
            window_text = window_to_text(feats)

            # skip very empty windows
            non_nan = sum(0 if math.isnan(feats[v]["last"]) else 1 for v in vars_list)
            if non_nan >= 3:
                prompt = build_prompt(window_text)
                target = json.dumps(rules_label(feats), ensure_ascii=False)
                examples.append(
                    {"stay_id": int(stay_id), "anchor_ts": str(anchor), "prompt": prompt, "target": target}
                )
                windows += 1

            if max_windows_per_stay > 0 and windows >= max_windows_per_stay:
                break

            anchor += timedelta(hours=stride_hours)

    return examples


def split_by_stay_id(examples: List[dict], val_frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    stay_ids = sorted({ex["stay_id"] for ex in examples})
    if not stay_ids:
        return [], []

    rng = np.random.default_rng(seed)
    rng.shuffle(stay_ids)
    n_val = max(1, int(len(stay_ids) * val_frac))
    val_set = set(stay_ids[:n_val])

    train = [ex for ex in examples if ex["stay_id"] not in val_set]
    val = [ex for ex in examples if ex["stay_id"] in val_set]
    return train, val


def write_jsonl(path: Path, recs: List[dict]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--long_csv",
        type=Path,
        default=Path("data") / "icu_long.csv",
        help="Long-format CSV: stay_id, ts, var, value, unit",
    )
    ap.add_argument("--out_train_jsonl", type=Path, default=Path("data") / "train.jsonl")
    ap.add_argument("--out_val_jsonl", type=Path, default=Path("data") / "val.jsonl")
    ap.add_argument("--window_hours", type=int, default=12)
    ap.add_argument("--stride_hours", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_windows_per_stay", type=int, default=10, help="limit windows per stay (0 = unlimited)")
    args = ap.parse_args()

    long_csv = args.long_csv.expanduser()

    if not long_csv.exists():
        raise SystemExit(f"long_csv not found: {long_csv}")

    df = pd.read_csv(long_csv)
    if df.empty:
        # Still write empty outputs for pipeline friendliness
        write_jsonl(args.out_train_jsonl, [])
        write_jsonl(args.out_val_jsonl, [])
        print("[warn] long_csv is empty; wrote empty train/val JSONL.")
        return

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["value"] = df["value"].apply(_safe_float)
    df = df[df["var"].isin(ALL_VARS)].copy()
    df = df[df["ts"].notna()].copy()

    examples = build_examples(df, args.window_hours, args.stride_hours, ALL_VARS, args.max_windows_per_stay)
    train, val = split_by_stay_id(examples, args.val_frac, args.seed)

    write_jsonl(args.out_train_jsonl, train)
    write_jsonl(args.out_val_jsonl, val)

    print(f"[ok] examples total: {len(examples):,}")
    print(f"[ok] train: {len(train):,} -> {args.out_train_jsonl.expanduser()}")
    print(f"[ok] val:   {len(val):,} -> {args.out_val_jsonl.expanduser()}")


if __name__ == "__main__":
    main()
