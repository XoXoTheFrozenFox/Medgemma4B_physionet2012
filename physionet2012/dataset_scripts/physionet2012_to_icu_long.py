import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# Map PhysioNet 2012 variable names -> the trend-copilot variables
VAR_MAP = {
    "HR": "HR",
    "MAP": "MAP",
    "SysABP": "SBP",
    "DiasABP": "DBP",
    "RespRate": "RR",
    "Temp": "TempC",
    # PhysioNet provides SaO2 (arterial). For a vitals trend summary demo, mapping to SpO2 is acceptable.
    "SaO2": "SpO2",
    "Lactate": "Lactate",
    "Creatinine": "Creatinine",
    "WBC": "WBC",
    "pH": "pH",
    "PaO2": "PaO2",
    "PaCO2": "PaCO2",
    "Na": "Na",
    "K": "K",
    "Glucose": "Glucose",
}

UNIT_MAP = {
    "HR": "bpm",
    "MAP": "mmHg",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "RR": "bpm",
    "SpO2": "%",
    "TempC": "C",
    "Lactate": "mmol/L",
    "Creatinine": "mg/dL",
    "WBC": "K/uL",
    "pH": "",
    "PaO2": "mmHg",
    "PaCO2": "mmHg",
    "Na": "mEq/L",
    "K": "mEq/L",
    "Glucose": "mg/dL",
}


def parse_time_to_minutes(t: str) -> int:
    """
    PhysioNet 2012 time format is usually HH:MM, where HH can exceed 24.
    Returns minutes from start. Raises ValueError if invalid.
    """
    t = str(t).strip()
    hh, mm = t.split(":")
    return int(hh) * 60 + int(mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_a_dir", type=Path, required=True, help="Path to extracted PhysioNet 2012 set-a folder")
    ap.add_argument("--out_csv", type=Path, default=Path("data") / "icu_long.csv")
    ap.add_argument("--base_date", type=str, default="2025-01-01", help="Base date for timestamps (YYYY-MM-DD)")
    ap.add_argument("--pattern", type=str, default="*.txt", help="File pattern inside set-a (default: *.txt)")
    args = ap.parse_args()

    set_a_dir = args.set_a_dir.expanduser()
    out_csv = args.out_csv.expanduser()

    if not set_a_dir.exists() or not set_a_dir.is_dir():
        raise SystemExit(f"set-a directory not found: {set_a_dir}")

    base = datetime.strptime(args.base_date, "%Y-%m-%d")

    paths = sorted(set_a_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matching {args.pattern} found in {set_a_dir}")

    rows = []
    kept_files = 0

    for p in paths:
        # Filename is typically a numeric ID like 132539.txt
        stem = p.stem
        try:
            stay_id = int(stem)
        except ValueError:
            # Skip any non-numeric filenames
            continue

        # PhysioNet 2012: CSV columns: Time,Parameter,Value
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        df = df.rename(columns={"Time": "time", "Parameter": "param", "Value": "value"})
        if not {"time", "param", "value"}.issubset(df.columns):
            continue

        df = df[df["param"].isin(VAR_MAP.keys())].copy()
        if df.empty:
            continue

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[df["value"].notna()].copy()
        if df.empty:
            continue

        # Convert minutes to a pseudo timestamp
        try:
            df["minutes"] = df["time"].astype(str).apply(parse_time_to_minutes)
        except Exception:
            # If time parsing fails for a file, skip it
            continue

        df["ts"] = df["minutes"].apply(lambda m: base + timedelta(minutes=int(m)))

        df["var"] = df["param"].map(VAR_MAP)
        df["unit"] = df["var"].map(lambda v: UNIT_MAP.get(v, ""))

        kept_files += 1
        rows.extend(
            zip(
                [stay_id] * len(df),
                df["ts"].astype(str).tolist(),
                df["var"].tolist(),
                df["value"].astype(float).tolist(),
                df["unit"].tolist(),
            )
        )

    out = pd.DataFrame(rows, columns=["stay_id", "ts", "var", "value", "unit"])

    # Cross-platform safe mkdir (also fine if out_csv is just a filename)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv, index=False)

    print(f"[ok] wrote {len(out):,} rows -> {out_csv}")
    print(f"[ok] converted files: {kept_files:,}")
    if not out.empty:
        print("vars:", sorted(out["var"].unique().tolist()))
    else:
        print("[warn] output CSV is empty (no matching variables/values found).")


if __name__ == "__main__":
    main()
