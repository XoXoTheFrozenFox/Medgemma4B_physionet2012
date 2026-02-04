import argparse
import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

"""Convert PhysioNet/CinC Sepsis Challenge (2019-style) per-patient PSV files -> one long-format CSV.

Input: directory containing per-patient .psv files. Files are pipe-separated with a header.
Each row corresponds to one ICU hour (ICULOS).

Output CSV (long format):
  stay_id, ts, iculos, var, value, unit, age, gender, unit1, unit2, hosp_adm_time

Optional labels CSV:
  stay_id, ts, iculos, sepsis_label

Notes:
- Timestamps are synthetic (base_date + ICULOS hours). Only relative time matters.
- Variable name mapping is chosen to match make_dataset.py presets.
"""


# PSV column name -> canonical name
VAR_MAP_2019 = {
    "HR": "HR",
    "O2Sat": "SpO2",
    "Temp": "TempC",
    "SBP": "SBP",
    "MAP": "MAP",
    "DBP": "DBP",
    "Resp": "RR",
    "EtCO2": "EtCO2",
    "BaseExcess": "BaseExcess",
    "HCO3": "HCO3",
    "FiO2": "FiO2",
    "pH": "pH",
    "PaCO2": "PaCO2",
    "SaO2": "SaO2",
    "AST": "AST",
    "BUN": "BUN",
    "Alkalinephos": "Alkalinephos",
    "Calcium": "Calcium",
    "Chloride": "Chloride",
    "Creatinine": "Creatinine",
    "Bilirubin_direct": "Bilirubin_direct",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "Magnesium": "Magnesium",
    "Phosphate": "Phosphate",
    "Potassium": "K",
    "Bilirubin_total": "Bilirubin",
    "TroponinI": "TroponinI",
    "Hct": "Hct",
    "Hgb": "Hgb",
    "PTT": "PTT",
    "WBC": "WBC",
    "Fibrinogen": "Fibrinogen",
    "Platelets": "Platelets",
}


UNIT_MAP = {
    "HR": "bpm",
    "MAP": "mmHg",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "RR": "breaths/min",
    "SpO2": "%",
    "TempC": "C",
    "EtCO2": "mmHg",
    "FiO2": "fraction",
    "pH": "",
    "PaCO2": "mmHg",
    "SaO2": "%",
    "BaseExcess": "mmol/L",
    "HCO3": "mmol/L",
    "Lactate": "mmol/L",
    "Creatinine": "mg/dL",
    "WBC": "K/uL",
    "Glucose": "mg/dL",
    "K": "mEq/L",
    "Platelets": "K/uL",
    "Hct": "%",
    "Hgb": "g/dL",
    "Bilirubin": "mg/dL",
    "Bilirubin_direct": "mg/dL",
    "AST": "U/L",
    "BUN": "mg/dL",
    "Alkalinephos": "U/L",
    "Calcium": "mg/dL",
    "Chloride": "mEq/L",
    "Magnesium": "mg/dL",
    "Phosphate": "mg/dL",
    "TroponinI": "ng/mL",
    "PTT": "s",
    "Fibrinogen": "mg/dL",
}


def _extract_stay_id(path: str, fallback: int = 0) -> int:
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    if not m:
        return int(fallback)
    return int(m.group(1))



def _first_non_nan(series: pd.Series):
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return np.nan
    return float(s.iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psv_dir", type=str, required=True, help="Folder with per-patient .psv files")
    ap.add_argument("--pattern", type=str, default="*.psv", help="Glob pattern (default: *.psv)")
    ap.add_argument("--out_csv", type=str, default="data/icu_long.csv")
    ap.add_argument(
        "--out_labels_csv",
        type=str,
        default="data/sepsis_labels.csv",
        help="Optional per-hour label file. Set to '' to disable.",
    )
    ap.add_argument("--base_date", type=str, default="2025-01-01", help="Base date for timestamps (YYYY-MM-DD)")
    args = ap.parse_args()

    base = datetime.strptime(args.base_date, "%Y-%m-%d")

    paths = sorted(glob.glob(os.path.join(args.psv_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched {os.path.join(args.psv_dir, args.pattern)}")

    long_frames = []
    label_frames = []
    kept_files = 0

    for p in paths:
        stay_id = _extract_stay_id(p)

        try:
            df = pd.read_csv(p, sep="|", engine="python")
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
            continue

        if df.empty:
            continue

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        # ICULOS (hours since ICU admission)
        if "ICULOS" in df.columns:
            iculos = pd.to_numeric(df["ICULOS"], errors="coerce")
        else:
            iculos = pd.Series(np.arange(len(df)), dtype=float)

        # Fallback for missing/NaN ICULOS
        iculos = iculos.fillna(pd.Series(np.arange(len(df)), dtype=float))
        df["iculos"] = iculos.astype(int)
        df["ts"] = df["iculos"].apply(lambda h: base + timedelta(hours=int(h)))

        # Static fields
        age = _first_non_nan(df["Age"]) if "Age" in df.columns else np.nan
        gender = _first_non_nan(df["Gender"]) if "Gender" in df.columns else np.nan
        unit1 = _first_non_nan(df["Unit1"]) if "Unit1" in df.columns else np.nan
        unit2 = _first_non_nan(df["Unit2"]) if "Unit2" in df.columns else np.nan
        hosp_adm_time = _first_non_nan(df["HospAdmTime"]) if "HospAdmTime" in df.columns else np.nan

        # Optional labels
        if args.out_labels_csv and "SepsisLabel" in df.columns:
            lab = pd.DataFrame(
                {
                    "stay_id": stay_id,
                    "ts": df["ts"].astype(str),
                    "iculos": df["iculos"].astype(int),
                    "sepsis_label": pd.to_numeric(df["SepsisLabel"], errors="coerce"),
                }
            )
            lab = lab[lab["sepsis_label"].notna()].copy()
            if not lab.empty:
                lab["sepsis_label"] = lab["sepsis_label"].astype(int)
                label_frames.append(lab)

        # Melt physiologic variables to long format
        cols_present = [c for c in VAR_MAP_2019.keys() if c in df.columns]
        if not cols_present:
            continue

        wide = df[["ts", "iculos"] + cols_present].copy()
        melted = wide.melt(id_vars=["ts", "iculos"], var_name="raw_var", value_name="value")
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted[melted["value"].notna()].copy()
        if melted.empty:
            continue

        melted["stay_id"] = stay_id
        melted["var"] = melted["raw_var"].map(VAR_MAP_2019)
        melted["unit"] = melted["var"].map(lambda v: UNIT_MAP.get(v, ""))
        melted["age"] = age
        melted["gender"] = gender
        melted["unit1"] = unit1
        melted["unit2"] = unit2
        melted["hosp_adm_time"] = hosp_adm_time

        melted = melted[[
            "stay_id",
            "ts",
            "iculos",
            "var",
            "value",
            "unit",
            "age",
            "gender",
            "unit1",
            "unit2",
            "hosp_adm_time",
        ]]
        melted["ts"] = melted["ts"].astype(str)

        long_frames.append(melted)
        kept_files += 1

    if not long_frames:
        raise SystemExit("No usable data extracted. Check PSV files and header names.")

    out = pd.concat(long_frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"[ok] wrote {len(out):,} rows -> {args.out_csv}")
    print(f"[ok] converted files: {kept_files:,}")
    print("vars:", sorted(out["var"].unique().tolist()))

    if args.out_labels_csv and label_frames:
        labels = pd.concat(label_frames, ignore_index=True)
        os.makedirs(os.path.dirname(args.out_labels_csv) or ".", exist_ok=True)
        labels.to_csv(args.out_labels_csv, index=False)
        print(f"[ok] wrote {len(labels):,} label rows -> {args.out_labels_csv}")


if __name__ == "__main__":
    main()
