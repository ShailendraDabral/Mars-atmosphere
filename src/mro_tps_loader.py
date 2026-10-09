"""
Loader for Mars radio-occultation temperature-pressure profiles (PDS3 TPS products):
  - MRO: DATA_SET_ID = MRO-M-RSS-5-TPS-V1.0  (volume mrors_2001; 27-column header)
  - MGS: DATA_SET_ID = MGS-M-RSS-5-TPS-V1.0  (volume mors_1101;  29-column header)
The format of every file is detected automatically, and both missions are written to one common schema
with a "mission" column. Geopotential is re-referenced to a common constant (GEOPOT_REF_COMMON) because
the two missions used different reference values.

HOW TO USE
  1. Put the path of each month folder in MONTH_FOLDERS below.
  2. Set OUTPUT_DIR.
  3. Run:  python mro_tps_loader.py
     (or in a notebook:  %run mro_tps_loader.py)

Each folder is searched (including its sub-folders) for .TPS / .tps files. Folders can hold MRO data,
MGS data, or both.
All profiles are combined and written to OUTPUT_DIR:
  - mro_tps_combined.csv : FINAL FILE. One row per vertical level, all months,
                           with the profile metadata attached to every row.
  - mro_tps_profiles.csv : one row per profile (metadata + quality/physics checks).

You can also pass folders on the command line instead of editing the list:
  python mro_tps_loader.py /path/month1 /path/month2 --out ./processed
"""

# ============================ EDIT THESE ============================
# Rules: start every path with r"..." and end every line with a comma.
MONTH_FOLDERS = [
    r"Y:\project-mars\atmos-data\201206",   # <- check this name (was "20120")
    r"Y:\project-mars\atmos-data\200809",
    r"Y:\project-mars\atmos-data\200810",
    r"Y:\project-mars\atmos-data\200912",
    r"Y:\project-mars\atmos-data\201001",
    r"Y:\project-mars\atmos-data\201112",
    r"Y:\project-mars\atmos-data\201201",
    r"Y:\project-mars\atmos-data\201202",
]
OUTPUT_DIR = r"Y:\project-mars\atmos-data\new"
# ====================================================================

import argparse
import csv
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

K_B = 1.380649e-23        # J/K
N_A = 6.02214076e23       # 1/mol
R_GEOID = 3_396_000.0     # m; reference geoid mean equatorial radius stated in the TPS label

# MRO header columns, in the order given by RSTP_HDR_TABLE in the MRO label (27 columns).
HEADER_FIELDS = [
    "n_samples", "start_time", "stop_time", "spacecraft_time", "orbit", "dsn_antenna",
    "config", "ray_path_dir_deg", "angle_from_diametric_deg", "lat_surface", "lon_surface",
    "subsolar_lat", "subsolar_lon", "Solar_Long", "Mars_Year", "sc_limb_dist_m",
    "sc_earth_dist_m", "LTST_hr", "SZA_deg", "sun_earth_sc_deg", "antenna_elev_deg",
    "gravity_model", "geopotential_ref", "trajectory_file", "rsr_file", "dlf_file", "tnf_file",
]
# MGS header columns, in the order given by RSTP_HDR_TABLE in the MGS label (29 columns).
# Names shared with MRO use the MRO names (e.g. OCCULTATION TIME -> spacecraft_time).
MGS_HEADER_FIELDS = [
    "start_time", "stop_time", "spacecraft_time", "orbit", "dsn_antenna", "ray_path_dir_deg",
    "angle_from_diametric_deg", "lat_surface", "sigma_lat_surface", "lon_surface", "sigma_lon_surface",
    "subsolar_lat", "subsolar_lon", "Solar_Long", "radius_surface_m", "sigma_radius_surface_m",
    "surface_pressure_pa", "sigma_surface_pressure_pa", "sc_limb_dist_m", "sc_earth_dist_m", "LTST_hr",
    "SZA_deg", "sun_earth_sc_deg", "antenna_elev_deg", "gravity_model", "geopotential_ref",
    "pck_file", "trajectory_file", "attitude_file",
]
MGS_MISSING = {"sigma_lat_surface": -9.999, "sigma_lon_surface": -9.999,
               "sigma_radius_surface_m": -9999.0, "sigma_surface_pressure_pa": -9.99}

GEOPOT_REF_COMMON = 12652807.0   # m^2/s^2; MRO value, so MRO geopotentials are unchanged

HEADER_NUMERIC = {"sigma_lat_surface", "sigma_lon_surface", "radius_surface_m", "sigma_radius_surface_m",
    "surface_pressure_pa", "sigma_surface_pressure_pa",
    "n_samples", "orbit", "dsn_antenna", "ray_path_dir_deg", "angle_from_diametric_deg",
    "lat_surface", "lon_surface", "subsolar_lat", "subsolar_lon", "Solar_Long", "Mars_Year",
    "sc_limb_dist_m", "sc_earth_dist_m", "LTST_hr", "SZA_deg", "sun_earth_sc_deg",
    "antenna_elev_deg", "geopotential_ref",
}

# Data columns, in the order given by RSTP_TABLE (10 columns).
LEVEL_FIELDS = [
    "Radius", "Latitude", "Longitude", "Geopotential",
    "Pressure", "Sigma_P", "Temperature", "Sigma_T", "Number_Density", "Sigma_n",
]


MY1_START = pd.Timestamp("1955-04-11")   # Clancy et al. (2000) convention, as in the MRO label
MARS_YEAR_DAYS = 686.9726

def mars_year(time_str, ls_deg):
    """Mars year from UTC time and solar longitude (the MGS header has no Mars-year field)."""
    try:
        t = pd.Timestamp(str(time_str).replace("Z", ""))
    except Exception:
        return np.nan
    x = (t - MY1_START).total_seconds() / 86400.0 / MARS_YEAR_DAYS
    my, frac = int(np.floor(x)) + 1, x - np.floor(x)
    if np.isfinite(ls_deg):             # fix the estimate near the year boundary using Ls
        d = frac - ls_deg / 360.0
        if d > 0.5:
            my += 1
        elif d < -0.5:
            my -= 1
    return float(my)


def parse_tps(path):
    """Parse one TPS file -> (header dict, levels DataFrame)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < 2:
        raise ValueError("file has no data rows")

    hdr_vals = [v.strip() for v in next(csv.reader([lines[0].strip()], skipinitialspace=True))]
    while hdr_vals and hdr_vals[-1] == "":
        hdr_vals.pop()
    first_is_count = bool(re.fullmatch(r"\d+", hdr_vals[0])) if hdr_vals else False
    if first_is_count and len(hdr_vals) == len(HEADER_FIELDS):
        mission, fields = "MRO", HEADER_FIELDS
    elif not first_is_count and len(hdr_vals) in (len(MGS_HEADER_FIELDS), len(MGS_HEADER_FIELDS) - 1):
        mission, fields = "MGS", MGS_HEADER_FIELDS     # trailing blank attitude-file field may be dropped
        hdr_vals += [""] * (len(fields) - len(hdr_vals))
    else:
        raise ValueError(f"unrecognised header: {len(hdr_vals)} fields (MRO has {len(HEADER_FIELDS)}, "
                         f"MGS has {len(MGS_HEADER_FIELDS)})")
    header = {"mission": mission}
    for k, v in zip(fields, hdr_vals):
        if k in HEADER_NUMERIC:
            try:
                header[k] = float(v)
            except ValueError:
                header[k] = np.nan
            if k in MGS_MISSING and np.isclose(header[k], MGS_MISSING[k]):
                header[k] = np.nan
        else:
            header[k] = v
    if mission == "MGS":
        header["n_samples"] = np.nan
        header["Mars_Year"] = mars_year(header["spacecraft_time"], header["Solar_Long"])

    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(LEVEL_FIELDS):
            raise ValueError(f"data row has {len(parts)} fields: {ln[:60]}...")
        rows.append([float(p) for p in parts])
    levels = pd.DataFrame(rows, columns=LEVEL_FIELDS)

    # Put geopotential on a common reference so profiles from both missions are comparable.
    if np.isfinite(header.get("geopotential_ref", np.nan)):
        levels["Geopotential"] = levels["Geopotential"] + header["geopotential_ref"] - GEOPOT_REF_COMMON
    n_expected = int(header["n_samples"]) if np.isfinite(header["n_samples"]) else len(levels)
    if n_expected != len(levels):
        warnings.warn(f"{Path(path).name}: header says {n_expected} samples, found {len(levels)}")
    return header, levels


def audit_profile(lv):
    """Physics/QC diagnostics for one profile (levels sorted by Radius)."""
    T, P, n, Phi = lv["Temperature"].values, lv["Pressure"].values, lv["Number_Density"].values, lv["Geopotential"].values
    ideal = P / (n * K_B * T)
    out = {
        "n_levels": len(lv),
        "z_span_km": (lv["Radius"].iloc[-1] - lv["Radius"].iloc[0]) / 1e3,
        "dz_median_m": float(np.median(np.diff(lv["Radius"]))) if len(lv) > 1 else np.nan,
        "ideal_ratio_maxdev": float(np.max(np.abs(ideal - 1))),
        "monotonic_P": bool(np.all(np.diff(P) < 0)),
        "monotonic_R": bool(np.all(np.diff(lv["Radius"]) > 0)),
        "any_nonpositive": bool((lv[["Pressure", "Temperature", "Number_Density"]] <= 0).any().any()),
        "sigmaT_bottom": float(lv["Sigma_T"].iloc[0]),
        "sigmaT_top": float(lv["Sigma_T"].iloc[-1]),
        "T_min": float(T.min()), "T_max": float(T.max()),
        "P_bottom": float(P[0]),
    }
    # Hydrostatic consistency in geopotential form: d lnP / dPhi = -m / (k_B T)
    if len(lv) > 2:
        dlnP = -np.diff(np.log(P))
        dPhi = np.diff(Phi)
        Tm = 0.5 * (T[1:] + T[:-1])
        ok = dPhi > 0
        m = K_B * Tm[ok] * dlnP[ok] / dPhi[ok]
        out["m_implied_gmol"] = float(np.median(m) * N_A * 1e3) if ok.any() else np.nan
        g_eff = dPhi / np.diff(lv["Radius"].values)
        out["g_eff_bottom"] = float(g_eff[0])
    else:
        out["m_implied_gmol"] = np.nan
        out["g_eff_bottom"] = np.nan
    return out


def find_tps_files(folder):
    """All .TPS/.tps files inside one folder (searched recursively)."""
    return sorted(p.resolve() for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() == ".tps")


def load_tps_folders(folders):
    """Load every TPS file from a list of month folders. Returns (levels_df, profiles_df)."""
    level_frames, profile_rows, failures = [], [], []
    seen = {}

    print("Scanning folders:")
    for folder in folders:
        folder_str = str(folder)
        if any(ord(ch) < 32 or 127 <= ord(ch) < 160 for ch in folder_str):
            print(f"  [BAD PATH] {folder_str!r}  -> contains hidden characters; "
                  f'put r in front of the quotes, like r"Y:\\...", skipped')
            continue
        if folder_str.count(":\\") > 1 or folder_str.count(":/") > 1:
            print(f"  [BAD PATH] {folder_str}  -> looks like two paths joined; "
                  f"check for a missing comma in MONTH_FOLDERS, skipped")
            continue
        folder = Path(folder)
        if not folder.is_dir():
            print(f"  [MISSING] {folder}  -> folder not found, skipped")
            continue
        files = find_tps_files(folder)
        n_loaded = 0
        for f in files:
            pid = f.stem.upper()
            if pid in seen:
                warnings.warn(f"duplicate Profile_ID {pid}: keeping {seen[pid]}, skipping {f}")
                continue
            try:
                header, lv = parse_tps(f)
            except Exception as e:  # keep going; report at the end
                failures.append((str(f), str(e)))
                continue
            seen[pid] = f
            n_loaded += 1

            lv = lv.sort_values("Radius").reset_index(drop=True)
            lv.insert(0, "Profile_ID", pid)
            lv["Level"] = np.arange(len(lv))
            lv["Height_above_base_m"] = lv["Radius"] - lv["Radius"].iloc[0]
            lv["Height_above_geoid_m"] = lv["Radius"] - R_GEOID
            level_frames.append(lv)

            row = {"Profile_ID": pid, "source_folder": str(folder), "source_file": str(f), **header}
            # Ingress/egress: from the ID letter after "MAO" when present, else from the geometry.
            m = re.match(r"^MAO([IE])", pid)
            if m:
                row["occultation"] = "ingress" if m.group(1) == "I" else "egress"
            else:
                row["occultation"] = "ingress" if abs(header["angle_from_diametric_deg"]) > 90 else "egress"
            row.update(audit_profile(lv))
            profile_rows.append(row)

        print(f"  [OK] {folder}  -> {len(files)} TPS file(s) found, {n_loaded} loaded")

    if failures:
        print(f"WARNING: {len(failures)} file(s) failed to parse:")
        for f, e in failures[:20]:
            print(f"  {f}: {e}")
    if not level_frames:
        raise FileNotFoundError("no TPS files were loaded; check the paths in MONTH_FOLDERS")

    levels = pd.concat(level_frames, ignore_index=True)
    profiles = pd.DataFrame(profile_rows)
    return levels, profiles


# Profile metadata copied onto every level row of the combined CSV.
META_COLS = ["mission", "source_folder", "Solar_Long", "Mars_Year", "LTST_hr", "SZA_deg",
             "lat_surface", "lon_surface", "occultation", "config", "orbit", "start_time"]


def combine(levels, profiles):
    """One table: every level row plus its profile's metadata."""
    cols = [c for c in META_COLS if c in profiles.columns]
    combined = levels.merge(profiles[["Profile_ID"] + cols], on="Profile_ID", how="left")
    lead = ["Profile_ID", "mission", "source_folder", "Mars_Year", "Solar_Long", "LTST_hr", "occultation"]
    return combined[lead + [c for c in combined.columns if c not in lead]]


def summarize(levels, profiles):
    print(f"Profiles: {len(profiles)}   Levels: {len(levels)}")
    print("Per mission:")
    print(profiles.groupby("mission").agg(profiles=("Profile_ID", "size"),
                                          MY_min=("Mars_Year", "min"), MY_max=("Mars_Year", "max"),
                                          Ls_min=("Solar_Long", "min"), Ls_max=("Solar_Long", "max"),
                                          LTST_min=("LTST_hr", "min"), LTST_max=("LTST_hr", "max")).round(2).to_string())
    print(f"Mars years: {sorted(profiles['Mars_Year'].dropna().unique().astype(int).tolist())}")
    print(f"Ls range: {profiles['Solar_Long'].min():.1f} - {profiles['Solar_Long'].max():.1f} deg")
    print(f"Latitude range (surface): {profiles['lat_surface'].min():.1f} - {profiles['lat_surface'].max():.1f} deg")
    print(f"LTST range: {profiles['LTST_hr'].min():.2f} - {profiles['LTST_hr'].max():.2f} h")
    print(f"Levels per profile: median {profiles['n_levels'].median():.0f} "
          f"(min {profiles['n_levels'].min()}, max {profiles['n_levels'].max()})")
    print(f"Vertical span: median {profiles['z_span_km'].median():.1f} km")
    print(f"Ideal-gas max deviation (worst profile): {profiles['ideal_ratio_maxdev'].max():.2e}")
    print(f"Implied molecular mass: median {profiles['m_implied_gmol'].median():.3f} g/mol "
          f"(IQR {profiles['m_implied_gmol'].quantile(.25):.3f}-{profiles['m_implied_gmol'].quantile(.75):.3f})")
    print(f"Effective gravity at base: median {profiles['g_eff_bottom'].median():.4f} m/s^2")
    print(f"Sigma_T: bottom median {profiles['sigmaT_bottom'].median():.2f} K, "
          f"top median {profiles['sigmaT_top'].median():.2f} K")
    bad = profiles[~profiles["monotonic_P"] | ~profiles["monotonic_R"] | profiles["any_nonpositive"]]
    print(f"Profiles failing basic QC (non-monotonic or non-positive): {len(bad)}")
    print(profiles["occultation"].value_counts().to_string())


if __name__ == "__main__":
    import sys
    # Inside Jupyter, sys.argv holds the kernel's own arguments (e.g. "-f kernel-xxx.json"),
    # not user folders, so command-line parsing is skipped there and MONTH_FOLDERS is used.
    in_notebook = "ipykernel" in sys.modules
    if in_notebook:
        folders, out = MONTH_FOLDERS, Path(OUTPUT_DIR)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("folders", nargs="*", help="month folders (optional; overrides MONTH_FOLDERS)")
        ap.add_argument("--out", default=None, help="output directory (optional; overrides OUTPUT_DIR)")
        args = ap.parse_args()
        folders = args.folders or MONTH_FOLDERS
        out = Path(args.out or OUTPUT_DIR)

    levels, profiles = load_tps_folders(folders)
    print()
    summarize(levels, profiles)

    combined = combine(levels, profiles)
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "mro_tps_combined.csv", index=False)
    profiles.to_csv(out / "mro_tps_profiles.csv", index=False)
    print(f"\nFINAL FILE: {out / 'mro_tps_combined.csv'}  ({len(combined)} rows, {combined['Profile_ID'].nunique()} profiles)")
    print(f"Profile summary: {out / 'mro_tps_profiles.csv'}")
