"""Data loading, quality control, physics audit, profile-level splits and gap scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import K_B, N_A, Physics, hydro_residual_np, implied_mass
from .synthetic import generate_dataset

REQUIRED = ["Profile_ID", "Mars_Year", "Solar_Long", "Radius", "Latitude", "Longitude", "Geopotential",
            "Pressure", "Sigma_P", "Temperature", "Sigma_T", "Number_Density", "Sigma_n"]
IN_COLS = ["Z_km", "Latitude", "sin_lon", "cos_lon", "sin_ls", "cos_ls"]
OUT_COLS = ["Temperature", "lnP"]


# --------------------------------------------------------------------------- loading
def load_raw(cfg: dict) -> pd.DataFrame:
    if cfg["DATA_SOURCE"] == "synthetic":
        return generate_dataset(cfg["SYNTH_PROFILES_PER_YEAR"], cfg["SYNTH_SEED"], cfg["SYNTH_NOISE_SCALE"])
    if cfg["DATA_SOURCE"] == "csv":
        return pd.read_csv(cfg["DATA_LEVELS"])
    raise ValueError(f"DATA_SOURCE must be 'synthetic' or 'csv', got {cfg['DATA_SOURCE']!r}")


def preprocess(lv: pd.DataFrame, phys: Physics) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in lv.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    lv = lv.sort_values(["Profile_ID", "Radius"]).reset_index(drop=True)
    lv["Z_km"] = lv["Geopotential"] / phys.g0 / 1e3
    lv["lnP"] = np.log(lv["Pressure"])
    lv["sig_lnP"] = lv["Sigma_P"] / lv["Pressure"]
    lon, ls = np.radians(lv["Longitude"]), np.radians(lv["Solar_Long"])
    lv["sin_lon"], lv["cos_lon"] = np.sin(lon), np.cos(lon)
    lv["sin_ls"], lv["cos_ls"] = np.sin(ls), np.cos(ls)
    return lv


def quality_control(lv: pd.DataFrame, cfg: dict):
    """Trim the top initialisation zone, require monotonic Z and a minimum length."""
    n_prof_raw, n_lev_raw = lv["Profile_ID"].nunique(), len(lv)
    z_top_before = lv.groupby("Profile_ID")["Z_km"].max()
    lv = lv[lv["Sigma_T"] <= cfg["SIGMA_T_MAX"]].copy()
    z_top_after = lv.groupby("Profile_ID")["Z_km"].max()
    trimmed_km = (z_top_before - z_top_after.reindex(z_top_before.index)).dropna()
    mono = pd.Series({pid: bool(np.all(np.diff(z.values) > 0)) for pid, z in lv.groupby("Profile_ID")["Z_km"]})
    counts = lv.groupby("Profile_ID").size()
    keep = mono[mono].index.intersection(counts[counts >= cfg["MIN_LEVELS"]].index)
    lv = lv[lv["Profile_ID"].isin(keep)].reset_index(drop=True)
    meta = lv.groupby("Profile_ID").agg(Mars_Year=("Mars_Year", "first"), Solar_Long=("Solar_Long", "first"),
                                        lat=("Latitude", "mean"), lon=("Longitude", "mean"),
                                        n_levels=("Z_km", "size"), z_min=("Z_km", "min"), z_max=("Z_km", "max"))
    meta["span_km"] = meta["z_max"] - meta["z_min"]
    qc = dict(profiles_raw=int(n_prof_raw), levels_raw=int(n_lev_raw), profiles_kept=len(meta),
              levels_kept=len(lv), median_top_trim_km=float(trimmed_km.median()),
              median_span_km=float(meta["span_km"].median()), median_levels=float(meta["n_levels"].median()))
    return lv, meta, qc


def physics_audit(lv: pd.DataFrame, phys: Physics) -> dict:
    """Recover the retrieval's molecular mass and check both physical laws on the reference."""
    m_prof = pd.Series({pid: implied_mass(g["Temperature"], g["lnP"], g["Geopotential"])
                        for pid, g in lv.groupby("Profile_ID")})
    phys.m = float(np.median(m_prof))
    ideal_dev = np.abs(lv["Pressure"] / (lv["Number_Density"] * K_B * lv["Temperature"]) - 1)
    ref_res = pd.Series({pid: np.sqrt(np.mean(hydro_residual_np(g["Z_km"], g["Temperature"], g["lnP"], phys) ** 2))
                         for pid, g in lv.groupby("Profile_ID")})
    phys.floor = float(np.median(ref_res))
    return dict(mean_molecular_mass_gmol=phys.m_gmol,
                molecular_mass_IQR_gmol=[float(q * N_A * 1e3) for q in m_prof.quantile([0.25, 0.75])],
                H_ref_km=phys.H_ref_km, ideal_gas_max_rel_dev=float(ideal_dev.max()),
                reference_hydro_residual_RMS_median=phys.floor)


# --------------------------------------------------------------------------- splits
def stratified_profile_split(meta: pd.DataFrame, fracs, seed):
    """Whole profiles go to train / val / test, stratified by Mars year (no leakage)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for _, g in meta.groupby("Mars_Year"):
        ids = list(rng.permutation(g.index.values))
        n = len(ids)
        if n < 3:
            tr += ids
            continue
        n_te = max(1, int(round(fracs[2] * n)))
        n_va = max(1, int(round(fracs[1] * n)))
        te += ids[:n_te]
        va += ids[n_te:n_te + n_va]
        tr += ids[n_te + n_va:]
    return sorted(tr), sorted(va), sorted(te)


# --------------------------------------------------------------------------- gaps
def make_gap(z, size_km, location, min_obs_km=5.0):
    """Boolean mask of hidden levels, or None if the profile is too short for this gap."""
    z = np.asarray(z)
    z0, z1 = z.min(), z.max()
    if (z1 - z0) - size_km < min_obs_km:
        return None
    if location == "lower":
        lo, hi = z0 - 1e-9, z0 + size_km
    elif location == "upper":
        lo, hi = z1 - size_km, z1 + 1e-9
    elif location == "middle":
        c = 0.5 * (z0 + z1)
        lo, hi = c - size_km / 2, c + size_km / 2
    else:
        raise ValueError(location)
    gap = (z >= lo) & (z <= hi)
    if gap.sum() < 3 or (~gap).sum() < 5:
        return None
    return gap


def build_cases(df: pd.DataFrame, ids, configs, min_obs_km):
    """All (profile, size, location) combinations for which a valid gap exists."""
    cases = []
    for pid in ids:
        z = df.loc[df["Profile_ID"] == pid, "Z_km"].values
        for s, loc in configs:
            if make_gap(z, s, loc, min_obs_km) is not None:
                cases.append((pid, s, loc))
    return cases
