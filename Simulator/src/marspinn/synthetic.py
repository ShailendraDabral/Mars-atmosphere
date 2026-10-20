"""Physically consistent synthetic Mars radio-occultation profiles.

The generator mimics the structure of the MRO RSS TPS archive (columns, sampling,
coverage per Mars year, retrieval uncertainties) so the whole workflow runs without
downloading data. Each profile is built as:

1. A "true" temperature profile with a pre-dawn near-surface inversion, a tropospheric
   lapse rate, a transition to a colder middle atmosphere, and vertically propagating
   waves. Its parameters depend smoothly on latitude, season (L_s) and Mars year.
2. A retrieval error that grows towards the top of the profile (the retrieval is
   initialised there), drawn as vertically correlated Gaussian noise.
3. Pressure obtained by integrating hydrostatic balance in geopotential height from the
   *retrieved* temperature, and number density from the ideal-gas law -- exactly the way
   the real retrieval makes T, P and n mutually consistent.

The synthetic data are for testing and teaching the workflow; they are not a substitute
for the real observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import K_B, Physics, integrate_lnP

R_MARS_M = 3389.5e3

# (Mars year, L_s range, latitude at start -> end) mirroring the real archive coverage
YEAR_TRACKS = {
    29: dict(ls=(106.8, 142.5), lat=(-12.5, 14.5)),
    30: dict(ls=(21.0, 45.5), lat=(69.3, 48.8)),
    31: dict(ls=(37.4, 75.1), lat=(65.8, 47.5)),
}


def _correlated_noise(rng, z, corr_km):
    """Unit-variance Gaussian noise with a squared-exponential vertical correlation."""
    dz = np.abs(z[:, None] - z[None, :])
    C = np.exp(-0.5 * (dz / corr_km) ** 2) + 1e-8 * np.eye(len(z))
    L = np.linalg.cholesky(C)
    return L @ rng.standard_normal(len(z))


def true_temperature(z, lat, ls, my, rng):
    """Smooth 'true' temperature profile (K) on geopotential heights z (km)."""
    phi, Ls = np.radians(lat), np.radians(ls)
    z0 = z.min()
    # surface-layer temperature: warmer at low latitudes and later in northern summer
    T_top_inv = 203 + 9 * np.cos(phi) + 4 * np.sin(Ls) * np.sign(lat + 1e-9) + rng.normal(0, 3)
    inv_depth = rng.uniform(1.0, 3.0)                     # km, pre-dawn inversion depth
    inv_strength = rng.uniform(6, 22) * (0.6 + 0.4 * abs(np.sin(phi)))
    gamma = rng.uniform(1.6, 2.4)                          # K/km lapse rate above inversion
    T_mid = 150 + 22 * np.cos(phi) ** 2 + (8 if my == 29 else 0) + rng.normal(0, 3)
    zi = z0 + inv_depth
    # troposphere: linear lapse, softly limited by a colder middle atmosphere
    T_lin = T_top_inv - gamma * (z - zi)
    k = 0.35
    T = T_mid + np.logaddexp(0, k * (T_lin - T_mid)) / k
    # inversion below zi: temperature *increases* with height from the surface
    below = z < zi
    frac = (zi - z[below]) / inv_depth
    T[below] = T[below] - inv_strength * frac ** 2
    # vertically propagating waves (amplitude grows with height)
    for _ in range(2):
        lam = rng.uniform(8, 20)
        amp = rng.uniform(0.5, 2.0)
        ph = rng.uniform(0, 2 * np.pi)
        T += amp * (1 + (z - z0) / 40) * np.sin(2 * np.pi * z / lam + ph)
    return T


def generate_profile(rng, pid, my, ls, lat, lon, phys: Physics, noise_scale=1.0, ltst=None):
    z_surf = rng.uniform(-4.8, 0.5)                       # km, local surface geopotential height
    span = rng.uniform(38, 46)                            # km, before trimming the top
    dz = rng.uniform(0.40, 0.75)                          # km, vertical sampling
    z = np.arange(z_surf, z_surf + span, dz)
    T_true = true_temperature(z, lat, ls, my, rng)
    ztop = z.max()
    sig_T = 0.15 + 9.8 * np.exp(-(ztop - z) / 7.0)        # grows to ~10 K at the top
    # the realised retrieval error is smaller than the stated 1-sigma bound and smooth in height
    T_ret = T_true + 0.5 * noise_scale * sig_T * _correlated_noise(rng, z, 2.5)
    T_ret = np.clip(T_ret, 110, 260)
    P_ref0 = 610.0 * (1 + 0.08 * np.sin(np.radians(ls - 250))) * np.exp(rng.normal(0, 0.03))
    lnP0 = np.log(P_ref0) - z_surf / 11.0                # surface pressure at z_surf
    lnP = integrate_lnP(z, T_ret, lnP0, phys)
    P = np.exp(lnP)
    n = P / (K_B * T_ret)
    sig_rel = 0.0015 + 0.03 * np.exp(-(ztop - z) / 7.0)
    lat_l = lat + np.linspace(0, rng.uniform(-0.6, 0.6), len(z))
    lon_l = (lon + np.linspace(0, rng.uniform(-0.6, 0.6), len(z))) % 360
    radius = R_MARS_M + z * 1e3 + rng.uniform(-3e3, 3e3)  # planetocentric radius, m (monotonic)
    return pd.DataFrame(dict(
        Profile_ID=pid, Mars_Year=float(my), Solar_Long=ls, Radius=radius, Latitude=lat_l,
        Longitude=lon_l, Geopotential=z * 1e3 * phys.g0, Pressure=P, Sigma_P=sig_rel * P,
        Temperature=T_ret, Sigma_T=sig_T, Number_Density=n, Sigma_n=sig_rel * n,
        LTST_hr=ltst if ltst is not None else rng.uniform(3, 5), T_true=T_true))


def generate_dataset(profiles_per_year=(65, 46, 73), seed=7, noise_scale=1.0,
                     phys: Physics | None = None) -> pd.DataFrame:
    """Synthetic archive with the same columns as ``mro_tps_combined.csv``."""
    phys = phys or Physics()
    rng = np.random.default_rng(seed)
    parts = []
    for (my, track), n_prof in zip(YEAR_TRACKS.items(), profiles_per_year):
        for i in range(n_prof):
            f = i / max(n_prof - 1, 1)
            ls = track["ls"][0] + f * (track["ls"][1] - track["ls"][0])
            lat = track["lat"][0] + f * (track["lat"][1] - track["lat"][0]) + rng.normal(0, 0.3)
            lon = rng.uniform(0, 360)
            pid = f"SYN{my}_{i:03d}"
            parts.append(generate_profile(rng, pid, my, ls, lat, lon, phys, noise_scale))
    return pd.concat(parts, ignore_index=True)
