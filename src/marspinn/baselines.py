"""Classical and statistical reconstruction baselines."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .data import IN_COLS, OUT_COLS
from .physics import Physics

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def interp_extrap(z_obs, y_obs, z_new, kind="linear", window_km=3.0):
    """Linear / PCHIP interpolation inside the observed range; linear fit to the nearest
    ``window_km`` of data outside it (lower and upper gaps)."""
    o = np.argsort(z_obs)
    z_obs, y_obs = np.asarray(z_obs)[o], np.asarray(y_obs)[o]
    z_new = np.asarray(z_new)
    out = np.empty(len(z_new))
    inside = (z_new >= z_obs[0]) & (z_new <= z_obs[-1])
    if kind == "pchip":
        out[inside] = PchipInterpolator(z_obs, y_obs)(z_new[inside])
    else:
        out[inside] = np.interp(z_new[inside], z_obs, y_obs)
    for side in ("below", "above"):
        sel = z_new < z_obs[0] if side == "below" else z_new > z_obs[-1]
        if sel.any():
            near = z_obs <= z_obs[0] + window_km if side == "below" else z_obs >= z_obs[-1] - window_km
            if near.sum() < 2:
                near = np.zeros_like(near)
                if side == "below":
                    near[:2] = True
                else:
                    near[-2:] = True
            c = np.polyfit(z_obs[near], y_obs[near], 1)
            out[sel] = np.polyval(c, z_new[sel])
    return out


def hydrostatic_integration(z_obs, T_obs, lnP_obs, z_gap, phys: Physics):
    """PCHIP temperature, then integrate d lnP/dZ = -1/H(T) from the gap edge(s); blended
    linearly in height when data exist on both sides."""
    z_obs, T_obs, lnP_obs, z_gap = map(np.asarray, (z_obs, T_obs, lnP_obs, z_gap))
    T_gap = interp_extrap(z_obs, T_obs, z_gap, "pchip")
    z_all = np.concatenate([z_obs, z_gap])
    T_all = np.concatenate([T_obs, T_gap])
    o = np.argsort(z_all)
    z_all, T_all = z_all[o], T_all[o]
    zf = np.linspace(z_all[0], z_all[-1], max(2000, 20 * len(z_all)))
    Tf = np.interp(zf, z_all, T_all)
    integrand = -phys.T_ref / (phys.H_ref_km * Tf)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zf))])
    Ig = np.interp(z_gap, zf, I)
    below, above = z_obs < z_gap.min(), z_obs > z_gap.max()
    preds = []
    zb = za = None
    if below.any():
        zb = z_obs[below].max()
        lb = lnP_obs[below][np.argmax(z_obs[below])]
        preds.append(lb + Ig - np.interp(zb, zf, I))
    if above.any():
        za = z_obs[above].min()
        la = lnP_obs[above][np.argmin(z_obs[above])]
        preds.append(la + Ig - np.interp(za, zf, I))
    if len(preds) == 2:
        w = (z_gap - zb) / (za - zb)
        lnP_gap = (1 - w) * preds[0] + w * preds[1]
    else:
        lnP_gap = preds[0]
    return T_gap, lnP_gap


def gp_predict(z_obs, y_obs, z_new):
    """GP with RBF + white-noise kernel around a linear mean; returns mean and std."""
    z_obs, y_obs, z_new = map(np.asarray, (z_obs, y_obs, z_new))
    c = np.polyfit(z_obs, y_obs, 1)
    kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(5.0, (0.3, 50.0)) + WhiteKernel(1e-2, (1e-8, 1.0))
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(z_obs[:, None], y_obs - np.polyval(c, z_obs))
    mu, sd = gp.predict(z_new[:, None], return_std=True)
    return mu + np.polyval(c, z_new), sd


def fit_rf(df, seed=0):
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    rf.fit(df[IN_COLS].values, df[OUT_COLS].values)
    return rf


def fit_eof(df, var, sig_col, cfg):
    """DINEOF-style EOFs on a common geopotential-height grid."""
    dz = cfg["EOF_DZ_KM"]
    grid = np.arange(np.floor(df["Z_km"].min()), np.ceil(df["Z_km"].max()) + dz / 2, dz)
    groups = list(df.groupby("Profile_ID"))
    X = np.full((len(groups), len(grid)), np.nan)
    for i, (_, g) in enumerate(groups):
        z, y = g["Z_km"].values, g[var].values
        inside = (grid >= z.min()) & (grid <= z.max())
        X[i, inside] = np.interp(grid[inside], z, y)
    keep = np.isfinite(X).mean(0) >= cfg["EOF_MIN_COVER"]
    grid, X = grid[keep], X[:, keep]
    mask = np.isfinite(X)
    Xf = np.where(mask, X, np.nanmean(X, 0))
    k = None
    for _ in range(200):
        mu = Xf.mean(0)
        U, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
        if k is None:
            frac = np.cumsum(S ** 2) / np.sum(S ** 2)
            k = int(min(cfg["EOF_MAX_K"], np.searchsorted(frac, cfg["EOF_VAR_FRAC"]) + 1, len(S)))
        R = (U[:, :k] * S[:k]) @ Vt[:k] + mu
        delta = np.max(np.abs(Xf[~mask] - R[~mask])) if (~mask).any() else 0.0
        Xf[~mask] = R[~mask]
        if delta < 1e-6 * (np.abs(mu).mean() + 1e-12):
            break
    mu = Xf.mean(0)
    U, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    R = (U[:, :k] * S[:k]) @ Vt[:k] + mu
    lam = S[:k] ** 2 / max(len(X) - 1, 1)
    sig2 = float(np.mean((X[mask] - R[mask]) ** 2))
    return dict(grid=grid, mu=mu, V=Vt[:k].T, lam=lam, sig2=sig2, k=k, var=var, sig_col=sig_col)


def eof_predict(E, z_obs, y_obs, sig_obs, z_new):
    """Bayesian conditioning of the EOF amplitudes on the observed levels."""
    g = E["grid"]
    zo, zn = np.clip(z_obs, g[0], g[-1]), np.clip(z_new, g[0], g[-1])
    Uo = np.column_stack([np.interp(zo, g, E["V"][:, j]) for j in range(E["k"])])
    Un = np.column_stack([np.interp(zn, g, E["V"][:, j]) for j in range(E["k"])])
    mo, mn = np.interp(zo, g, E["mu"]), np.interp(zn, g, E["mu"])
    rinv = 1.0 / (E["sig2"] + np.asarray(sig_obs) ** 2)
    Sa = np.linalg.inv(np.diag(1.0 / E["lam"]) + Uo.T @ (rinv[:, None] * Uo))
    ma = Sa @ (Uo.T @ (rinv * (np.asarray(y_obs) - mo)))
    var = np.einsum("ij,jk,ik->i", Un, Sa, Un) + E["sig2"]
    return mn + Un @ ma, np.sqrt(var)


def fit_eof_pair(df, cfg):
    return dict(T=fit_eof(df, "Temperature", "Sigma_T", cfg), P=fit_eof(df, "lnP", "sig_lnP", cfg))
