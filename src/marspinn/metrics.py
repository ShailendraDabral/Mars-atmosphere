"""Metrics, profile-level aggregation, paired significance tests and conformal calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .benchmark import OURS
from .physics import hydro_residual_np

COV_LEVELS = [0.5, 0.8, 0.9, 0.95]
POINT_METRICS = ["T_MAE", "T_RMSE", "T_bias", "T_within_2sig", "P_relMAE", "n_relMAE", "Hydro_RMS"]


def case_metrics(g, phys):
    g = g.sort_values("Z_km")
    eT = g["T_pred"].values - g["T_ref"].values
    elnP = g["lnP_pred"].values - g["lnP_ref"].values
    n_ratio = np.exp(elnP) * g["T_ref"].values / g["T_pred"].values
    d = dict(T_MAE=np.mean(np.abs(eT)), T_RMSE=np.sqrt(np.mean(eT ** 2)), T_bias=np.mean(eT),
             T_within_2sig=np.mean(np.abs(eT) <= 2 * g["sig_T"].values),
             P_relMAE=100 * np.mean(np.abs(np.exp(elnP) - 1)), n_relMAE=100 * np.mean(np.abs(n_ratio - 1)),
             Hydro_RMS=np.sqrt(np.mean(hydro_residual_np(g["Z_km"].values, g["T_pred"].values,
                                                         g["lnP_pred"].values, phys) ** 2)) if len(g) >= 3 else np.nan)
    if g["T_std"].notna().all():
        sT = np.sqrt(g["T_std"].values ** 2 + g["sig_T"].values ** 2)
        sP = np.sqrt(g["lnP_std"].values ** 2 + g["sig_lnP"].values ** 2)
        for q in COV_LEVELS:
            zq = stats.norm.ppf(0.5 + q / 2)
            d[f"T_cov{int(q * 100)}"] = np.mean(np.abs(eT) <= zq * sT)
            d[f"P_cov{int(q * 100)}"] = np.mean(np.abs(elnP) <= zq * sP)
        d["T_std_mean"] = g["T_std"].mean()
    return d


def compute_metrics(pw, phys, keys=("case_id", "Profile_ID", "gap_km", "location", "method")):
    keys = [k for k in keys if k in pw.columns]
    rows = []
    for k, g in pw.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **case_metrics(g, phys)})
    return pd.DataFrame(rows)


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def summarize(M, metrics, by=("method",), n_boot=5000):
    """Profile-level means, then mean and bootstrap CI over profiles."""
    by = list(by)
    metrics = [m for m in metrics if m in M.columns]
    per_prof = M.groupby(by + ["Profile_ID"])[metrics].mean().reset_index()
    rows = []
    for key, g in per_prof.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_profiles"] = g["Profile_ID"].nunique()
        for m in metrics:
            lo, hi = boot_ci(g[m].values, n_boot)
            r[m], r[m + "_lo"], r[m + "_hi"] = g[m].mean(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def paired_tests(M, metric, ours=OURS, by=None, n_boot=5000, alpha=0.05):
    """Profile-level paired Wilcoxon tests with Holm correction. Positive mean_diff = ours better."""
    out = []
    groups = [(None, M)] if by is None else list(M.groupby(by))
    for gkey, G in groups:
        pp = G.groupby(["method", "Profile_ID"])[metric].mean().unstack(0)
        if ours not in pp:
            continue
        for other in [c for c in pp.columns if c != ours]:
            d = (pp[other] - pp[ours]).dropna()
            if len(d) < 3:
                continue
            if np.allclose(d.values, 0):
                p = 1.0
            else:
                try:
                    p = stats.wilcoxon(d.values).pvalue
                except ValueError:
                    p = 1.0
            lo, hi = boot_ci(d.values, n_boot)
            out.append(dict(group=gkey, metric=metric, competitor=other, n_profiles=len(d),
                            mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                            rel_improvement_pct=100 * d.mean() / pp[other].loc[d.index].mean(),
                            ours_better_frac=(d > 0).mean(), p_wilcoxon=p))
    T = pd.DataFrame(out)
    if len(T):
        T["p_holm"] = np.nan
        for _, idx in T.groupby(["group", "metric"], dropna=False).groups.items():
            p = T.loc[idx, "p_wilcoxon"].values
            o = np.argsort(p)
            m = len(p)
            adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p[o]))
            T.loc[np.array(idx)[o], "p_holm"] = adj
        T["significant"] = T["p_holm"] < alpha
    return T


# --------------------------------------------------------------------------- conformal
QGRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def _scores(g):
    sT = np.sqrt(g.T_std ** 2 + g.sig_T ** 2)
    sP = np.sqrt(g.lnP_std ** 2 + g.sig_lnP ** 2)
    return (np.abs(g.T_pred - g.T_ref) / sT).values, (np.abs(g.lnP_pred - g.lnP_ref) / sP).values


def fit_conformal(pw_cal):
    """Mondrian (per gap location) split-conformal quantiles of normalised scores."""
    Q = {}
    for (m, loc), g in pw_cal[pw_cal.T_std.notna()].groupby(["method", "location"]):
        for var, r in zip(("T", "P"), _scores(g)):
            n = len(r)
            for q in QGRID:
                Q[(m, loc, var, float(q))] = np.quantile(r, min(1.0, np.ceil((n + 1) * q) / n))
    return Q


def conformal_coverage(pw, Q):
    rows = []
    methods = set(k[0] for k in Q)
    for (cid, pid, loc, m), g in pw[pw.method.isin(methods)].groupby(["case_id", "Profile_ID", "location", "method"]):
        if g.T_std.isna().any() or (m, loc, "T", 0.95) not in Q:
            continue
        rT, rP = _scores(g)
        d = dict(case_id=cid, Profile_ID=pid, location=loc, method=m)
        for q in COV_LEVELS:
            d[f"T_ccov{int(q * 100)}"] = np.mean(rT <= Q[(m, loc, "T", q)])
            d[f"P_ccov{int(q * 100)}"] = np.mean(rP <= Q[(m, loc, "P", q)])
        rows.append(d)
    return pd.DataFrame(rows)


def uncertainty_summary(M, PW, methods, n_boot=5000):
    uq_cols = [c for c in M.columns if "_cov" in c] + ["T_std_mean"]
    UQ = summarize(M[M["method"].isin(methods)], uq_cols, n_boot=n_boot)
    rho = []
    for m in methods:
        g = PW[(PW["method"] == m) & PW["T_std"].notna()]
        if len(g) > 10 and g["T_std"].nunique() > 1 and g["lnP_std"].nunique() > 1:
            rho.append(dict(method=m,
                            spearman_T=stats.spearmanr(g["T_std"], np.abs(g["T_pred"] - g["T_ref"])).correlation,
                            spearman_lnP=stats.spearmanr(g["lnP_std"], np.abs(g["lnP_pred"] - g["lnP_ref"])).correlation))
    return UQ.merge(pd.DataFrame(rho, columns=["method", "spearman_T", "spearman_lnP"]), on="method", how="left")
