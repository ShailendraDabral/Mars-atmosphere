"""Figures (PNG + PDF), mirroring those of the paper."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .benchmark import NAMES, OURS  # noqa: E402

ORDER = list(NAMES.values())
COLORS = dict(zip(ORDER, ["#9e9e9e", "#616161", "#8d6e63", "#7cb342", "#9c27b0", "#26a69a",
                          "#f4a582", "#92c5de", "#e7a3a0", "#9ec3e6", "#d6604d", "#2166ac"]))

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def savefig(fig, folder, name):
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_data_overview(lv, meta, folder):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    years = sorted(meta["Mars_Year"].unique())
    cmap = plt.get_cmap("viridis", len(years))
    for i, my in enumerate(years):
        s = meta[meta["Mars_Year"] == my]
        ax[0].scatter(s["Solar_Long"], s["lat"], s=18, color=cmap(i), label=f"MY {int(my)}")
    ax[0].set_xlabel("Solar longitude $L_s$ (deg)"); ax[0].set_ylabel("Latitude (deg)")
    ax[0].set_title("(a) Coverage"); ax[0].legend(fontsize=8)
    for pid, g in lv.groupby("Profile_ID"):
        i = years.index(meta.loc[pid, "Mars_Year"])
        ax[1].plot(g["Temperature"], g["Z_km"], lw=0.5, alpha=0.5, color=cmap(i))
    ax[1].set_xlabel("Temperature (K)"); ax[1].set_ylabel("Geopotential height Z (km)"); ax[1].set_title("(b) All profiles")
    h_top = lv.groupby("Profile_ID")["Z_km"].transform("max") - lv["Z_km"]
    ax[2].scatter(lv["Sigma_T"], h_top, s=2, alpha=0.3)
    ax[2].set_xlabel(r"Retrieval $\sigma_T$ (K)"); ax[2].set_ylabel("Distance below (trimmed) top (km)")
    ax[2].invert_yaxis(); ax[2].set_title("(c) Retrieval uncertainty")
    fig.tight_layout(); savefig(fig, folder, "fig01_data_overview")


def fig_examples(PW, test_df, test_ids, folder):
    ex_methods = [NAMES[k] for k in ("lin", "hydro", "eof", "mlp_rft", "pinn_rft")]
    have = set(PW["case_id"])
    pid = next((p for p in test_ids if f"{p}|10|middle" in have and f"{p}|10|upper" in have), None)
    if pid is None:
        return
    cases = [(pid, 10, "middle"), (pid, 10, "upper")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2), squeeze=False)
    for r, case in enumerate(cases):
        prof = test_df[test_df.Profile_ID == case[0]].sort_values("Z_km")
        g = PW[PW.case_id == "|".join(map(str, case))]
        zg = g["Z_km"].unique()
        ax = axes[r]
        ax[0].plot(prof["Temperature"], prof["Z_km"], "k-", lw=1.8, label="reference")
        ax[0].fill_betweenx(prof["Z_km"], prof["Temperature"] - 2 * prof["Sigma_T"],
                            prof["Temperature"] + 2 * prof["Sigma_T"], color="k", alpha=0.08)
        for a in ax:
            a.axhspan(zg.min(), zg.max(), color="gold", alpha=0.15)
        for m in ex_methods:
            gm = g[g.method == m].sort_values("Z_km")
            if not len(gm):
                continue
            ax[0].plot(gm["T_pred"], gm["Z_km"], color=COLORS[m], lw=1.6, label=m)
            ax[1].plot(100 * (np.exp(gm["lnP_pred"] - gm["lnP_ref"]) - 1), gm["Z_km"], color=COLORS[m], lw=1.6)
            nr = np.exp(gm["lnP_pred"] - gm["lnP_ref"]) * gm["T_ref"] / gm["T_pred"]
            ax[2].plot(100 * (nr - 1), gm["Z_km"], color=COLORS[m], lw=1.6)
        for a in ax[1:]:
            a.axvline(0, color="k", lw=0.8)
            a.set_ylim(zg.min() - 1, zg.max() + 1)
        ax[0].set_xlabel("T (K)"); ax[1].set_xlabel("P error (%)"); ax[2].set_xlabel("n error (%)")
        ax[0].set_ylabel(f"Z (km) — {case[1]} km {case[2]} gap")
        if r == 0:
            ax[0].legend(fontsize=7)
    fig.suptitle(f"Profile {pid}: reconstruction inside the gap (shaded)")
    fig.tight_layout(); savefig(fig, folder, "fig02_example_reconstructions")


def fig_gap_size(S_BY_GAP, locations, folder):
    fig, axes = plt.subplots(3, len(locations), figsize=(5 * len(locations), 11), squeeze=False)
    for j, loc in enumerate(locations):
        for i, (met, lab) in enumerate([("T_MAE", "T MAE (K)"), ("P_relMAE", "P error (%)"), ("n_relMAE", "n error (%)")]):
            ax = axes[i, j]
            for m in [o for o in ORDER if o not in (NAMES["mlp_ft"], NAMES["pinn_ft"])]:
                s = S_BY_GAP[(S_BY_GAP.location == loc) & (S_BY_GAP.method == m)].sort_values("gap_km")
                if len(s):
                    ax.errorbar(s["gap_km"], s[met], yerr=[s[met] - s[met + "_lo"], s[met + "_hi"] - s[met]],
                                color=COLORS[m], marker="o", ms=4, capsize=2, lw=2.2 if m == OURS else 1.2, label=m)
            ax.set_yscale("log"); ax.set_xlabel("Gap size (km)"); ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{loc} gap")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); savefig(fig, folder, "fig03_error_vs_gap_size")


def fig_hydro(M, floor, folder):
    order = [o for o in ORDER if o in set(M.method)]
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [M.loc[M.method == m, "Hydro_RMS"].dropna().values for m in order]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=30, ha="right")
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.7)
    ax.axhline(floor, color="k", ls="--", lw=1, label="reference-data floor")
    ax.set_yscale("log"); ax.set_ylabel(r"RMS hydrostatic residual"); ax.legend()
    fig.tight_layout(); savefig(fig, folder, "fig04_hydrostatic_residual")


def fig_calibration(PW, Q, folder):
    from .metrics import QGRID
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for m in [NAMES["gp"], NAMES["eof"], NAMES["mlp_rft"], NAMES["pinn_rft"]]:
        g = PW[(PW.method == m) & PW.T_std.notna()]
        if not len(g):
            continue
        eT = np.abs(g.T_pred - g.T_ref); sT = np.sqrt(g.T_std ** 2 + g.sig_T ** 2)
        eP = np.abs(g.lnP_pred - g.lnP_ref); sP = np.sqrt(g.lnP_std ** 2 + g.sig_lnP ** 2)
        ax[0].plot(QGRID, [np.mean(eT <= stats.norm.ppf(0.5 + q / 2) * sT) for q in QGRID], "o--", ms=2, lw=1, alpha=0.6, color=COLORS[m])
        ax[1].plot(QGRID, [np.mean(eP <= stats.norm.ppf(0.5 + q / 2) * sP) for q in QGRID], "o--", ms=2, lw=1, alpha=0.6, color=COLORS[m])
        if Q:
            locs = g["location"].values
            for a, r, var in ((ax[0], (eT / sT).values, "T"), (ax[1], (eP / sP).values, "P")):
                cov = [np.mean([ri <= Q.get((m, l, var, float(q)), np.inf) for ri, l in zip(r, locs)]) for q in QGRID]
                a.plot(QGRID, cov, "o-", ms=3, color=COLORS[m], label=f"{m} (conformal)")
        bins = pd.qcut(g.T_std, 5, duplicates="drop")
        b = pd.DataFrame(dict(s=g.T_std, e=eT)).groupby(bins, observed=True).mean()
        ax[2].plot(b["s"], b["e"], "o-", color=COLORS[m], label=m)
    for a, t in zip(ax[:2], ["Temperature", "ln P"]):
        a.plot([0, 1], [0, 1], "k--", lw=1); a.set_xlabel("Nominal coverage"); a.set_ylabel("Empirical coverage")
        a.set_title(f"Calibration: {t} (dashed = raw)")
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    ax[2].set_xlabel(r"Predicted $\sigma_T$ (K), quintile mean"); ax[2].set_ylabel("Mean |T error| (K)")
    ax[2].set_title("Does uncertainty rank errors?"); ax[2].legend(fontsize=8)
    fig.tight_layout(); savefig(fig, folder, "fig05_uncertainty_calibration")


def fig_paired(SIG, folder):
    mets = ["T_MAE", "P_relMAE", "n_relMAE"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, met in zip(ax, mets):
        s = SIG[SIG.metric == met].set_index("competitor").reindex([o for o in ORDER if o != OURS]).dropna(subset=["mean_diff"])
        y = np.arange(len(s))
        a.errorbar(s["mean_diff"], y, xerr=[s["mean_diff"] - s["ci_lo"], s["ci_hi"] - s["mean_diff"]], fmt="o", color="#2166ac", capsize=3)
        for yi, (_, r) in zip(y, s.iterrows()):
            a.text(r["ci_hi"], yi, "  *" if r["significant"] else "", va="center")
        a.axvline(0, color="k", lw=1); a.set_yticks(y); a.set_yticklabels(s.index)
        a.set_title(f"{met}: competitor − PINN (>0 favours PINN)", fontsize=9)
    fig.tight_layout(); savefig(fig, folder, "fig06_paired_improvement")


def fig_noise(S_NOISE, folder):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in S_NOISE.method.unique():
        s = S_NOISE[S_NOISE.method == m].sort_values("noise_mult")
        for a, met in zip(ax, ["T_MAE", "P_relMAE"]):
            a.errorbar(s.noise_mult, s[met], yerr=[s[met] - s[met + "_lo"], s[met + "_hi"] - s[met]],
                       marker="o", capsize=2, color=COLORS.get(m), label=m)
    ax[0].set_ylabel("T MAE (K)"); ax[1].set_ylabel("P error (%)")
    for a in ax:
        a.set_xlabel(r"Added noise ($\times\sigma_{ret}$)"); a.set_yscale("log")
    ax[0].legend(fontsize=7); fig.tight_layout(); savefig(fig, folder, "fig07_noise_robustness")


def fig_lambda(S_LAM, lambdas, floor, folder):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    zero = lambdas[1] / 10 if len(lambdas) > 1 and lambdas[1] > 0 else 1e-3
    for loc in S_LAM.location.unique():
        s = S_LAM[S_LAM.location == loc].sort_values("lam_override")
        x = s.lam_override.replace(0, zero)
        for a, met in zip(ax, ["T_MAE", "P_relMAE", "Hydro_RMS"]):
            a.errorbar(x, s[met], yerr=[s[met] - s[met + "_lo"], s[met + "_hi"] - s[met]], marker="o", capsize=2, label=loc)
            a.set_xscale("log"); a.set_xlabel(r"$\lambda$ (leftmost point = 0)"); a.set_ylabel(met)
    ax[2].axhline(floor, color="k", ls="--", lw=1); ax[0].legend()
    fig.tight_layout(); savefig(fig, folder, "fig08_lambda_sensitivity")


def fig_holdout(S_HOLD, folder):
    piv = S_HOLD.pivot(index="method", columns="holdout_MY", values="T_MAE")
    piv = piv.reindex([o for o in ORDER if o in piv.index])
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(piv.columns)
    for k, my in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + k * w, piv[my], w, label=f"held-out MY {my}")
    ax.set_xticks(np.arange(len(piv)) + 0.4 - w / 2); ax.set_xticklabels(piv.index, rotation=30, ha="right")
    ax.set_ylabel("T MAE (K)"); ax.legend(); fig.tight_layout(); savefig(fig, folder, "fig09_year_holdout")


def fig_training(log_dir, folder):
    files = sorted(log_dir.glob("history_main_*_seed*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for f in files:
        h = pd.read_csv(f)
        fam = "PINN" if "_PINN_" in f.name else "MLP"
        c = "#2166ac" if fam == "PINN" else "#d6604d"
        ax[0].plot(h.epoch, h.val_data, color=c, alpha=0.6, lw=1)
        if fam == "PINN":
            ax[1].plot(h.epoch, h.train_phys, color=c, alpha=0.6, lw=1)
    ax[0].set_yscale("log"); ax[0].set_title("Validation data loss (blue: PINN, red: MLP)"); ax[0].set_xlabel("epoch")
    ax[1].set_yscale("log"); ax[1].set_title("PINN physics loss during pre-training"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); savefig(fig, folder, "fig10_training_curves")
