"""The complete workflow as a sequence of controllable steps.

    data -> split -> pretrain -> tune -> e1 -> calibrate -> e2 -> e3 -> e4 -> report

Each step can be run on its own (its prerequisites run automatically and are cached on
disk), so you can stop, change a setting and resume.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .baselines import fit_eof_pair, fit_rf
from .benchmark import NAMES, OURS, Runner
from .config import save_config
from .data import (build_cases, load_raw, physics_audit, preprocess, quality_control,
                   stratified_profile_split)
from .metrics import (POINT_METRICS, compute_metrics, conformal_coverage, fit_conformal, paired_tests,
                      summarize, uncertainty_summary)
from .nn import NNToolkit, compute_norm
from .physics import Physics

STEPS = ["data", "split", "pretrain", "tune", "e1", "calibrate", "e2", "e3", "e4", "report"]
DEPENDS = {"data": [], "split": ["data"], "pretrain": ["split"], "tune": ["pretrain"],
           "e1": ["tune"], "calibrate": ["e1"], "e2": ["tune"], "e3": ["pretrain"],
           "e4": ["tune"], "report": ["e1", "calibrate"]}


class Pipeline:
    def __init__(self, cfg: dict, verbose: bool = True):
        self.cfg, self.verbose = cfg, verbose
        self.out = Path(cfg["OUTPUT_DIR"])
        self.fig, self.tab = self.out / "figures", self.out / "tables"
        self.ckpt, self.raw = self.out / "checkpoints", self.out / "raw"
        for d in (self.fig, self.tab, self.ckpt, self.raw):
            d.mkdir(parents=True, exist_ok=True)
        self.phys = Physics()
        self.done: set[str] = set()
        self.R: dict = {}           # results of each step
        save_config(cfg, self.out / "run_config.yaml")

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ orchestration
    def run(self, steps=None):
        steps = steps or STEPS
        t0 = time.time()
        for s in steps:
            self._ensure(s)
        self.log(f"\nFinished {steps} in {(time.time() - t0) / 60:.1f} min. Outputs in {self.out.resolve()}")
        return self.R

    def _ensure(self, step):
        if step in self.done:
            return
        if step not in STEPS:
            raise ValueError(f"Unknown step {step!r}; choose from {STEPS}")
        for dep in DEPENDS[step]:
            self._ensure(dep)
        skip = {"e2": "RUN_NOISE", "e3": "RUN_LAMBDA_SWEEP", "e4": "RUN_YEAR_HOLDOUT"}
        if step in skip and not self.cfg[skip[step]]:
            self.log(f"[{step}] disabled by {skip[step]}=False")
        else:
            self.log(f"\n=== step: {step} ===")
            getattr(self, f"step_{step}")()
        self.done.add(step)

    # ------------------------------------------------------------------ steps
    def step_data(self):
        cfg = self.cfg
        raw = load_raw(cfg)
        lv = preprocess(raw, self.phys)
        lv, meta, qc = quality_control(lv, cfg)
        audit = physics_audit(lv, self.phys)
        audit.update(qc)
        audit["data_source"] = cfg["DATA_SOURCE"]
        json.dump(audit, open(self.out / "data_audit.json", "w"), indent=2)
        self.lv, self.meta = lv, meta
        self.R["audit"] = audit
        self.log(json.dumps(audit, indent=2))
        self.log(meta.groupby("Mars_Year")["Solar_Long"].agg(["count", "min", "max"]).round(1).to_string())
        plots.fig_data_overview(lv, meta, self.fig)

    def step_split(self):
        cfg = self.cfg
        tr, va, te = stratified_profile_split(self.meta, cfg["SPLIT_FRACS"], cfg["SPLIT_SEED"])
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        if cfg["MAX_TEST_PROFILES"]:
            te = te[:cfg["MAX_TEST_PROFILES"]]
        self.TRAIN_IDS, self.VAL_IDS, self.TEST_IDS = tr, va, te
        json.dump(dict(train=tr, val=va, test=te), open(self.out / "splits.json", "w"), indent=2)
        lv = self.lv
        self.train_df = lv[lv.Profile_ID.isin(tr)].reset_index(drop=True)
        self.val_df = lv[lv.Profile_ID.isin(va)].reset_index(drop=True)
        self.test_df = lv[lv.Profile_ID.isin(te)].reset_index(drop=True)
        for name, ids in [("train", tr), ("val", va), ("test", te)]:
            self.log(f"{name:5s}: {len(ids):3d} profiles | per Mars year:",
                     self.meta.loc[ids].groupby("Mars_Year").size().to_dict())
        configs = [(s, l) for s in cfg["GAP_SIZES_KM"] for l in cfg["GAP_LOCATIONS"]]
        self.CASES = build_cases(self.test_df, te, configs, cfg["MIN_OBSERVED_KM"])
        self.log(f"Main benchmark cases: {len(self.CASES)} of {len(te) * len(configs)} possible")

    def step_pretrain(self):
        cfg = self.cfg
        self.nnt = NNToolkit(cfg, self.phys, self.train_df, self.ckpt, self.raw)
        self.LAM = {"MLP": 0.0, "PINN": cfg["LAMBDA_PHYS"]}
        self.runner = Runner(cfg, self.phys, self.nnt, self.LAM, self.raw, self.verbose)
        self.log(f"device={self.nnt.device}  H_ref={self.phys.H_ref_km:.2f} km  m={self.phys.m_gmol:.3f} g/mol")
        t0 = time.time()
        self.BASE = {fam: [] for fam in self.LAM}
        for fam, lam in self.LAM.items():
            for s in cfg["SEEDS"]:
                self.BASE[fam].append((s, self.nnt.pretrain(self.train_df, self.val_df, fam, lam, s,
                                                            verbose=self.verbose)))
                self.log(f"  pre-trained {fam} seed {s}  ({time.time() - t0:.0f}s)")
        self.RF = fit_rf(self.train_df)
        self.EOF = fit_eof_pair(self.train_df, cfg)
        self.log(f"EOF modes kept: T {self.EOF['T']['k']}, lnP {self.EOF['P']['k']}")

    def step_tune(self):
        cfg = self.cfg
        hp_file = self.out / "selected_hyperparameters.json"
        if cfg["HP_OVERRIDE"]:
            HP = cfg["HP_OVERRIDE"]
            self.log("Using HP_OVERRIDE:", HP)
        elif hp_file.exists():
            HP = json.load(open(hp_file))
            self.log("Loaded selected hyper-parameters:", HP)
        else:
            tune_cases = build_cases(self.val_df, self.VAL_IDS, cfg["TUNE_CONFIGS"], cfg["MIN_OBSERVED_KM"])
            rows = []
            for fam in ["MLP", "PINN"]:
                for lam in ([0.0] if fam == "MLP" else cfg["TUNE_LAMBDAS"]):
                    for alpha in cfg["TUNE_ALPHAS"]:
                        for es in cfg["TUNE_ES"]:
                            hpo = {fam: dict(lam=lam, alpha=alpha, es=es)}
                            pw = self.runner.run_benchmark(
                                tune_cases, self.val_df, {fam: self.BASE[fam]}, None,
                                f"tune_{fam}_lam{lam}_alpha{alpha}_es{int(es)}",
                                methods=[f"{fam.lower()}_rft"], seeds_limit=1, hp_override=hpo)
                            e = pw.assign(aT=(pw.T_pred - pw.T_ref).abs(),
                                          aP=100 * (np.exp(pw.lnP_pred - pw.lnP_ref) - 1).abs())
                            pc = e.groupby(["Profile_ID", "case_id"])[["aT", "aP"]].mean().groupby("Profile_ID").mean()
                            rows.append(dict(family=fam, lam=lam, alpha=alpha, es=es,
                                             T_MAE=pc["aT"].mean(), P_relMAE=pc["aP"].mean()))
            T = pd.DataFrame(rows)
            T["score"] = (T.groupby("family")["T_MAE"].transform(lambda x: x / x.min())
                          + T.groupby("family")["P_relMAE"].transform(lambda x: x / x.min()))
            T.to_csv(self.tab / "tuning_validation.csv", index=False)
            self.log(T.round(4).to_string())
            HP = {}
            for fam, g in T.groupby("family"):
                r = g.loc[g["score"].idxmin()]
                HP[fam] = dict(lam=float(r["lam"]), alpha=float(r["alpha"]), es=bool(r["es"]))
            json.dump(HP, open(hp_file, "w"), indent=2)
        self.runner.HP = self.HP = HP
        self.R["HP"] = HP
        self.log("Selected (validation):", HP)

    def step_e1(self):
        PW = self.runner.run_benchmark(self.CASES, self.test_df, self.BASE, self.RF, "pointwise_main", eof=self.EOF)
        M = compute_metrics(PW, self.phys)
        M.to_csv(self.tab / "metrics_per_case_main.csv", index=False)
        nb = self.cfg["N_BOOT"]
        S = summarize(M, POINT_METRICS, n_boot=nb)
        S_LOC = summarize(M, POINT_METRICS, by=("location", "method"), n_boot=nb)
        S_GAP = summarize(M, POINT_METRICS, by=("location", "gap_km", "method"), n_boot=nb)
        SIG = pd.concat([paired_tests(M, m, n_boot=nb, alpha=self.cfg["ALPHA"])
                         for m in ["T_MAE", "P_relMAE", "n_relMAE"]], ignore_index=True)
        SIG_LOC = pd.concat([paired_tests(M, m, by="location", n_boot=nb, alpha=self.cfg["ALPHA"])
                             for m in ["T_MAE", "P_relMAE", "n_relMAE"]], ignore_index=True)
        for name, d in [("summary_overall", S), ("summary_by_location", S_LOC), ("summary_by_gap", S_GAP),
                        ("significance_overall", SIG), ("significance_by_location", SIG_LOC)]:
            d.to_csv(self.tab / f"{name}.csv", index=False)
        self.PW, self.M = PW, M
        self.R.update(S_OVERALL=S, S_BY_LOC=S_LOC, S_BY_GAP=S_GAP, SIG=SIG, SIG_LOC=SIG_LOC)
        self.log(S.set_index("method")[["n_profiles", "T_MAE", "T_within_2sig", "P_relMAE", "n_relMAE", "Hydro_RMS"]]
                 .sort_values("T_MAE").round(3).to_string())
        uq_methods = [NAMES[k] for k in ("gp", "eof", "mlp_rft", "pinn_rft", "mlp_ft", "pinn_ft", "mlp_zs", "pinn_zs")]
        UQ = uncertainty_summary(M, PW, uq_methods, n_boot=nb)
        UQ.to_csv(self.tab / "uncertainty_summary.csv", index=False)
        self.R["UQ"] = UQ

    def step_calibrate(self):
        cfg = self.cfg
        configs = [(s, l) for s in cfg["CAL_GAP_SIZES_KM"] for l in cfg["GAP_LOCATIONS"]]
        cal_cases = build_cases(self.val_df, self.VAL_IDS, configs, cfg["MIN_OBSERVED_KM"])
        PW_CAL = self.runner.run_benchmark(cal_cases, self.val_df, self.BASE, None, "pointwise_calibration",
                                           methods=["gp", "eof", "mlp_rft", "pinn_rft"], eof=self.EOF)
        self.Q = fit_conformal(PW_CAL)
        CC = conformal_coverage(self.PW, self.Q)
        ccols = [c for c in CC.columns if "_ccov" in c]
        UQC = summarize(CC, ccols, n_boot=cfg["N_BOOT"])
        UQ = self.R["UQ"].merge(UQC[["method"] + ccols], on="method", how="left")
        UQ.to_csv(self.tab / "uncertainty_summary.csv", index=False)
        self.R["UQ"] = UQ
        self.log(UQ.set_index("method")[[c for c in UQ.columns if c.endswith(("cov90", "cov95"))]].round(3).to_string())

    def step_e2(self):
        cfg = self.cfg
        cases = [(pid, s, loc, (("noise_mult", k),))
                 for (pid, s, loc) in build_cases(self.test_df, self.TEST_IDS, cfg["NOISE_CASES"], cfg["MIN_OBSERVED_KM"])
                 for k in cfg["NOISE_MULTIPLIERS"]]
        PW = self.runner.run_benchmark(cases, self.test_df, self.BASE, None, "pointwise_noise",
                                       methods=["lin", "pchip", "hydro", "gp", "eof", "mlp_rft", "pinn_rft"],
                                       seeds_limit=cfg["NOISE_SEEDS"], eof=self.EOF)
        M = compute_metrics(PW, self.phys, keys=("case_id", "Profile_ID", "gap_km", "location", "noise_mult", "method"))
        S = summarize(M, ["T_MAE", "P_relMAE", "n_relMAE"], by=("noise_mult", "method"), n_boot=cfg["N_BOOT"])
        S.to_csv(self.tab / "summary_noise.csv", index=False)
        self.R["S_NOISE"] = S
        self.log(S.pivot(index="method", columns="noise_mult", values="T_MAE").round(3).to_string())

    def step_e3(self):
        cfg = self.cfg
        base_cases = build_cases(self.test_df, self.TEST_IDS, cfg["LAMBDA_CASES"], cfg["MIN_OBSERVED_KM"])
        parts = []
        for lam in cfg["LAMBDAS"]:
            cases = [(pid, s, loc, (("lam_override", lam),)) for (pid, s, loc) in base_cases]
            parts.append(self.runner.run_benchmark(cases, self.test_df, {"PINN": self.BASE["PINN"]}, None,
                                                   f"pointwise_lambda_{lam}", methods=["pinn_ft"], seeds_limit=1))
        PW = pd.concat(parts, ignore_index=True)
        M = compute_metrics(PW, self.phys, keys=("case_id", "Profile_ID", "gap_km", "location", "lam_override", "method"))
        S = summarize(M, ["T_MAE", "P_relMAE", "n_relMAE", "Hydro_RMS"], by=("location", "lam_override"),
                      n_boot=cfg["N_BOOT"])
        S.to_csv(self.tab / "summary_lambda.csv", index=False)
        self.R["S_LAM"] = S
        self.log(S[["location", "lam_override", "T_MAE", "P_relMAE", "n_relMAE", "Hydro_RMS"]].round(4).to_string())

    def step_e4(self):
        cfg = self.cfg
        methods = ["hydro", "gp", "eof", "rf", "mlp_zs", "pinn_zs", "mlp_rft", "pinn_rft"]
        seeds = cfg["SEEDS"][:cfg["HOLDOUT_SEEDS"]]
        lv, meta = self.lv, self.meta
        parts = []
        for my in sorted(meta["Mars_Year"].unique()):
            test_ids = sorted(meta.index[meta["Mars_Year"] == my])[:cfg["HOLDOUT_MAX_TEST"]]
            rest = meta.index[meta["Mars_Year"] != my].values
            if len(test_ids) < 3 or len(rest) < 10:
                continue
            rest = np.random.default_rng(cfg["SPLIT_SEED"]).permutation(rest)
            n_va = max(1, int(0.15 * len(rest)))
            tr_h = lv[lv.Profile_ID.isin(rest[n_va:])]
            va_h = lv[lv.Profile_ID.isin(rest[:n_va])]
            te_h = lv[lv.Profile_ID.isin(test_ids)]
            norm_h = compute_norm(tr_h)
            base_h = {fam: [(s, self.nnt.pretrain(tr_h, va_h, fam, self.LAM[fam], s, tag=f"holdMY{int(my)}", norm=norm_h))
                            for s in seeds] for fam in self.LAM}
            rf_h, eof_h = fit_rf(tr_h), fit_eof_pair(tr_h, cfg)
            cases = build_cases(te_h, test_ids, cfg["HOLDOUT_CASES"], cfg["MIN_OBSERVED_KM"])
            pw = self.runner.run_benchmark(cases, te_h, base_h, rf_h, f"pointwise_holdout_MY{int(my)}",
                                           methods=methods, eof=eof_h)
            pw["holdout_MY"] = int(my)
            parts.append(pw)
        if not parts:
            return
        PW = pd.concat(parts, ignore_index=True)
        M = compute_metrics(PW, self.phys, keys=("case_id", "Profile_ID", "gap_km", "location", "holdout_MY", "method"))
        S = summarize(M, ["T_MAE", "P_relMAE", "n_relMAE"], by=("holdout_MY", "method"), n_boot=cfg["N_BOOT"])
        S.to_csv(self.tab / "summary_year_holdout.csv", index=False)
        self.R["S_HOLD"] = S
        self.log(S.pivot(index="method", columns="holdout_MY", values="T_MAE").round(3).to_string())

    # ------------------------------------------------------------------ report
    def step_report(self):
        R = self.R
        # results of optional experiments run in an earlier session are read back from disk
        for key, name in [("S_NOISE", "summary_noise"), ("S_LAM", "summary_lambda"),
                          ("S_HOLD", "summary_year_holdout")]:
            f = self.tab / f"{name}.csv"
            if key not in R and f.exists():
                R[key] = pd.read_csv(f)
        if "S_OVERALL" in R:
            plots.fig_examples(self.PW, self.test_df, self.TEST_IDS, self.fig)
            plots.fig_gap_size(R["S_BY_GAP"], self.cfg["GAP_LOCATIONS"], self.fig)
            plots.fig_hydro(self.M, self.phys.floor, self.fig)
            plots.fig_calibration(self.PW, getattr(self, "Q", None), self.fig)
            if len(R["SIG"]):
                plots.fig_paired(R["SIG"], self.fig)
        if "S_NOISE" in R:
            plots.fig_noise(R["S_NOISE"], self.fig)
        if "S_LAM" in R:
            plots.fig_lambda(R["S_LAM"], self.cfg["LAMBDAS"], self.phys.floor, self.fig)
        if "S_HOLD" in R:
            plots.fig_holdout(R["S_HOLD"], self.fig)
        plots.fig_training(self.raw, self.fig)
        self._latex_tables()
        text = self._summary_text()
        (self.out / "results_summary.md").write_text(text)
        env = dict(python=sys.version.split()[0], platform=platform.platform())
        json.dump(env, open(self.out / "environment.json", "w"), indent=2)
        self.log("\n" + text)

    def _latex_tables(self):
        R = self.R
        if "S_OVERALL" not in R:
            return

        def ci(r, m, d=2):
            return f"{r[m]:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]"
        order = [o for o in plots.ORDER if o in set(R["S_OVERALL"].method)]
        main = R["S_OVERALL"].set_index("method").reindex(order)
        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Method & T MAE (K) & within $2\sigma_T$ & P error (\%) & $n$ error (\%) & Hydro.\ residual \\", r"\midrule"]
        for m, r in main.iterrows():
            lines.append(f"{m} & {ci(r, 'T_MAE')} & {r['T_within_2sig']:.2f} & {ci(r, 'P_relMAE')} & "
                         f"{ci(r, 'n_relMAE')} & {r['Hydro_RMS']:.3g} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (self.tab / "table1_main_results.tex").write_text("\n".join(lines).replace("%", r"\%").replace(r"\\%", r"\%"))

    def _summary_text(self):
        R, lines = self.R, []
        a = R.get("audit", {})
        lines.append(f"Data source: {a.get('data_source')} | profiles kept {a.get('profiles_kept')} | "
                     f"m = {a.get('mean_molecular_mass_gmol', float('nan')):.3f} g/mol | "
                     f"reference hydrostatic floor {a.get('reference_hydro_residual_RMS_median', float('nan')):.3g}")
        if "S_OVERALL" in R:
            S, SIG = R["S_OVERALL"], R["SIG"]
            o = S.set_index("method")
            best = S.sort_values("T_MAE").iloc[0]
            lines.append(f"Test set: {len(self.TEST_IDS)} profiles, {self.M.case_id.nunique()} gap cases.")
            if len(self.TEST_IDS) < 6:
                lines.append("NOTE: fewer than 6 test profiles - significance tests cannot reach p < 0.05 (smoke test).")
            lines.append(f"Lowest temperature MAE: {best.method} ({best.T_MAE:.2f} K).")
            if OURS in o.index:
                r = o.loc[OURS]
                lines.append(f"{OURS}: T MAE {r.T_MAE:.2f} K, P error {r.P_relMAE:.2f} %, n error {r.n_relMAE:.2f} %, "
                             f"hydrostatic residual {r.Hydro_RMS:.3g}.")
            for met, lab in [("T_MAE", "temperature"), ("P_relMAE", "pressure"), ("n_relMAE", "density")]:
                s = SIG[SIG.metric == met] if len(SIG) else SIG
                if not len(s):
                    continue
                better = s[s.significant & (s.mean_diff > 0)].competitor.tolist()
                worse = s[s.significant & (s.mean_diff < 0)].competitor.tolist()
                lines.append(f"[{lab}] significantly better than: {better or 'none'}; worse than: {worse or 'none'}.")
        if "HP" in R:
            lines.append(f"[selection] hyper-parameters chosen on validation profiles: {R['HP']}")
        if "UQ" in R and "T_ccov95" in R["UQ"]:
            for _, r in R["UQ"].dropna(subset=["T_ccov95"]).iterrows():
                lines.append(f"[uncertainty] {r.method}: raw 95% covers {100 * r.T_cov95:.0f}% of T; "
                             f"conformal 95% covers {100 * r.T_ccov95:.0f}% of T and {100 * r.P_ccov95:.0f}% of lnP.")
        return "\n".join("- " + l for l in lines)
