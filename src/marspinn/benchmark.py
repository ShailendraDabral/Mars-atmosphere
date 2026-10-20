"""Run every method on one profile with one gap, and loop over many cases with caching."""
from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import eof_predict, gp_predict, hydrostatic_integration, interp_extrap
from .data import IN_COLS, make_gap

NAMES = dict(lin="Linear", pchip="PCHIP", hydro="Hydrostatic int.", gp="Gaussian process",
             eof="EOF (opt. interp.)", rf="Random forest", mlp_zs="MLP (zero-shot)",
             pinn_zs="PINN (zero-shot)", mlp_ft="MLP (fine-tuned)", pinn_ft="PINN (fine-tuned)",
             mlp_rft="MLP (reg. fine-tuned)", pinn_rft="PINN (reg. fine-tuned)")
OURS = NAMES["pinn_rft"]
ALL_METHODS = list(NAMES)


def case_key(case) -> str:
    return "|".join(map(str, case))


def case_seed(case) -> int:
    return zlib.crc32(case_key(case).encode()) % 2 ** 31


class Runner:
    """Holds everything a reconstruction needs: config, physics, NN toolkit, physics weights
    per family (LAM) and the selected regularised fine-tuning hyper-parameters (HP)."""

    def __init__(self, cfg, phys, nnt, lam: dict, cache_dir: Path, verbose=True):
        self.cfg, self.phys, self.nnt, self.LAM = cfg, phys, nnt, lam
        self.HP = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    # ------------------------------------------------------------------ one case
    def run_case(self, prof, gap, base, rf, methods=ALL_METHODS, lam_override=None, noise_mult=0.0,
                 rng_seed=0, seeds_limit=None, eof=None, hp_override=None):
        cfg, nnt = self.cfg, self.nnt
        rng = np.random.default_rng(rng_seed)
        obs, hid = prof[~gap].copy(), prof[gap]
        if noise_mult > 0:
            se = nnt.sigma_eff(obs)
            obs["Temperature"] += rng.normal(0, noise_mult * se[:, 0])
            obs["lnP"] += rng.normal(0, noise_mult * se[:, 1])
        zo, To, Po, zh = obs["Z_km"].values, obs["Temperature"].values, obs["lnP"].values, hid["Z_km"].values
        P = {}
        if "lin" in methods:
            P["lin"] = (interp_extrap(zo, To, zh), interp_extrap(zo, Po, zh), None, None)
        if "pchip" in methods:
            P["pchip"] = (interp_extrap(zo, To, zh, "pchip"), interp_extrap(zo, Po, zh, "pchip"), None, None)
        if "hydro" in methods:
            P["hydro"] = (*hydrostatic_integration(zo, To, Po, zh, self.phys), None, None)
        if "gp" in methods:
            (mT, sT), (mP, sP) = gp_predict(zo, To, zh), gp_predict(zo, Po, zh)
            P["gp"] = (mT, mP, sT, sP)
        if "rf" in methods and rf is not None:
            y = rf.predict(hid[IN_COLS].values)
            P["rf"] = (y[:, 0], y[:, 1], None, None)
        if "eof" in methods and eof is not None:
            mT, sT = eof_predict(eof["T"], zo, To, obs["Sigma_T"].values, zh)
            mP, sP = eof_predict(eof["P"], zo, Po, obs["sig_lnP"].values, zh)
            P["eof"] = (mT, mP, sT, sP)
        nn_needed = [m for m in methods if m.endswith(("_zs", "_ft", "_rft"))]
        if nn_needed and base:
            gl, gh = zh.min(), zh.max()
            colloc = np.vstack([nnt.densify(prof, cfg["N_COLLOC_GAP"], rng, gl, gh),
                                nnt.densify(prof, cfg["N_COLLOC_GAP"] // 2, rng)])
            for fam in base:
                key = fam.lower()
                members = base[fam][:seeds_limit] if seeds_limit else base[fam]
                zs, ft, rft = [], [], []
                for _, b in members:
                    if f"{key}_zs" in methods:
                        zs.append(nnt.predict(b, hid))
                    if f"{key}_ft" in methods:
                        lam = self.LAM[fam] if lam_override is None or fam == "MLP" else lam_override
                        ft.append(nnt.predict(nnt.finetune(b, obs, colloc, lam), hid))
                    if f"{key}_rft" in methods:
                        h = (hp_override or self.HP)[fam]
                        rft.append(nnt.predict(nnt.finetune_reg(b, obs, colloc, h["lam"], h["alpha"], gl, gh,
                                                                h.get("es", True)), hid))
                for tag, lst in ((f"{key}_zs", zs), (f"{key}_ft", ft), (f"{key}_rft", rft)):
                    if lst:
                        A = np.array(lst)   # (members, 2, n)
                        sd = A.std(axis=0, ddof=1) if len(lst) > 1 else np.full(A.shape[1:], np.nan)
                        P[tag] = (A[:, 0].mean(0), A[:, 1].mean(0), sd[0], sd[1])
        rows = []
        for k, (Tp, Pp, sT, sP) in P.items():
            rows.append(pd.DataFrame(dict(method=NAMES[k], Z_km=zh, T_ref=hid["Temperature"].values,
                                          lnP_ref=hid["lnP"].values, sig_T=hid["Sigma_T"].values,
                                          sig_lnP=hid["sig_lnP"].values, T_pred=Tp, lnP_pred=Pp,
                                          T_std=np.nan if sT is None else sT,
                                          lnP_std=np.nan if sP is None else sP)))
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------ many cases
    def run_benchmark(self, cases, df, base, rf, cache_name, progress=None, **kw):
        """Run cases (pid, size, location[, extra-kwargs]); results are cached per case so an
        interrupted run resumes where it stopped."""
        cache = self.cache_dir / f"{cache_name}.pkl"
        done = pd.read_pickle(cache) if cache.exists() else pd.DataFrame()
        done_keys = set(done["case_id"].unique()) if len(done) else set()
        parts, t0 = [done], time.time()
        todo = [c for c in cases if case_key(c) not in done_keys]
        if self.verbose:
            print(f"  {cache_name}: {len(cases) - len(todo)} cached, {len(todo)} to run")
        for i, case in enumerate(todo):
            pid, size, loc = case[:3]
            prof = df[df["Profile_ID"] == pid].sort_values("Z_km").reset_index(drop=True)
            gap = make_gap(prof["Z_km"].values, size, loc, self.cfg["MIN_OBSERVED_KM"])
            extra = dict(case[3]) if len(case) > 3 else {}
            out = self.run_case(prof, gap, base, rf, rng_seed=case_seed(case[:3]), **kw, **extra)
            out["case_id"] = case_key(case)
            out["Profile_ID"] = pid
            out["gap_km"] = size
            out["location"] = loc
            for k, v in extra.items():
                out[k] = v
            parts.append(out)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                pd.concat(parts, ignore_index=True).to_pickle(cache)
                if self.verbose:
                    el = time.time() - t0
                    print(f"    {i + 1}/{len(todo)} cases  {el:.0f}s  (~{el / (i + 1) * (len(todo) - i - 1):.0f}s left)")
            if progress:
                progress(i + 1, len(todo))
        return pd.concat(parts, ignore_index=True)
