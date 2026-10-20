"""Neural networks: data-only MLP and physics-informed PINN (identical architecture).

The network maps x = (Z, lat, sin lon, cos lon, sin Ls, cos Ls) to (T, ln P). Number density
is derived as n = P / (k_B T), so the ideal-gas law holds exactly. Hydrostatic balance enters
as the dimensionless residual  r = H_ref dlnP/dZ + T_ref/T  evaluated at collocation points.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import IN_COLS, OUT_COLS
from .physics import Physics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_norm(df):
    return dict(in_mu=df[IN_COLS].mean().values, in_sd=np.maximum(df[IN_COLS].std().values, 1e-6),
                out_mu=df[OUT_COLS].mean().values, out_sd=df[OUT_COLS].std().values)


class ProfileNet(nn.Module):
    """Tanh MLP with internal input/output scaling; forward() works in physical units."""

    def __init__(self, norm, hidden=64, depth=4):
        super().__init__()
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(np.asarray(v, dtype=np.float32)))
        layers, d = [], len(IN_COLS)
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, len(OUT_COLS)))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net((x - self.in_mu) / self.in_sd) * self.out_sd + self.out_mu


class NNToolkit:
    """Bundles the normalisation, weighting, physics and training routines for one dataset."""

    def __init__(self, cfg: dict, phys: Physics, train_df: pd.DataFrame, ckpt_dir: Path, log_dir: Path):
        self.cfg, self.phys = cfg, phys
        dev = cfg.get("DEVICE", "auto")
        self.device = torch.device("cuda" if (dev == "auto" and torch.cuda.is_available()) or dev == "cuda" else "cpu")
        self.norm = compute_norm(train_df)
        self.w_norm = (1.0 / self.sigma_eff(train_df) ** 2).mean(axis=0)
        self.ckpt_dir, self.log_dir = Path(ckpt_dir), Path(log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def sigma_eff(self, df):
        return np.column_stack([np.maximum(df["Sigma_T"].values, self.cfg["SIGMA_FLOOR_T"]),
                                np.maximum(df["sig_lnP"].values, self.cfg["SIGMA_FLOOR_LNP"])])

    def to_t(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def xyw(self, df):
        if self.cfg["SIGMA_WEIGHTING"]:
            W = (1.0 / self.sigma_eff(df) ** 2) / self.w_norm
        else:
            W = np.ones((len(df), 2))
        return self.to_t(df[IN_COLS].values), self.to_t(df[OUT_COLS].values), self.to_t(W)

    def new_model(self, norm=None):
        return ProfileNet(norm if norm is not None else self.norm, self.cfg["HIDDEN"], self.cfg["DEPTH"]).to(self.device)

    @staticmethod
    def data_loss(model, X, Y, W):
        r = (model(X) - Y) / model.out_sd
        return (W * r ** 2).mean()

    def hydro_residual(self, model, Xc):
        Xc = Xc.detach().clone().requires_grad_(True)
        Y = model(Xc)
        dlnP = torch.autograd.grad(Y[:, 1].sum(), Xc, create_graph=True)[0][:, 0]   # d lnP / dZ_km
        return self.phys.H_ref_km * dlnP + self.phys.T_ref / torch.clamp(Y[:, 0], min=50.0)

    @staticmethod
    def densify(prof, n, rng, z_lo=None, z_hi=None):
        """Collocation coordinates along a profile's geometry (no atmospheric values used)."""
        z = prof["Z_km"].values
        zs = rng.uniform(z.min() if z_lo is None else z_lo, z.max() if z_hi is None else z_hi, n)
        lat = np.interp(zs, z, prof["Latitude"].values)
        sl, cl = np.interp(zs, z, prof["sin_lon"].values), np.interp(zs, z, prof["cos_lon"].values)
        r = np.hypot(sl, cl)
        return np.column_stack([zs, lat, sl / r, cl / r,
                                np.full(n, prof["sin_ls"].iloc[0]), np.full(n, prof["cos_ls"].iloc[0])])

    @torch.no_grad()
    def predict(self, model, df):
        Y = model(self.to_t(df[IN_COLS].values)).cpu().numpy()
        return Y[:, 0], Y[:, 1]

    # ------------------------------------------------------------------ training
    def pretrain(self, tr_df, va_df, family, lam, seed, tag="main", norm=None, verbose=False):
        """Global pre-training on full training profiles; checkpoint chosen by validation data loss."""
        cfg = self.cfg
        ckpt = self.ckpt_dir / f"pretrain_{tag}_{family}_seed{seed}.pt"
        set_seed(seed)
        model = self.new_model(norm)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.eval()
            return model
        rng = np.random.default_rng(seed)
        X, Y, W = self.xyw(tr_df)
        Xv, Yv, Wv = self.xyw(va_df)
        pool = self.to_t(np.vstack([self.densify(g, cfg["COLLOC_POOL_PER_PROFILE"], rng)
                                    for _, g in tr_df.groupby("Profile_ID")]))
        E = cfg["PRETRAIN_EPOCHS"]
        opt = torch.optim.Adam(model.parameters(), lr=cfg["PRETRAIN_LR"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E, eta_min=1e-5)
        best, best_state, hist = np.inf, copy.deepcopy(model.state_dict()), []
        for ep in range(1, E + 1):
            model.train()
            opt.zero_grad()
            ld = self.data_loss(model, X, Y, W)
            loss, lp = ld, torch.tensor(0.0)
            if lam > 0:
                idx = torch.randint(len(pool), (min(cfg["COLLOC_BATCH"], len(pool)),), device=self.device)
                lp = (self.hydro_residual(model, pool[idx]) ** 2).mean()
                loss = loss + lam * lp
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ep % cfg["EVAL_EVERY"] == 0 or ep == E:
                model.eval()
                with torch.no_grad():
                    vl = self.data_loss(model, Xv, Yv, Wv).item()
                hist.append(dict(epoch=ep, train_data=ld.item(), train_phys=float(lp.item()), val_data=vl))
                if vl < best:
                    best, best_state = vl, copy.deepcopy(model.state_dict())
                if verbose and ep % (cfg["EVAL_EVERY"] * 20) == 0:
                    print(f"    [{family} seed {seed}] epoch {ep}/{E}  val {vl:.4g}  phys {float(lp):.3g}")
        model.load_state_dict(best_state)
        model.eval()
        torch.save(best_state, ckpt)
        pd.DataFrame(hist).to_csv(self.log_dir / f"history_{tag}_{family}_seed{seed}.csv", index=False)
        return model

    def finetune(self, base, obs_df, colloc, lam, steps=None, lr=None):
        """Plain per-profile fine-tuning."""
        steps = steps or self.cfg["FINETUNE_STEPS"]
        lr = lr or self.cfg["FINETUNE_LR"]
        model = copy.deepcopy(base)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        X, Y, W = self.xyw(obs_df)
        Xc = self.to_t(colloc)
        for _ in range(steps):
            opt.zero_grad()
            loss = self.data_loss(model, X, Y, W)
            if lam > 0:
                loss = loss + lam * (self.hydro_residual(model, Xc) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        return model

    def finetune_reg(self, base, obs_df, colloc, lam, alpha, gap_lo, gap_hi, use_es=True, steps=None, lr=None):
        """Fine-tuning with L2-SP towards the pre-trained weights and optional early stopping
        on observed levels next to the gap."""
        cfg = self.cfg
        steps = steps or cfg["FINETUNE_STEPS"]
        lr = lr or cfg["FINETUNE_LR"]
        obs_df = obs_df.sort_values("Z_km").reset_index(drop=True)
        z, win = obs_df["Z_km"].values, cfg["ES_WINDOW_KM"]
        hold = np.zeros(len(z), bool)
        for side in (np.where((z < gap_lo) & (z >= gap_lo - win))[0][::-1],
                     np.where((z > gap_hi) & (z <= gap_hi + win))[0]):
            hold[side[::2]] = True
        if not use_es or hold.sum() < 2 or (~hold).sum() < 5:
            hold[:] = False
        model = copy.deepcopy(base)
        model.train()
        theta0 = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        X, Y, W = self.xyw(obs_df[~hold])
        Xc = self.to_t(colloc)
        es = bool(hold.any())
        if es:
            Xe, Ye, We = self.xyw(obs_df[hold])
            with torch.no_grad():
                best = self.data_loss(model, Xe, Ye, We).item()
            best_state = copy.deepcopy(model.state_dict())
        for step in range(1, steps + 1):
            opt.zero_grad()
            loss = self.data_loss(model, X, Y, W)
            if lam > 0:
                loss = loss + lam * (self.hydro_residual(model, Xc) ** 2).mean()
            if alpha > 0:
                loss = loss + alpha * sum(((p - p0) ** 2).sum() for p, p0 in zip(model.parameters(), theta0))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if es and step % cfg["ES_EVAL_EVERY"] == 0:
                with torch.no_grad():
                    v = self.data_loss(model, Xe, Ye, We).item()
                if v < best:
                    best, best_state = v, copy.deepcopy(model.state_dict())
        if es:
            model.load_state_dict(best_state)
        model.eval()
        return model
