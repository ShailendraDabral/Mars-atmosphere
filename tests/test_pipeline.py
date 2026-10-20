"""Tests for the Python pipeline: physics, synthetic data, baselines, statistics and a tiny end-to-end run."""
import numpy as np
import pandas as pd
import pytest
import torch

from marspinn.baselines import eof_predict, fit_eof_pair, gp_predict, hydrostatic_integration, interp_extrap
from marspinn.config import load_config
from marspinn.data import make_gap, physics_audit, preprocess, quality_control, stratified_profile_split
from marspinn.metrics import paired_tests
from marspinn.nn import NNToolkit
from marspinn.physics import Physics, hydro_residual_np, integrate_lnP
from marspinn.pipeline import Pipeline
from marspinn.synthetic import generate_dataset


@pytest.fixture(scope="module")
def data():
    phys = Physics()
    cfg = load_config()
    lv = preprocess(generate_dataset((12, 10, 12), seed=3), phys)
    lv, meta, qc = quality_control(lv, cfg)
    audit = physics_audit(lv, phys)
    return cfg, phys, lv, meta, qc, audit


def test_synthetic_obeys_physics(data):
    _, phys, lv, _, _, audit = data
    assert abs(audit["mean_molecular_mass_gmol"] - 43.49) < 0.05
    assert audit["ideal_gas_max_rel_dev"] < 1e-9
    assert audit["reference_hydro_residual_RMS_median"] < 0.01
    assert 9.5 < audit["H_ref_km"] < 11


def test_quality_control_trims_top(data):
    cfg, _, lv, _, qc, _ = data
    assert lv["Sigma_T"].max() <= cfg["SIGMA_T_MAX"]
    assert qc["median_top_trim_km"] > 0
    assert (lv.groupby("Profile_ID").size() >= cfg["MIN_LEVELS"]).all()


def test_split_has_no_leakage(data):
    _, _, _, meta, _, _ = data
    tr, va, te = stratified_profile_split(meta, (0.7, 0.15, 0.15), 1)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(meta)


def test_gap_definitions():
    z = np.linspace(0, 30, 61)
    for loc in ("lower", "middle", "upper"):
        g = make_gap(z, 10, loc)
        assert g is not None and abs((z[g].max() - z[g].min()) - 10) <= 0.5 + 1e-9
    assert make_gap(z, 28, "middle") is None          # less than 5 km would remain observed


def test_hydrostatic_integration_is_exact_for_hydrostatic_profile():
    phys = Physics()
    z = np.linspace(0, 30, 121)
    T = 210 - 1.5 * z
    lnP = integrate_lnP(z, T, np.log(600), phys)
    gap = make_gap(z, 8, "middle")
    Tg, Pg = hydrostatic_integration(z[~gap], T[~gap], lnP[~gap], z[gap], phys)
    assert np.max(np.abs(Pg - lnP[gap])) < 1e-4
    assert np.max(np.abs(Tg - T[gap])) < 1e-6        # PCHIP reproduces a straight line
    assert np.sqrt(np.mean(hydro_residual_np(z, T, lnP, phys) ** 2)) < 1e-3


def test_interp_and_gp_shapes():
    z = np.linspace(0, 20, 40)
    y = np.sin(z / 3)
    zn = np.array([-2.0, 5.1, 25.0])
    assert interp_extrap(z, y, zn, "pchip").shape == (3,)
    mu, sd = gp_predict(z, y, zn)
    assert mu.shape == sd.shape == (3,) and np.all(sd > 0)


def test_eof_prior_reconstructs_training_profile(data):
    cfg, _, lv, _, _, _ = data
    E = fit_eof_pair(lv, cfg)
    g = lv[lv.Profile_ID == lv.Profile_ID.iloc[0]]
    mu, sd = eof_predict(E["T"], g.Z_km.values, g.Temperature.values, g.Sigma_T.values, g.Z_km.values)
    assert np.mean(np.abs(mu - g.Temperature.values)) < 3.0 and np.all(sd > 0)


def test_physics_gradient_matches_finite_difference(data, tmp_path):
    cfg, phys, lv, _, _, _ = data
    nnt = NNToolkit(dict(cfg, DEVICE="cpu"), phys, lv, tmp_path / "c", tmp_path / "l")
    torch.manual_seed(0)
    model = nnt.new_model().double()
    prof = lv[lv.Profile_ID == lv.Profile_ID.iloc[0]]
    x = torch.tensor(nnt.densify(prof, 5, np.random.default_rng(0)), dtype=torch.float64)
    r = nnt.hydro_residual(model, x).detach()
    h = 1e-5
    xp, xm = x.clone(), x.clone()
    xp[:, 0] += h
    xm[:, 0] -= h
    dlnP = (model(xp)[:, 1] - model(xm)[:, 1]) / (2 * h)
    r_fd = phys.H_ref_km * dlnP + phys.T_ref / model(x)[:, 0]
    assert torch.allclose(r, r_fd.detach(), atol=1e-5)


def test_paired_tests_detect_clear_difference():
    rows = []
    for p in range(10):
        rows.append(dict(method="PINN (reg. fine-tuned)", Profile_ID=p, T_MAE=1.0 + 0.01 * p))
        rows.append(dict(method="Linear", Profile_ID=p, T_MAE=2.0 + 0.02 * p))
    T = paired_tests(pd.DataFrame(rows), "T_MAE", n_boot=200)
    assert T.loc[0, "significant"] and T.loc[0, "mean_diff"] > 0


def test_tiny_end_to_end_run(tmp_path):
    cfg = load_config("configs/quick.yaml", [
        f"OUTPUT_DIR={tmp_path}", "SYNTH_PROFILES_PER_YEAR=[10,10,10]", "MAX_TEST_PROFILES=3", "SEEDS=[0,1]",
        "PRETRAIN_EPOCHS=40", "COLLOC_POOL_PER_PROFILE=20", "FINETUNE_STEPS=10", "GAP_SIZES_KM=[5]",
        "N_BOOT=100", "LAMBDAS=[0.0,1.0]", "HOLDOUT_MAX_TEST=3", "DEVICE=cpu",
        "HP_OVERRIDE={'MLP': {'lam': 0.0, 'alpha': 1.0, 'es': False}, 'PINN': {'lam': 0.1, 'alpha': 1.0, 'es': False}}"])
    pl = Pipeline(cfg, verbose=False)
    R = pl.run()
    assert {"S_OVERALL", "SIG", "UQ", "S_NOISE", "S_LAM", "S_HOLD"} <= set(R)
    assert (tmp_path / "results_summary.md").exists()
    assert (tmp_path / "figures" / "fig03_error_vs_gap_size.png").exists()
    assert np.isfinite(R["S_OVERALL"]["T_MAE"]).all()
