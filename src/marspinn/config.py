"""Configuration handling.

Every setting of the experiment lives in one flat dictionary. Defaults reproduce the
full experiment of the paper; YAML files in ``configs/`` and command-line overrides
(``--set KEY=VALUE``) change individual entries.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import yaml

DEFAULTS: dict = dict(
    # ---- data ----
    DATA_SOURCE="synthetic",          # "synthetic" or "csv"
    DATA_LEVELS="data/mro_tps_combined.csv",
    OUTPUT_DIR="results",
    SYNTH_PROFILES_PER_YEAR=[65, 46, 73],   # MY29, MY30, MY31 (mirrors the real archive)
    SYNTH_SEED=7,
    SYNTH_NOISE_SCALE=1.0,            # multiplies the simulated retrieval error

    # ---- quality control ----
    SIGMA_T_MAX=5.0,
    MIN_LEVELS=40,

    # ---- splits ----
    SPLIT_FRACS=[0.70, 0.15, 0.15],
    SPLIT_SEED=2024,

    # ---- gap scenarios ----
    GAP_SIZES_KM=[2, 5, 10, 20],
    GAP_LOCATIONS=["lower", "middle", "upper"],
    MIN_OBSERVED_KM=5.0,
    MAX_TEST_PROFILES=None,

    # ---- neural networks ----
    SEEDS=[0, 1, 2, 3, 4],
    HIDDEN=64, DEPTH=4,
    PRETRAIN_EPOCHS=3000, PRETRAIN_LR=2e-3, EVAL_EVERY=25,
    COLLOC_POOL_PER_PROFILE=200, COLLOC_BATCH=2048,
    FINETUNE_STEPS=400, FINETUNE_LR=5e-4, N_COLLOC_GAP=64,
    LAMBDA_PHYS=1.0,
    ES_WINDOW_KM=3.0, ES_EVAL_EVERY=25,
    TUNE_CONFIGS=[[10, "middle"], [20, "lower"], [20, "upper"]],
    TUNE_ALPHAS=[0.0, 0.1, 1.0], TUNE_LAMBDAS=[0.1, 1.0], TUNE_ES=[True, False],
    HP_OVERRIDE=None,                 # e.g. {"MLP": {...}, "PINN": {...}} to skip tuning
    CAL_GAP_SIZES_KM=[2, 5, 10, 20],
    EOF_DZ_KM=0.5, EOF_VAR_FRAC=0.99, EOF_MAX_K=15, EOF_MIN_COVER=0.2,
    SIGMA_WEIGHTING=True, SIGMA_FLOOR_T=0.5, SIGMA_FLOOR_LNP=0.003,

    # ---- extra experiments ----
    RUN_NOISE=True, NOISE_MULTIPLIERS=[0, 2, 5], NOISE_CASES=[[10, "middle"]], NOISE_SEEDS=3,
    RUN_LAMBDA_SWEEP=True, LAMBDAS=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
    LAMBDA_CASES=[[10, "middle"], [10, "upper"]],
    RUN_YEAR_HOLDOUT=True, HOLDOUT_CASES=[[10, "middle"], [10, "upper"]], HOLDOUT_SEEDS=3,
    HOLDOUT_MAX_TEST=20,

    # ---- statistics ----
    N_BOOT=5000, ALPHA=0.05,

    # ---- runtime ----
    DEVICE="auto",                    # "auto", "cpu" or "cuda"
)


def _parse_value(text: str):
    """Interpret a command-line value: Python/YAML literal if possible, else a string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        return text


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Defaults <- YAML file <- ``KEY=VALUE`` overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(cfg)
        if unknown:
            raise KeyError(f"Unknown config keys in {path}: {sorted(unknown)}")
        cfg.update(user)
    for item in overrides or []:
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in cfg:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = _parse_value(val.strip())
    # normalise list-of-lists into tuples where code expects (size, location)
    for k in ("TUNE_CONFIGS", "NOISE_CASES", "LAMBDA_CASES", "HOLDOUT_CASES"):
        cfg[k] = [tuple(x) for x in cfg[k]]
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    def plain(v):
        if isinstance(v, tuple):
            return [plain(x) for x in v]
        if isinstance(v, list):
            return [plain(x) for x in v]
        return v
    with open(path, "w") as f:
        yaml.safe_dump({k: plain(v) for k, v in cfg.items()}, f, sort_keys=False)
