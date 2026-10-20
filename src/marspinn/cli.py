"""Command-line interface.

Examples
--------
    marspinn run --config configs/quick.yaml                 # full workflow, fast settings
    marspinn run --config configs/paper.yaml --data csv --csv data/mro_tps_combined.csv
    marspinn run --steps data split pretrain e1 report --set SEEDS=[0,1] GAP_SIZES_KM=[5,10]
    marspinn synth --out data/synthetic_tps.csv --set SYNTH_SEED=3
    marspinn audit --data csv --csv data/mro_tps_combined.csv
    marspinn steps
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline import DEPENDS, STEPS, Pipeline


def _cfg(args):
    cfg = load_config(args.config, args.set)
    if getattr(args, "data", None):
        cfg["DATA_SOURCE"] = args.data
    if getattr(args, "csv", None):
        cfg["DATA_LEVELS"], cfg["DATA_SOURCE"] = args.csv, "csv"
    if getattr(args, "out", None) and args.cmd == "run":
        cfg["OUTPUT_DIR"] = args.out
    if getattr(args, "quick", False):
        cfg.update(GAP_SIZES_KM=[5, 10], GAP_LOCATIONS=["middle", "upper"], SEEDS=[0, 1], MAX_TEST_PROFILES=5,
                   PRETRAIN_EPOCHS=300, FINETUNE_STEPS=100, NOISE_SEEDS=2, HOLDOUT_SEEDS=2, HOLDOUT_MAX_TEST=4,
                   LAMBDAS=[0.0, 1.0, 10.0], N_BOOT=500, TUNE_ALPHAS=[0.0, 1.0], TUNE_LAMBDAS=[1.0],
                   TUNE_CONFIGS=[(10, "middle"), (10, "upper")], CAL_GAP_SIZES_KM=[5, 10])
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="marspinn", description="Physics-informed gap reconstruction of Mars RO profiles")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="YAML config file (defaults are used for missing keys)")
        sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
        sp.add_argument("--data", choices=["synthetic", "csv"], help="data source")
        sp.add_argument("--csv", help="path to mro_tps_combined.csv (implies --data csv)")

    r = sub.add_parser("run", help="run the workflow (all steps or a subset)")
    common(r)
    r.add_argument("--steps", nargs="*", default=None, help=f"subset of {STEPS} (dependencies run automatically)")
    r.add_argument("--out", help="output directory")
    r.add_argument("--quick", action="store_true", help="smoke-test settings (minutes on a CPU)")
    r.add_argument("-q", "--quiet", action="store_true")

    s = sub.add_parser("synth", help="write a synthetic dataset in the MRO TPS CSV format")
    common(s)
    s.add_argument("--out", default="data/synthetic_tps.csv")

    a = sub.add_parser("audit", help="load data, apply QC and print the physics audit")
    common(a)

    sub.add_parser("steps", help="list the workflow steps and their dependencies")

    args = p.parse_args(argv)
    if args.cmd == "steps":
        for st in STEPS:
            print(f"{st:10s} <- {', '.join(DEPENDS[st]) or '-'}")
        return 0
    if args.cmd == "synth":
        from .synthetic import generate_dataset
        cfg = _cfg(args)
        df = generate_dataset(cfg["SYNTH_PROFILES_PER_YEAR"], cfg["SYNTH_SEED"], cfg["SYNTH_NOISE_SCALE"])
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} levels / {df.Profile_ID.nunique()} profiles to {args.out}")
        return 0
    if args.cmd == "audit":
        cfg = _cfg(args)
        cfg["OUTPUT_DIR"] = cfg["OUTPUT_DIR"]
        pl = Pipeline(cfg, verbose=False)
        pl.run(["data"])
        print(json.dumps(pl.R["audit"], indent=2))
        return 0
    cfg = _cfg(args)
    Pipeline(cfg, verbose=not args.quiet).run(args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
