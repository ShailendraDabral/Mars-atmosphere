# Notebooks

| Notebook | Purpose |
|---|---|
| [`01_mro_rss_pinn_reconstruction.ipynb`](01_mro_rss_pinn_reconstruction.ipynb) | Self-contained version of the complete experiment, without outputs. Reads `../data/processed/mro_tps_combined.csv` and writes to `../results/`. Set `QUICK_MODE = True` for a smoke test of a few minutes. |
| [`02_mro_rss_pinn_reconstruction_executed_kaggle.ipynb`](02_mro_rss_pinn_reconstruction_executed_kaggle.ipynb) | Executed record of the reported run (Kaggle GPU, 394.7 min) with all outputs. Paths point to the Kaggle environment. The files in [`../results/`](../results/) come from this run. |

Both notebooks follow the same structure: configuration → data audit → splits → gap scenarios → physics →
methods → pre-training → E1 benchmark with validation tuning → metrics and statistics → conformal calibration →
E2–E4 → figures and tables → auto-generated summary.

The [`marspinn`](../src/marspinn/) package implements the same protocol as a modular, tested pipeline with a
command-line interface; prefer it for new experiments.
