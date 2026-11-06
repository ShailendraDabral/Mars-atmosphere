# Notebooks

| Notebook | Purpose |
|---|---|
| [`01_mro_rss_pinn_reconstruction.ipynb`](01_mro_rss_pinn_reconstruction.ipynb) | **Current protocol (v1.1)**, self-contained and without outputs: $L_s$-blocked split with a near-neighbour buffer, Mondrian conformal calibration per gap location and size, pressure compared against hydrostatic integration, density not scored separately, E4 re-tuned without the held-out year, configuration-keyed caches. Matches the `marspinn` package. Reads `../data/processed/mro_tps_combined.csv` and writes to `../results/`. Set `QUICK_MODE = True` for a smoke test. |
| [`02_mro_rss_pinn_reconstruction_executed_kaggle.ipynb`](02_mro_rss_pinn_reconstruction_executed_kaggle.ipynb) | Executed record of the **v1.0** run (random profile split; Kaggle GPU, 394.7 min) with all outputs. Paths point to the Kaggle environment. The files in [`../results/`](../results/) come from this run. |

Both notebooks follow the same structure: configuration → data audit → splits → gap scenarios → physics →
methods → pre-training → E1 benchmark with validation tuning → metrics and statistics → conformal calibration →
E2–E4 → figures and tables → auto-generated summary.

The [`marspinn`](../src/marspinn/) package implements the same protocol as a modular, tested pipeline with a
command-line interface; prefer it for new experiments.
