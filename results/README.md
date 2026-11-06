# Results

Outputs of the reported full run on the MRO RSS data (Kaggle GPU, 394.7 min), produced by
[`notebooks/02_mro_rss_pinn_reconstruction_executed_kaggle.ipynb`](../notebooks/02_mro_rss_pinn_reconstruction_executed_kaggle.ipynb).
Every number quoted in the main README is traceable to a file here.

> **Version note.** These results come from protocol v1.0, which split profiles at random. The current code and
> `notebooks/01_mro_rss_pinn_reconstruction.ipynb` (v1.1) use an $L_s$-blocked split that keeps close neighbours of
> test profiles out of training, and score pressure against hydrostatic integration. Re-run
> `marspinn run --config configs/paper.yaml` (or notebook 01 with `QUICK_MODE = False`) to regenerate this folder.

| File | Content |
|---|---|
| `results_summary.md` | Auto-generated summary; superiority is claimed only where Holm-corrected tests are significant |
| `tables/summary_overall.csv` | All 12 methods: mean and 95 % bootstrap CI per metric |
| `tables/uncertainty_summary.csv` | Raw and conformal interval coverage, Spearman(σ, error) |
| `figures/fig01`–`fig10` | See below |

| Figure | Content |
|---|---|
| `fig01_data_overview` | Overview of the profile dataset by Mars year |
| `fig02_example_reconstructions` | Example reconstructions across gaps |
| `fig03_error_vs_gap_size` | Error vs gap size by location, all methods |
| `fig04_hydrostatic_residual` | Hydrostatic residual of the reconstructions |
| `fig05_uncertainty_calibration` | Interval calibration before and after conformal rescaling |
| `fig06_paired_improvement` | Paired per-profile differences between methods |
| `fig07_noise_robustness` | E2: added retrieval noise |
| `fig08_lambda_sensitivity` | E3: physics weight λ |
| `fig09_year_holdout` | E4: Mars-year hold-out |
| `fig10_training_curves` | Pre-training curves of the MLP and PINN ensembles |

Figures are 300 dpi PNG files. Re-running the notebook or `marspinn run --config configs/paper.yaml` regenerates all
figures (PNG and PDF) and the remaining tables (summaries by location and gap size, significance tests, tuning,
noise, λ sweep, hold-out, LaTeX tables). New runs write to `results/<name>/`, which is ignored by git.
