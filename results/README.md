# Results

Outputs of the reported full run (Kaggle GPU, 394.7 min; notebook in `../release_assets/` of the
GitHub Release). Every number in the paper should be traceable to a file here.

| File | Content |
|---|---|
| `results_summary.md` | Auto-generated summary; superiority is claimed only where Holm-corrected tests are significant |
| `tables/summary_overall.csv` | All 12 methods: mean and 95 % bootstrap CI per metric |
| `tables/uncertainty_summary.csv` | Raw and conformal coverage, Spearman(σ, error) |
| `figures/fig01`–`fig10` | Data overview, examples, error vs gap size, physics residual, calibration, paired improvement, noise, λ sweep, campaign hold-out, training curves |

Figures are the 300 dpi PNG files saved by the notebook (PDF versions can be added beside them).
Re-running the notebook regenerates all figures and the remaining tables (summaries by location and gap
size, significance tests, tuning, noise, λ sweep, hold-out, LaTeX tables).
