# Physics-Informed Reconstruction of Martian Radio-Occultation Profiles

**Shailendra Dabral** ([ORCID 0009-0007-2475-7349](https://orcid.org/0009-0007-2475-7349)) and **Keshav Agrawal**

Reconstructing missing intervals in Mars Reconnaissance Orbiter (MRO) radio-occultation
temperature–pressure profiles with a physics-informed neural network (PINN), benchmarked against
eleven classical, statistical and machine-learning methods.

**Key result.** On 28 unseen profiles and 330 synthetic gap cases, the regularised PINN is the only
method in the top group for temperature, pressure *and* density at once. Which method is best depends
on the gap: interpolation and hydrostatic integration win short interior gaps, learned priors (PINN,
EOF) win large and one-sided gaps, and the hydrostatic constraint is what makes pressure extrapolation
work (upper gaps: 7.0 % → 1.1 % pressure error).

## Results at a glance

Mean over 28 test profiles; 95 % bootstrap confidence intervals in brackets.

| Method | T MAE (K) | P error (%) | n error (%) | Hydrostatic residual |
|---|---|---|---|---|
| **PINN (regularised fine-tuned)** | **3.00 [2.43, 3.64]** | **1.01 [0.85, 1.18]** | **1.92 [1.55, 2.32]** | **0.010** |
| EOF optimal interpolation | 2.78 [2.43, 3.15] | 1.90 [1.48, 2.40] | 2.52 [2.10, 3.04] | 0.165 |
| MLP (regularised fine-tuned) | 2.85 [2.32, 3.45] | 4.44 [3.35, 5.76] | 5.10 [3.89, 6.63] | 0.120 |
| Hydrostatic integration | 3.91 [3.25, 4.66] | 0.86 [0.64, 1.14] | 2.25 [1.78, 2.84] | 0.003 |
| PCHIP (in ln P) | 3.91 [3.25, 4.66] | 2.63 [2.34, 2.88] | 4.23 [3.77, 4.69] | 0.064 |
| Gaussian process | 4.61 [3.93, 5.28] | 4.67 [4.14, 5.13] | 6.39 [5.76, 7.01] | 0.115 |

Full tables (all 12 methods), significance tests and figures are in [`results/`](results/).
Conformal calibration brings the PINN's nominal 95 % intervals from 73 % to 92 % empirical coverage.

## Method in brief

- **Inputs** geopotential height Z, latitude, sin/cos longitude, sin/cos solar longitude.
- **Outputs** temperature T and ln P; number density follows exactly from the ideal-gas law, n = P/(k_B T).
- **Physics** dimensionless hydrostatic residual in geopotential form,
  r = H_ref ∂ln P/∂Z + T_ref/T, with the molecular mass (43.485 g/mol) taken from the data.
- **Training** global pre-training on 128 profiles, then per-profile fine-tuning with an L2-SP penalty
  toward the pre-trained weights; α and λ selected on validation profiles only.
- **Uncertainty** 5-member deep ensemble + split conformal calibration.
- **Statistics** profile-level bootstrap CIs, paired Wilcoxon tests, Holm correction.

## Repository structure

```
├── data/
│   ├── README.md              data source, provenance and citation
│   ├── profile_ids.txt        the 185 profiles used
│   └── processed/             loader output (CSV)
├── notebooks/
│   └── mro_rss_pinn_reconstruction.ipynb   full experiment (outputs cleared)
├── results/                   tables, figures and auto-generated summary from the reported run
├── src/
│   └── mro_tps_loader.py      PDS3 TPS reader (MRO and MGS formats)
└── docs/                      technical report
```

## Reproducing the results

```bash
git clone https://github.com/<your-username>/martian-ro-pinn.git
cd martian-ro-pinn
pip install -r requirements.txt
cd notebooks
jupyter notebook mro_rss_pinn_reconstruction.ipynb
```

1. Run once with `QUICK_MODE=True` in the configuration cell (about 5 minutes) as a smoke test.
2. Set `QUICK_MODE=False` for the full experiment. The reported run took 6.6 hours on a Kaggle GPU.
   Progress is cached, so an interrupted run resumes where it stopped.
3. Outputs are written to `results/`.

To rebuild the processed CSVs from raw PDS files, see [`data/README.md`](data/README.md).

## Scope and limitations

All profiles are MRO pre-dawn (03–05 LTST) ingress occultations at Ls 21–142° (northern spring–summer),
Mars years 29–31. The reference profiles are retrievals built with hydrostatic balance and the ideal-gas
law, so physical-consistency results measure agreement with the retrieval physics. Holding out a whole
observing campaign at an unseen season and latitude band (Mars year 29) degrades every method to 6.5–7.7 K.

## Citation

If you use this code, please cite it using the metadata in [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button). Please also cite the data source listed in
[`data/README.md`](data/README.md).

## License

Code: [MIT](LICENSE). Text, figures and tables: CC BY 4.0.
