# Data

## Source

| Item | Value |
|---|---|
| Archive | NASA Planetary Data System (PDS), Atmospheres Node |
| Volume | `mrors_2001` (MRO RS TPS archive) |
| Dataset ID | `MRO-M-RSS-5-TPS-V1.0` |
| Instrument | Mars Reconnaissance Orbiter, Radio Science Subsystem |
| Producer | D. P. Hinson, SETI Institute |
| Retrieval references | Hinson et al., J. Geophys. Res. 104, 26997–27012 (1999); J. Geophys. Res. 109, E12002 (2004) |

Month folders used: `200808`, `200809`, `200810`, `200912`, `201001`, `201112`, `201201`, `201202`
(185 profiles; IDs in [`profile_ids.txt`](profile_ids.txt)). Downloaded: October 2026.

## Processed files

Produced by [`../src/mro_tps_loader.py`](../src/mro_tps_loader.py) without changing any values; only
reformatting and metadata columns were added. Local file paths were reduced to month-folder and file
names. Quality control (trimming levels with σ_T > 5 K, minimum 40 levels) is applied in the notebook,
not here.

| File | Content |
|---|---|
| `processed/mro_tps_combined.csv` | One row per vertical level (16,686 rows): radius, latitude, longitude, geopotential, P, σ_P, T, σ_T, n, σ_n, plus profile metadata |
| `processed/mro_tps_profiles.csv` | One row per profile (185 rows): header fields and audit values (implied molecular mass, effective gravity, ideal-gas check) |

## Rebuilding from raw PDS files

1. Download the month folders listed above from the PDS Atmospheres Node, volume `mrors_2001`.
2. Edit `MONTH_FOLDERS` at the top of `src/mro_tps_loader.py`, or pass the folders on the command line:

```bash
python src/mro_tps_loader.py /path/to/200808 /path/to/200809 ... --out data/processed
```

## Credit

The radio-occultation profiles are NASA PDS products. Please cite the dataset and the retrieval
references above when using these data.
