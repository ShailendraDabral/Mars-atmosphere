# Data

## Source

MRO Radio Science Subsystem (RSS) temperature–pressure profiles, data set **`MRO-M-RSS-5-TPS-V1.0`**
(D. P. Hinson, NASA Planetary Data System, Atmospheres Node). Each PDS3 `.TPS` product is one radio-occultation
retrieval of temperature, pressure and number density (with 1σ uncertainties) versus radius and geopotential.

## Contents

| File | Description |
|---|---|
| `processed/mro_tps_combined.csv` | One row per vertical level (16 686 levels), with the profile metadata repeated on each row. Input to the pipeline. |
| `processed/mro_tps_profiles.csv` | One row per profile (185 profiles): PDS header fields plus quality and physics diagnostics (level count, vertical span, ideal-gas deviation, implied molecular mass, effective gravity). |
| `profile_ids.txt` | The 185 profile identifiers used in the study. |

### Coverage

| Mars year | Profiles | PDS month folders |
|---|---:|---|
| 29 | 65 | 2008-08, 2008-09, 2008-10 |
| 30 | 47 | 2009-12, 2010-01 |
| 31 | 73 | 2011-12, 2012-01, 2012-02 |

All profiles are pre-dawn ingress occultations (≈ 03–05 LTST), *L*ₛ ≈ 21°–142°, latitudes ≈ 13° S–69° N.

## Columns of `mro_tps_combined.csv`

The columns required by the pipeline:

| Column | Meaning | Unit |
|---|---|---|
| `Profile_ID` | profile identifier (PDS product name) | – |
| `Mars_Year` | Mars year | – |
| `Solar_Long` | solar longitude *L*ₛ (season) | deg |
| `Radius` | planetocentric radius (used for sorting) | m |
| `Latitude`, `Longitude` | location of the level | deg |
| `Geopotential` | geopotential above the reference geoid | m² s⁻² |
| `Pressure`, `Sigma_P` | pressure and 1σ uncertainty | Pa |
| `Temperature`, `Sigma_T` | temperature and 1σ uncertainty | K |
| `Number_Density`, `Sigma_n` | number density and 1σ uncertainty | m⁻³ |

Additional metadata columns (`LTST_hr`, `SZA_deg`, `occultation`, `orbit`, `start_time`, …) are carried over from the
PDS header for reference.

## Rebuilding from the PDS products

Download the `.TPS` products for the month folders listed above from the PDS Atmospheres Node, then run:

```bash
python scripts/build_mro_dataset.py /path/to/200808 /path/to/200809 ... --out data/processed
```

The builder also reads MGS (`MGS-M-RSS-5-TPS-V1.0`) products and puts both missions on a common geopotential
reference. Its current version adds a `mission` column, which the pipeline ignores.

## Synthetic data

For development without the real data, write a synthetic archive with the same columns and coverage:

```bash
marspinn synth --out data/synthetic_tps.csv
```

## Terms of use

The PDS data are publicly available. Please cite the data set (see the main [README](../README.md#citation)) when
using these files.
