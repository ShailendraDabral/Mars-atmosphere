# Data

The pipeline runs out of the box on **synthetic** profiles (`DATA_SOURCE: synthetic`), generated on the fly.

To use the **real observations**, download the MRO Radio Science temperature–pressure profiles
(`MRO-M-RSS-5-TPS-V1.0`, NASA Planetary Data System, Atmospheres Node), combine the per-profile tables into one CSV
and save it here as `mro_tps_combined.csv`. Required columns (one row per height level):

| Column | Meaning | Unit |
|---|---|---|
| `Profile_ID` | profile identifier | – |
| `Mars_Year` | Mars year | – |
| `Solar_Long` | solar longitude L_s (season) | deg |
| `Radius` | planetocentric radius (used for sorting) | m |
| `Latitude`, `Longitude` | location of the level | deg |
| `Geopotential` | geopotential above the reference geoid | m² s⁻² |
| `Pressure`, `Sigma_P` | pressure and 1σ uncertainty | Pa |
| `Temperature`, `Sigma_T` | temperature and 1σ uncertainty | K |
| `Number_Density`, `Sigma_n` | number density and 1σ uncertainty | m⁻³ |

Then run `marspinn run --config configs/paper.yaml` or upload the file in the web simulator (step 1).
CSV files in this folder are ignored by git so that large data files are not committed by accident.
To see the exact format, write a synthetic file: `marspinn synth --out data/synthetic_tps.csv`.
