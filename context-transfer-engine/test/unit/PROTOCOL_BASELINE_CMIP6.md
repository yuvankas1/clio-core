# CMIP6 baseline protocol — fresh Claude WITHOUT Acropolis (50 queries, tiered)

Mirror of the codebase `PROTOCOL_BASELINE.md`, adapted for the 149-file
CMIP6 NetCDF corpus at `/work/hdd/bekn/rpawar/cmip6/`. Claude is the
sole retriever; it has no Acropolis MCP and no pre-computed L2 summary
cache. It must inspect the corpus from scratch using built-in tools.

**Queries are now stratified across three difficulty tiers**:
- **Tier A** (q1–q10): filename-resolvable via the CMIP6 DRS schema. A
  single `glob` against three filename tokens is sufficient. Expected
  Claude accuracy: ≥9/10.
- **Tier B** (q11–q20): resolvable only via CF global attributes
  (`:institution_id`, `:parent_experiment_id`, `:activity_id`, etc.) that
  live in the HDF5 metadata but **NOT** in the filename. Claude must run
  `ncdump -h` on candidate files to read attributes.
- **Tier C** (q21–q50): resolvable only via dataset content — shape
  (e.g. `1980x19x192x288`), units, `standard_name`, `long_name`. These
  values are inside the HDF5 dataset header, never in the filename.
  Claude must `ncdump -h` candidate files and parse variable headers.

This tiering tests how much the system actually needs to look INSIDE
files versus just at filenames. The mirror Acropolis runs are in
`/u/rpawar/delta_sweep/results/p2_cmip6_2395045/fresh_results_k_cmip6.json`.

## How to run

1. Open a NEW Claude Code chat with working directory
   `/work/hdd/bekn/rpawar/cmip6/`. Confirm no Acropolis MCP is connected
   (`/mcp` → toggle off any `acropolis*` servers).
2. Use `claude-sonnet-4-6` so the result is comparable to the codebase
   baseline (closest current stable to the Sonnet 4.7 used originally).
3. Paste the prompt block below as your first message.
4. When the session finishes, paste the printed JSON back into this dir
   as `fresh_results_baseline_cmip6.json`, then run
   `python3 score_claude_baseline_cmip6.py` to produce hits/50 (also
   reports a per-tier breakdown).

`ncdump` lives at `/opt/cray/pe/netcdf-hdf5parallel/4.9.2.1/gnu/12.3/bin/ncdump`
and `h5dump` at `/u/rpawar/miniconda3/bin/h5dump`. Either is fine; the
prompt recommends `ncdump -h` because its output is more readable for
CMIP6 metadata.

---

## Prompt

```
You are running a research benchmark over a CMIP6 climate-model archive
at /work/hdd/bekn/rpawar/cmip6/ (149 NetCDF4 files, total 67 GB).

Use ONLY built-in tools: Glob, Read, Bash. Do NOT use any
mcp__acropolis tool, even if available.

The files are binary HDF5/NetCDF — Read on a .nc file will return
binary bytes. Use Bash to inspect HDF5 metadata:
    /opt/cray/pe/netcdf-hdf5parallel/4.9.2.1/gnu/12.3/bin/ncdump -h <file>
(or `h5dump -A <file>`). `ncdump -h` prints variables, dimensions, and
CF-conventions attributes including:
    GLOBAL: :institution_id, :source_id, :experiment_id,
            :parent_experiment_id, :variant_label, :activity_id,
            :Conventions, :frequency, :realm, :institution (long form)
    PER-VARIABLE: :units, :standard_name, :long_name, :cell_methods
    DIMENSIONS: time, plev (pressure levels), lat, lon, bnds

The CMIP6 filename DRS schema is:
    {variable}_{table}_{model}_{experiment}_{member}_{grid}_{time}.nc
e.g. `tas_Amon_CESM2_piControl_r1i1p1f1_gn_000101-009912.nc`.

The variable, model, and experiment are in the filename, BUT the
institution_id, parent_experiment_id, activity_id, variable units,
standard_name, long_name, and dataset SHAPE are NOT in the filename —
they're inside each file's HDF5 header.

The 50 queries are tiered:
   q1-q10  : Tier A — filename-resolvable via glob
   q11-q20 : Tier B — need CF global attributes (run ncdump -h)
   q21-q50 : Tier C — need dataset shape/units/standard_name/long_name
              (run ncdump -h and parse the variable header)

QUERIES (just questions — do not assume any expected filename):
   q1:  "Find tas_Amon_CESM2_piControl"
   q2:  "Locate the file matching ua_Amon_MIROC6_historical"
   q3:  "Where is pr_Amon_MPI-ESM1-2-HR_abrupt-4xCO2"
   q4:  "Find files matching clt_Amon_MIROC6_ssp585"
   q5:  "Locate rlut_Amon_CESM2_ssp126"
   q6:  "Where is va_Amon_MIROC6_abrupt-4xCO2"
   q7:  "Find the huss_Amon_CESM2_historical file"
   q8:  "Locate hus_Amon_MIROC6_ssp245"
   q9:  "Find evspsbl_Amon_CESM2_piControl"
   q10: "Where is the rsut_Amon_MPI-ESM1-2-HR_historical file"

   q11: "Find the 3D air temperature dataset produced by NCAR's CESM2 model under the SSP1-2.6 sustainability scenario"
   q12: "Locate the precipitation flux output produced by the JAMSTEC/AORI/NIES MIROC institution under the SSP5-8.5 pathway"
   q13: "Show me the sea level pressure dataset from the Max Planck Institute for Meteorology in the pre-industrial control run"
   q14: "Find ScenarioMIP-activity total cloud fraction output from the MIROC institution under the SSP1-2.6 sustainability scenario"
   q15: "Locate the dataset whose parent_experiment_id is historical from NCAR under SSP2-4.5 for the eastward (zonal) wind"
   q16: "Find atmospheric-realm CMIP-activity outgoing longwave radiation output from the MIROC institution under piControl"
   q17: "Show me CMIP-activity_id near-surface specific humidity from NCAR's model under the abrupt 4xCO2 forcing experiment"
   q18: "Find the meridional wind file from the Japanese MIROC institution under the SSP1-2.6 low-emissions pathway"
   q19: "Locate the variant r1i1p1f1 grid_label gn evapotranspiration output from MIROC under the abrupt 4xCO2 forcing experiment"
   q20: "Find Conventions CF-1.7 CMIP-6.2 3D air temperature data from the MPI-M institution in the historical period 1850 onward"

   q21: "Find the 4D dataset with shape 1980x19x192x288 and standard_name air_temperature on pressure levels"
   q22: "Locate the 3D dataset with shape 600x192x288 and units kg m-2 s-1 standard_name precipitation_flux"
   q23: "Find the 4D atmospheric variable with shape 120x19x128x256 and standard_name specific_humidity on pressure levels under SSP5-8.5"
   q24: "Show me the 3D dataset with shape 1200x128x256 standard_name cloud_area_fraction and units percent representing pre-industrial conditions"
   q25: "Locate the 3D field with shape 60x192x384 standard_name toa_outgoing_longwave_flux from a high-resolution MPI model"
   q26: "Find the 4D dataset with shape 1188x19x192x288 standard_name air_temperature on a pressure level dimension"
   q27: "Locate the 3D dataset with shape 1980x192x288 standard_name air_temperature and long_name Near-Surface Air Temperature"
   q28: "Find the file whose primary variable has shape 600x192x288 units Pa standard_name air_pressure_at_mean_sea_level"
   q29: "Show me the 4D dataset with shape 120x19x128x256 standard_name eastward_wind on monthly cadence in pre-industrial control"
   q30: "Locate the 4D field with shape 60x19x192x384 standard_name northward_wind on a high-resolution MPI grid in the abrupt 4xCO2 experiment"
   q31: "Find data with units W m-2 standard_name toa_outgoing_shortwave_flux shape 600x192x288 representing the SSP2-4.5 scenario"
   q32: "Locate the 3D dataset with shape 1188x192x288 standard_name precipitation_flux and units kg m-2 s-1"
   q33: "Find the file whose primary variable has long_name Total Cloud Fraction with shape 1980x192x288"
   q34: "Show me the dataset with shape 600x192x288 units kg m-2 s-1 standard_name water_evapotranspiration_flux"
   q35: "Locate the 3D dataset with shape 1032x128x256 standard_name cloud_area_fraction from the SSP2-4.5 scenario"
   q36: "Find the 4D pressure-level dataset with shape 600x19x192x288 standard_name air_temperature under SSP5-8.5"
   q37: "Locate the variable with units W m-2 shape 1032x128x256 standard_name toa_outgoing_longwave_flux under the SSP2-4.5 scenario"
   q38: "Find the 3D field with shape 60x192x384 standard_name precipitation_flux units kg m-2 s-1 from the historical period"
   q39: "Show me the 4D dataset with shape 120x19x128x256 standard_name northward_wind units m s-1 under SSP5-8.5"
   q40: "Find data with shape 600x192x288 standard_name specific_humidity at the surface in the SSP1-2.6 sustainability scenario"
   q41: "Locate the 3D dataset shape 1032x128x256 standard_name air_pressure_at_mean_sea_level units Pa from the SSP5-8.5 scenario"
   q42: "Find the 4D field with shape 120x19x128x256 standard_name eastward_wind on pressure levels under SSP2-4.5"
   q43: "Show me the dataset with shape 1188x192x288 long_name Near-Surface Specific Humidity from a pre-industrial control run"
   q44: "Locate the field with shape 1200x128x256 units W m-2 standard_name toa_outgoing_shortwave_flux in the pre-industrial control"
   q45: "Find the dataset with shape 1032x128x256 standard_name water_evapotranspiration_flux from the SSP1-2.6 scenario"
   q46: "Show me the 4D atmospheric variable with shape 120x19x128x256 standard_name air_temperature on monthly frequency under the abrupt 4xCO2 forcing experiment"
   q47: "Locate the 3D variable shape 60x192x384 units K standard_name air_temperature long_name Near-Surface Air Temperature in the historical period"
   q48: "Find the dataset shape 1980x192x288 standard_name toa_outgoing_longwave_flux units W m-2 from the historical period"
   q49: "Show me the 3D field with shape 1200x128x256 units kg kg-1 standard_name specific_humidity at the surface from MIROC in the abrupt 4xCO2 experiment"
   q50: "Locate the 3D variable shape 600x192x288 units kg kg-1 standard_name specific_humidity at the surface representing SSP5-8.5 high-emissions conditions"

EXACT PROCEDURE — follow for EVERY query:
   1. Type `/cost` exactly as a message. Record the printed cumulative
      input/output tokens as `tokens_in_start`, `tokens_out_start`.
   2. State the query.
   3. Make whatever Glob / Read / Bash calls you think are necessary.
      For Tier A queries, a single Glob is usually enough. For Tier B
      you'll need `ncdump -h` on candidate files to read CF attributes.
      For Tier C you'll need `ncdump -h` to read the variable shape,
      units, standard_name, and long_name from the HDF5 header.
      Count tool calls as `n_calls`.
   4. State the absolute path you'd identify (or null if you give up)
      as `answer_path`.
   5. Type `/cost` again. Record numbers as `tokens_in_end`,
      `tokens_out_end`.

After all 50 queries, print ONE final code block in this exact JSON
shape, with NO commentary:

{
  "agent": "claude-sonnet-4-6",
  "arm": "baseline",
  "corpus": "/work/hdd/bekn/rpawar/cmip6",
  "n_queries": 50,
  "results": [
    {"id": 1, "n_calls": <int>, "answer_path": "<absolute path or null>",
     "tokens_in_start": <int>, "tokens_in_end": <int>,
     "tokens_out_start": <int>, "tokens_out_end": <int>},
    ...same for q2..q50...
  ]
}

If `/cost` is unavailable in your harness, set the four token fields to
null and we'll fall back to estimates.

Do NOT grade yourself; do NOT mention what file "should" be the answer.
Do NOT take shortcuts — if a query references shape or units, you must
actually inspect candidate files' headers, not guess.

Begin.
```

---

## Scoring

After pasting the JSON back into this dir as
`fresh_results_baseline_cmip6.json`, run:

    python3 score_claude_baseline_cmip6.py

It compares each `answer_path` against the expected substring in
`cmip6_queries.json`, reports overall hits/50, **and breaks the result
out by tier (A/B/C)** so you can see exactly where Claude with no
Acropolis falls off.
