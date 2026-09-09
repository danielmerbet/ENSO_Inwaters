# ENSO_Inwaters

**Do super El Niño events perturb the world's rivers, lakes and
groundwater in a way that a scaled-up ordinary El Niño does not?**

A complete, reproducible workflow that answers that question from the
ISIMIP3a multi-model ensemble, from raw download to rendered
manuscript. Every figure, table and number in the paper is produced by a
numbered script; nothing is typed by hand.

---

## The question

Four El Niño events since 1901 reached an ONI of 2 °C or more —
1972/73, 1982/83, 1997/98 and 2015/16. Their atmospheric teleconnections
are well mapped. What reaches rivers, lakes and aquifers is not
precipitation, though: it is precipitation filtered through soil
moisture, routing and storage, each nonlinear and each with memory. So:

1. **Where, and how strongly**, do super El Niños perturb inland waters?
2. **Is the response nonlinear** in event amplitude, or is a super event
   just a strong event times 1.8?
3. **How long does it take to arrive, and how long does it last** — from
   runoff, through discharge and recharge, to groundwater storage?

`docs/analysis_plan.md` states the falsifiable predictions and, in
advance, what a null result would look like.

## Quick start

```bash
# 1. environment
mamba env create -f environment.yml && conda activate enso-inwaters
#    (or: pip install -r requirements.txt)

# 2. check the environment, the disk and the data hosts
python workflow/00_check_environment.py

# 3. prove the code works, offline, in a few minutes
bash tests/test_workflow_smoke.sh

# 4. the real thing (needs ~600 GB and about two days, mostly download)
bash workflow/run_all.sh
```

Step 3 generates a synthetic dataset with a **known** teleconnection,
runs the entire pipeline on it, and asserts that the analysis recovers
what was put in — the right super events, the right dry/wet pattern, the
right response lags. Run it before and after any change to the code.

## The workflow

Scripts run in numeric order. Each is idempotent: existing outputs are
skipped unless `--overwrite` is given. Each writes its own log to
`logs/`.

| # | script | what it does |
|---|--------|--------------|
| 00 | `00_check_environment.py` | packages, disk, reachability of the data hosts |
| 01 | `01_fetch_enso_indices.py` | Niño3.4/3/4 SST from NOAA PSL and CPC |
| 02 | `02_define_enso_events.py` | ONI with sliding base periods → the event catalogue |
| 03 | `03_fetch_isimip3a.py` | query the ISIMIP API, build a manifest, download |
| 04 | `04_fetch_auxiliary_data.py` | basins, lakes, population, evaluation datasets |
| 05 | `05_preprocess_harmonize.py` | common grid, calendar, units, land mask |
| 06 | `06_anomalies_and_indices.py` | standardised anomalies + SRI/SSI/SGI |
| 07 | `07_event_composites.py` | superposed-epoch composites by event class |
| 08 | `08_significance_and_agreement.py` | Monte-Carlo null, FDR, model agreement |
| 09 | `09_nonlinearity.py` | **Q2**: excess over linear scaling + hinge regression |
| 10 | `10_lag_cascade.py` | **Q3**: peak lags, memory, recovery times |
| 11 | `11_basin_aggregation.py` | major river basins |
| 12 | `12_lake_analysis.py` | lake surface temperature, ice, stratification |
| 13 | `13_extremes.py` | drought and flood months, odds ratios |
| 14 | `14_scenario_contrasts.py` | obsclim−counterclim, histsoc−nosoc |
| 15 | `15_exposure.py` | area and population in the response regions |
| 16 | `16_validation.py` | GRDC, GRACE, satellite LSWT evaluation |
| 17 | `17_uncertainty.py` | variance partition, leave-one-out |
| 20–26 | `2*_figure*.py` | the six main figures and the supplements |
| 30 | `30_assemble_tables.py` | manuscript tables + a provenance index |
| 31 | `31_paper_numbers.py` | every number the text cites → JSON |
| 40 | `40_render_manuscript.py` | fill the template's placeholders |

```bash
bash workflow/run_all.sh --from 07        # resume
bash workflow/run_all.sh --only 08,09     # just these
bash workflow/run_all.sh --skip-download  # data already staged
python workflow/09_nonlinearity.py --help # every step has its own options
```

## Layout

```
config/config.yaml              every parameter; nothing is hard-coded
config/config_synthetic.yaml    the smoke-test variant (inherits via `_base`)
src/enso_inwaters/              the reusable science code
  enso.py                       ONI, event detection, superposed epochs
  climatology.py                anomalies, moving climatology, SRI/SSI/SGI
  stats.py                      Monte-Carlo tests, FDR, ANOVA
  regions.py                    areas, masks, basin and lake aggregation
  isimip_io.py                  ISIMIP naming, units, netCDF conventions
  plotting.py                   journal-ready figure style
  download.py                   ISIMIP API and HTTP retrieval
workflow/                       the numbered driver scripts
tests/                          unit tests + the end-to-end smoke test
docs/                           analysis plan, methods, data sources, compute
paper/                          manuscript template and rendered draft
data/  results/  logs/          generated; not versioned
```

## Method, in one page

**Events.** ONI = 3-month running mean of ERSSTv5 Niño3.4 anomalies
against *centred 30-year base periods updated every 5 years* — the CPC
convention, extended to 1901 so that a century of warming does not make
recent events look artificially strong. El Niño = ONI ≥ +0.5 °C for ≥ 5
consecutive seasons; **super** = peak ONI ≥ 2.0 °C.

**Anomalies.** Standardised against a centred moving 30-year
climatology. Standardising is what makes the Amazon and the Limpopo
comparable; without it a global composite is a map of the Amazon.

**Composites.** Superposed-epoch means anchored on **December of the
developing year**, lags −18 to +30 months, so every season code means the
same thing for every event.

**Inference.** Four events, autocorrelated fields, ~60 000 cells tested
at once — no parametric test is defensible. Composites are tested
against a Monte-Carlo null of the same number of anchor years drawn from
ENSO-neutral years; field significance is controlled by Benjamini–
Hochberg FDR (α_FDR = 0.10); and a signal is called *robust* only when
it is also agreed in sign by ≥ 2/3 of the models.

**Power, stated up front.** With n = 4 the composite standard error is
about 0.5 σ. Effects below ~0.3 σ cannot be resolved, so a null result
here is not evidence of linearity — and the paper says so.

`docs/methods.md` gives the full detail, including why the nonlinearity
test uses a piecewise-linear hinge rather than a quadratic (the
quadratic is not identifiable on an El-Niño-only sample: ONI and ONI²
correlate at r > 0.99).

## Data

| | source | licence |
|---|---|---|
| Simulations | ISIMIP3a `water_global` + `lakes_global`, GSWP3-W5E5, 0.5°, monthly, 1901–2019 | CC BY 4.0, per model |
| ENSO indices | NOAA PSL (ERSSTv5 Niño3.4/3/4), NOAA CPC | public domain |
| Basins, lakes, population | GRDC Major River Basins, HydroLAKES, HYDE 3.2 | see each |
| Evaluation | GRDC discharge, GRACE/GRACE-FO, ESA CCI Lakes, ERA5-Land | free, some need registration |

Run `python workflow/03_fetch_isimip3a.py --discover-only` and read
`results/tables/table_S2_isimip_availability.csv` **before** committing
to the download — ISIMIP3a coverage is uneven, and what is missing
changes what the paper can claim. Full details, including a CDS request
template, in `docs/data_sources.md`; sizes and runtimes in
`docs/compute_requirements.md`.

## Reproducibility

* One configuration file drives everything; the smoke-test variant
  inherits from it via a `_base:` key, so the two cannot drift apart.
* Every output netCDF and CSV carries the git revision, the config file
  and its checksum in its attributes.
* Every step logs to `logs/<step>.log`.
* The synthetic smoke test verifies the *science*, not just that the
  code runs: it asserts that the pipeline recovers a known imposed
  teleconnection, known response lags and a known event catalogue.
* `results/tables/README.md` maps every table to the step that made it.
* No number in the manuscript is typed by hand — step 31 extracts them
  and step 40 substitutes them into the template.

## Status

The workflow is complete and validated end to end on synthetic data.
**It has not yet been run on the real ISIMIP3a archive**, which requires
network access to `data.isimip.org` and about 600 GB of disk. Nothing in
`paper/` is a scientific result yet: the manuscript template is a
scaffold that states what each section must argue and which numbers it
must cite.

Next steps, in order:

1. Run steps 01, 03, 04 on a machine with data access; review the
   availability matrix and adjust `config/config.yaml`.
2. Run steps 05–17, then the figure and table steps.
3. Complete the manual downloads in step 04 so that step 16 can
   evaluate the ensemble against observations — this is the part
   reviewers will ask about.
4. Write the manuscript against the rendered numbers.

## Licence

MIT for the code (see `LICENSE`); the input datasets carry their own
licences and citation requirements.
