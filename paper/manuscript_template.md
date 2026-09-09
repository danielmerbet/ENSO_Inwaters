---
title: >-
  Super El Nino events and global inland waters: a nonlinear, lagged
  response of rivers, lakes and groundwater in the ISIMIP3a ensemble
running_title: Super El Nino and inland waters
target_journals:
  - Nature Water
  - Nature Communications
  - Water Resources Research
  - Hydrology and Earth System Sciences
status: TEMPLATE - placeholders are filled by workflow/40_render_manuscript.py
---

<!--
  Placeholders address results/paper_numbers.json by dotted path, with an
  optional format spec:  {{events.n_super}}  {{basins.median_amplification:.2f}}
  Anything unresolved is left visible in the rendered draft on purpose.

  WRITE THE TEXT ONLY AFTER THE ANALYSIS HAS RUN ON REAL DATA. The
  paragraphs below are a scaffold: they state what each section must
  argue and which numbers it must cite, not what the answer is.
-->

# Abstract

*(150-200 words. Structure: what is known; what is not; what we did;
what we found, with numbers; why it matters.)*

The strongest El Nino events -- 1972/73, 1982/83, 1997/98 and 2015/16 --
are usually described through their atmospheric teleconnections, yet
their consequences are felt in rivers, lakes and aquifers, on timescales
the atmosphere does not set. Using {{significance.n_models}} global
hydrological and lake models driven by observed climate over
{{events.period.0}}-{{events.period.1}} in ISIMIP3a, we composite the
inland-water response to the {{events.n_super}} super El Nino events
(mean peak ONI {{events.mean_peak_oni_super:.2f}} degC) and compare it
with {{events.n_reference}} moderate and strong events. *(Then: the
headline result -- the size of the response, whether it exceeds a linear
scaling by {{events.linear_scaling_factor:.2f}}, how far the peak in
groundwater lags the peak in runoff, and how long the perturbation
lasts. Then: the exposure figure and the policy consequence.)*

# 1. Introduction

*Three moves, roughly a paragraph each.*

1. **Why inland waters, and why the extremes of ENSO.** ENSO is the
   dominant mode of interannual climate variability and its
   teleconnections to precipitation are mapped in detail. The water that
   reaches rivers, lakes and aquifers is not precipitation, though: it
   is precipitation filtered through soil moisture deficits, routing and
   storage, each of which is nonlinear and each of which has memory.
   The strongest events are precisely where a linear reading of the
   teleconnection is least likely to hold.
2. **What is already known and what is not.** Event studies of 1997/98
   and 2015/16, basin-scale ENSO-streamflow relationships, GRACE-era
   storage analyses -- and the gap: no global, multi-model, multi-sector
   assessment that treats super events as a class and tests whether
   their inland-water response is separable from a scaled-up strong
   event.
3. **What this paper does.** State the three questions answered:
   *(i)* where and how strongly do super El Ninos perturb rivers, lakes
   and groundwater; *(ii)* is that response nonlinear in event
   amplitude; *(iii)* how long does it take to arrive, and how long does
   it last.

# 2. Data and methods

## 2.1 ENSO index and event classification

ONI from ERSSTv5 Nino3.4 anomalies, 3-month running mean, centred
30-year base periods updated every five years, extended back to
{{events.period.0}}. El Nino events are runs of ONI >= +0.5 degC for at
least five consecutive overlapping seasons; amplitude classes follow the
CPC convention, with **super** defined as peak ONI >= 2.0 degC. This
yields {{events.n_super}} super events ({{events.super_events}}) and
{{events.n_reference}} moderate/strong reference events in the main
sample. Pre-satellite candidates are carried only as a sensitivity test
because the SST analyses that define them are far less constrained.
Events are additionally labelled Eastern- or Central-Pacific using the
Ren and Jin (2011) transformed indices.

## 2.2 ISIMIP3a simulations

Global water-sector models (
{{significance.n_models}} models) and lake-sector models forced by
GSWP3-W5E5 observed climate, 0.5-degree, monthly, 1901-2019. The main
configuration is obsclim/histsoc; counterclim (warming removed) and
nosoc (direct human water use removed) are used for the factorial
contrasts in Section 3.5. Variables: total runoff, river discharge,
groundwater recharge, groundwater storage, root-zone soil moisture,
total water storage, lake surface temperature and lake ice.

## 2.3 Anomalies and composites

Standardised anomalies against a centred moving 30-year climatology,
which removes the century-scale drift without imposing a trend model
while leaving the 2-7 year ENSO band untouched. Composites are
superposed-epoch means anchored on December of the developing year, from
18 months before to 30 months after, so the seasonal meaning of each lag
is identical across events.

## 2.4 Statistical inference

With a sample of {{events.n_super}} events, parametric tests are not
defensible. Composites are tested against a Monte-Carlo null built from
the same number of anchor years drawn at random from ENSO-neutral years,
with field significance controlled by the Benjamini-Hochberg FDR
procedure (alpha_FDR = 0.10). A response is called *robust* only when it
is FDR-significant **and** at least two thirds of the available models
agree on its sign. Two nonlinearity tests are applied: the excess of the
super composite over the moderate/strong composite scaled by the ratio
of mean peak ONI ({{events.linear_scaling_factor:.2f}}), and a
piecewise-linear regression of the response on ONI across all years with
a knot at ONI = 1.5 degC. Uncertainty is partitioned between impact
model, event and residual by two-way ANOVA, and every headline number is
re-computed leaving out one model and one event at a time.

# 3. Results

## 3.1 The global footprint of a super El Nino (Fig. 2)

*(Cite: fraction of the land surface with a robust response per
variable -- runoff {{significance.by_variable.qtot.frac_robust:.1%}},
discharge {{significance.by_variable.dis.frac_robust:.1%}}, recharge
{{significance.by_variable.qr.frac_robust:.1%}}, storage
{{significance.by_variable.tws.frac_robust:.1%}}; the maximum composite
amplitude; the named regions of coherent drying and wetting.)*

## 3.2 The response is delayed, and the delay grows down the cascade (Fig. 4)

The area-mean response peaks in the order
{{lag_cascade.ordering}}, with peak lags of
{{lag_cascade.peak_lag_months}} months relative to the mature phase.
The independent ONI cross-correlation gives the same ordering.
Groundwater storage stays perturbed for a median of
{{lag_cascade.median_recovery_months.groundwstor}} months after the
event peak -- *(state what that means for a water manager: the drought
that matters begins after the El Nino is over.)*

## 3.3 Super events are not scaled-up strong events (Fig. 3)

*(Cite: mean absolute nonlinear excess per variable; the fraction of the
land surface where it is significant; the fraction of cells where the
response steepens beyond ONI = 1.5; the median steepening ratio. Be
explicit about the power limitation: with {{events.n_super}} events, an
excess smaller than about 0.3 sigma cannot be resolved, so a null result
is not evidence of linearity.)*

At basin scale the median super/reference amplification is
{{basins.median_amplification:.2f}} across {{basins.n_basins}} major
basins, against a linear expectation of
{{events.linear_scaling_factor:.2f}}.

## 3.4 Droughts, floods and exposure (Fig. 6)

Super-El-Nino windows contain {{extremes.super_drought_months:.1f}}
drought months in the two years after the peak against
{{extremes.neutral_drought_months:.1f}} in neutral windows (median odds
ratio {{extremes.median_odds_ratio:.2f}}).
*(Then the exposure figures -- area {{exposure.area_dry_1e6km2:.1f}}
million km2, population {{exposure.pop_dry_millions:.0f}} million -- with
the explicit caveat that these count people living where the anomaly is
robust, not people harmed.)*

## 3.5 What modulates the response: warming and water management

*(Cite the two contrasts from table 7 and Fig. 6c,d. State clearly which
is larger and where.)*

## 3.6 Model agreement and what it limits

*(Cite the variance partition: the fraction of variance attributable to
the choice of impact model versus to which event it was, per variable.
Where the model term dominates, say so and reduce the claim
accordingly. Report the leave-one-out sensitivity.)*

# 4. Discussion

1. **Mechanism.** Why the lag grows down the cascade, and why storage
   integrates: the catchment as a low-pass filter with a memory set by
   soil and aquifer storage.
2. **Why nonlinearity is expected on physical grounds** -- runoff
   generation thresholds, soil-moisture deficits that must be filled
   before recharge resumes, reservoir operating rules -- and what the
   results do and do not show about it.
3. **Comparison with observations and previous work.**
   *(If the evaluation in Section 2 could not be run for want of the
   observational datasets, say so here explicitly.)*
4. **Limitations.** Four super events. Impact-model spread. The
   pre-satellite events. Monthly resolution, which cannot resolve flood
   peaks. Gridded lake models that represent a lake by its cells. The
   fact that composites describe the average of four events, not the
   next one.
5. **Implications.** For seasonal water-resource forecasting, for
   groundwater monitoring after an event ends, and for how the
   1972-2016 record should inform expectations of the next super event.

# 5. Conclusions

*(Five to seven sentences. No new numbers.)*

# Data availability

ISIMIP3a simulations: https://data.isimip.org (ISIMIP3a OutputData,
water_global and lakes_global sectors). ENSO indices: NOAA PSL and CPC.
Auxiliary datasets are listed with their DOIs in `docs/data_sources.md`.

# Code availability

The complete workflow that produced every figure, table and number in
this paper is at *(repository URL)*, released under the MIT licence.
Numbered scripts in `workflow/` reproduce the analysis end to end;
`config/config.yaml` holds every parameter. Configuration used for this
manuscript: `{{_config}}`.

# Author contributions

*(CRediT taxonomy.)*

# Competing interests

*(Declare.)*

# Acknowledgements

The ISIMIP community and the modelling groups that contributed the
simulations used here.

---

## Figures

| # | file | content |
|---|------|---------|
| 1 | `results/figures/fig01_enso_events.*` | ONI record, event catalogue and the super-event sample |
| 2 | `results/figures/fig02_global_composites_super.*` | global composite response by variable and phase |
| 3 | `results/figures/fig03_nonlinearity.*` | nonlinear excess and the response-amplitude relationship |
| 4 | `results/figures/fig04_lag_cascade.*` | the lagged cascade from runoff to groundwater |
| 5 | `results/figures/fig05_basins_lakes.*` | basin-scale and lake responses |
| 6 | `results/figures/fig06_impacts.*` | droughts, exposure, and the scenario contrasts |
| S1 | `results/figures/figS1_per_model.*` | per-model composites |
| S2 | `results/figures/figS2_variance_partition.*` | sources of ensemble spread |
| S3 | `results/figures/figS3_enso_symmetry.*` | El Nino / La Nina symmetry |

## Tables

See `results/tables/README.md` for the full list with captions and the
workflow step that produced each.
