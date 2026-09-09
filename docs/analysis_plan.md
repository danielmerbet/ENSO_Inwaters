# Analysis plan

Written before the analysis was run, in the spirit of a pre-registration.
Deviations from it are recorded at the bottom, with reasons.

## Question

Do the strongest El Nino events -- the four "super" events of 1972/73,
1982/83, 1997/98 and 2015/16 -- perturb global inland waters in a way
that is **not** predictable by scaling up the response to ordinary
strong events, and how does that perturbation propagate and persist
through rivers, lakes and groundwater?

Three testable sub-questions:

| # | Question | Test | Falsifiable prediction |
|---|----------|------|------------------------|
| Q1 | Where and how strongly do super El Ninos perturb inland waters? | Superposed-epoch composite against a Monte-Carlo null, FDR-controlled, with a 2/3 model-agreement requirement | A coherent, model-agreed dry signal over the maritime continent, Amazon, southern Africa and eastern Australia, and a wet signal over the southern US, coastal Peru/Ecuador and the Parana |
| Q2 | Is the response nonlinear in event amplitude? | (a) super composite minus the moderate/strong composite scaled by the ratio of mean peak ONI; (b) piecewise-linear regression of the response on ONI with a knot at 1.5 degC | If linear, the excess is zero within the bootstrap interval and the hinge coefficient is zero |
| Q3 | How does the signal propagate and how long does it last? | Lag of the composite extremum per variable; ONI cross-correlation; AR(1) memory; months beyond 0.5 sigma after the peak | Peak lag increases monotonically: runoff < discharge < soil moisture < recharge < total storage < groundwater storage |

## Sample

* **Super events**: peak ONI >= 2.0 degC, ONI computed from ERSSTv5
  Nino3.4 with centred 30-year base periods. Expected: 1972, 1982, 1997,
  2015 (developing years).
* **Reference events**: moderate (1.0-1.5) and strong (1.5-2.0) El
  Ninos pooled, ~10-12 events over 1901-2019.
* **Pre-satellite candidates** (1918, 1925, 1940) are excluded from the
  main sample and used only in a sensitivity test, because the SST
  analyses that define them are weakly constrained.

## What would make the answer "no"

Stated in advance so a null result is publishable rather than buried:

* **Q1 null**: fewer than 5% of land cells show a robust response for
  any variable. That would itself be a finding -- it would mean the
  multi-model ensemble has no agreed ENSO hydrology.
* **Q2 null**: the nonlinear excess is indistinguishable from zero
  everywhere and the hinge coefficient's bootstrap interval spans zero.
  **This is a likely outcome**: with four events the composite standard
  error is roughly sigma/2, so an excess below ~0.3 sigma cannot be
  resolved. The paper must report the detection limit, not present a
  null as evidence of linearity.
* **Q3 null**: peak lags do not order as predicted, or the ordering is
  not shared across models. That would point to routing and storage
  parameterisations rather than to hydrology.

## Pre-specified statistical choices

Fixed before looking at the results, to keep the analysis honest:

* Composites anchored on **December of the developing year** (not the
  event's own peak month), so the lag axis has a fixed seasonal meaning.
* **10 000** Monte-Carlo iterations; null anchors drawn from
  ENSO-neutral years excluding the year either side of any event.
* Field significance by **Benjamini-Hochberg FDR at alpha_FDR = 0.10**
  (Wilks 2016), not raw local p-values.
* "Robust" requires **>= 3 models** and **>= 2/3 sign agreement** in
  addition to significance.
* Standardised anomalies against a **centred moving 30-year
  climatology**; the fixed-base-period alternative is a sensitivity test.
* Main scenario **obsclim/histsoc**; counterclim and nosoc used only for
  the factorial contrasts.

## Sensitivity tests to run before drawing conclusions

1. Epoch anchored on the peak month instead of December.
2. Fixed 1901-2019 climatology instead of the moving window.
3. Pre-satellite events included.
4. Leave-one-model-out and leave-one-event-out for every headline number.
5. Absolute rather than standardised anomalies for the basin-scale
   water-balance statements.
6. SSI/SRI computed with the empirical rather than the gamma
   distribution.

## Deviations from the plan

*(Record every deviation here, with the date and the reason. An
undocumented deviation is the difference between an analysis and a
fishing expedition.)*

* 2026-09-09 - The quadratic response function specified in the first
  draft of this plan was replaced by a **piecewise-linear (hinge)**
  fit. Reason: over an El-Nino-only sample ONI and ONI^2 correlate at
  r > 0.99 and the quadratic coefficient is not identified -- the fitted
  b2 came out systematically opposite in sign to b1, which is an
  artefact of the collinearity, not a physical result. Adding all years
  fixes the sign but leaves ONI^2 and ONI|ONI| correlated at ~0.95. The
  hinge basis, with the knot at the strong-event threshold, has a
  maximum predictor correlation of ~0.6 and a directly interpretable
  coefficient. Verified on synthetic data with a known steepening.
