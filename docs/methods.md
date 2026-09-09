# Methods, in detail

This document is the technical companion to `docs/analysis_plan.md`. It
records what each step does and, where a choice was made, why the
alternative was rejected.

## 1. The ENSO index

The Oceanic Nino Index is the 3-month running mean of Nino3.4
(5N-5S, 170W-120W) SST anomalies. Two details matter over a 119-year
record:

**Base period.** CPC references each 5-year block of the record to the
30-year climatology centred on it. Over 1901-2019 the tropical Pacific
warms by roughly 0.7 degC, so a *fixed* base period would make late
events look systematically stronger than early ones and would put
1997/98 and 2015/16 in a different class from 1972/73 for reasons that
have nothing to do with ENSO. The sliding base period removes that drift
without a parametric detrend. `enso.base_period.method: sliding_30yr`.

**A side effect worth knowing.** A 30-year window necessarily contains
several El Ninos, so the December climatology it defines is warm, and
every event's ONI is 0.3-0.4 degC smaller than the raw SST bump that
produced it. This is a property of the operational index, not an error;
the amplitude thresholds are calibrated to it.

**Event definition.** ONI >= +0.5 degC for at least five consecutive
overlapping seasons. Amplitude classes from the peak ONI: weak
0.5-1.0, moderate 1.0-1.5, strong 1.5-2.0, **super >= 2.0**. Events
that share a developing year (double-peaked cases) are merged, keeping
the larger peak.

**Flavour.** EP and CP events are separated with the Ren and Jin (2011)
transformed indices (NCT, NWP) computed from Nino3 and Nino4, rather
than with the EOF-based E/C indices of Takahashi et al. (2011). The
transformed indices need only two standard SST indices and are
reproducible from public monthly data; the EOF indices require the full
SST field and a defensible EOF domain. All four super events are
expected to be EP-type, which is worth stating explicitly because it
means the "super" and "EP" classifications are nearly confounded and
neither can be attributed the response on its own.

## 2. Anomalies

Standardised anomalies against a **centred moving 30-year
climatology**, computed per calendar month:

```
anomaly(t) = (x(t) - mean_30yr(month(t))) / sd_30yr(month(t))
```

Three reasons for standardising rather than using absolute units:

1. A composite in physical units is a map of the Amazon. Discharge
   spans five orders of magnitude between basins; the question is the
   *relative* perturbation.
2. It makes variables with incompatible units -- m3/s, kg/m2,
   degrees -- comparable on one colour scale, which is what the cascade
   figure needs.
3. It is the same normalisation the standardised drought indices use, so
   the composite maps and the drought statistics are on one footing.

Absolute anomalies are retained for the basin-scale water-balance
numbers, where "12% less discharge" is the meaningful statement.

The moving window is computed by cumulative sums over the year axis,
which is O(n) rather than O(n x window) and is what makes the step
tractable on the full 0.5-degree grid. Correctness is checked against a
direct windowed mean in `tests/test_climatology.py`.

## 3. Composites

Superposed-epoch analysis anchored on **December of the developing
year**, lags -18 to +30 months. Anchoring on each event's own peak month
maximises the composite amplitude but blurs the seasonal cycle -- the
1972 peak was in December, 2015's in November -- so seasons like "JJA of
the developing year" stop meaning the same thing across events. December
anchoring keeps the season codes exact; peak anchoring is available as
`--anchor peak` for a sensitivity test.

Season codes are relative to that anchor: `JJA0` = lags -6..-4,
`SON0` = -3..-1, `DJF01` = 0..2, `MAM1` = 3..5, `JJA1` = 6..8,
`SON1` = 9..11, `DJF12` = 12..14.

## 4. Inference with four events

**The problem.** Four events, autocorrelated fields, non-Gaussian
anomalies, ~60 000 land cells tested at once. No parametric test is
defensible.

**The null.** Composites of the same number of anchor years drawn at
random from ENSO-neutral years, with the years either side of any real
event excluded so the null cannot sample a genuine teleconnection.
Each null draw uses contiguous epoch windows, so the serial correlation
of hydrological memory is preserved by construction. 10 000 iterations;
p is reported as (count + 1)/(n + 1) so it is never zero.

**Efficiency.** Every draw takes windows from the same fixed pool of
candidate anchor years, so the pool is extracted once
(`stats.build_anchor_pool`) and the bootstrap is a mean over rows. This
turns a 10 000-iteration test on a global grid from hours into minutes.

**Field significance.** Testing 60 000 cells at p < 0.05 produces 3 000
false positives by construction. Maps report Benjamini-Hochberg
FDR-controlled significance with alpha_FDR = 0.10, following Wilks
(2016), which is far less conservative than Bonferroni and has a clear
interpretation: at most 10% of the flagged cells are expected to be
false.

**Robustness.** A cell is called robust only if it is FDR-significant
*and* at least 2/3 of the available models agree on the sign of the
ensemble mean. A significant ensemble mean driven by one outlier model
is not a result.

**Power.** With n = 4 and standardised anomalies, the standard error of
the composite mean is about 0.5 sigma. Effects smaller than roughly
0.3 sigma are undetectable. The paper must state this limit wherever it
reports a null.

## 5. Testing for nonlinearity

Two independent tests, deliberately different in their assumptions.

**(a) Excess over a linear scaling.** The moderate/strong composite is
multiplied by the ratio of mean peak ONI (super / reference) and
subtracted from the super composite. The residual is the nonlinear
excess. Uncertainty comes from resampling events with replacement within
each group -- the only resampling unit that respects the sample size.
Assumption-free, but low-powered.

**(b) Piecewise-linear response function.** Across **every year of the
record**, the seasonal response of each cell is regressed on the DJF ONI
as

```
y = b0 + b1*ONI + h_pos*max(ONI - 1.5, 0) + h_neg*min(ONI + 1.5, 0)
```

`h_pos` is the extra sensitivity beyond the strong-event threshold, and
`h_pos / b1` is the headline steepening number. `h_neg` gives the La
Nina counterpart, so the symmetry of the nonlinearity is measured rather
than assumed.

**Why not a quadratic.** It is not identifiable here. On an El-Nino-only
sample, ONI and ONI^2 correlate at r > 0.99; the coefficients trade off
and the fitted quadratic term comes out systematically opposite in sign
to the linear term. Extending the sample to both phases fixes the sign
but leaves ONI^2 and ONI|ONI| correlated at ~0.95, because the large
values of the sample are not symmetric. The hinge basis has a maximum
predictor correlation of about 0.6 on the same sample, and its
coefficient answers the question directly.

**A known attenuation.** The hinge coefficient is a *lower bound*. The
regression uses a seasonal-mean ONI while the true driver is monthly, so
part of the steepening is absorbed by the linear term. On synthetic data
with a known steepening of +0.30, the fit recovers about +0.09 with the
correct sign in 55% of signal-bearing cells. Report the coefficient as a
conservative estimate.

## 6. The cascade

Peak lag is the lag of the composite extremum, per cell and for the area
mean. It is cross-checked against the lag of maximum ONI correlation,
which does not depend on the event sample at all -- if the two disagree,
the composite is being driven by one event. Memory is the AR(1)
e-folding time of the anomalies, which sets the ceiling on how long a
perturbation *can* persist in a given model. Recovery time is the last
lag after the peak at which |composite| still exceeds 0.5 sigma.

## 7. Factorial contrasts

ISIMIP3a is designed for this. Holding everything else fixed:

* `obsclim - counterclim` isolates the effect of the warmed background
  state on the El Nino hydrological response;
* `histsoc - nosoc` isolates reservoirs, irrigation and abstraction.

Both are differenced **per event** before averaging, so the contrast is
paired and the event-to-event variability cancels.

A caveat: standardised anomalies are invariant to a constant rescaling
of a field, so a scenario that changes only the mean level of a store
produces no contrast in sigma units. Where the mean level is the point,
use the absolute anomalies.

## 8. Uncertainty

Two-way ANOVA of the (model x event) response array at every cell,
splitting the variance into a model term, an event term and a residual.
Where the model term dominates, the honest statement is "models
disagree". Where the event term dominates, super El Ninos are diverse
and no composite describes the next one well. Every headline number is
additionally recomputed leaving out one model and one event at a time;
a number that moves by more than half when one member is dropped is
reported as such.
