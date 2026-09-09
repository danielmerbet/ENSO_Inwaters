"""ENSO index handling, event detection and superposed-epoch compositing.

Definitions follow NOAA/CPC practice:

* **Nino3.4** SST anomaly, 5N-5S / 170W-120W (ERSSTv5).
* **ONI** = 3-month running mean of Nino3.4 anomalies, labelled by the
  central month, computed against *centred 30-year base periods updated
  every 5 years* so that the tropical warming trend is not aliased into
  the index (CPC procedure; extended back to 1901 here).
* **El Nino event** = ONI >= +0.5 degC for at least 5 consecutive
  overlapping seasons.
* **Amplitude class** from the peak ONI: weak 0.5-1.0, moderate 1.0-1.5,
  strong 1.5-2.0, **super (very strong) >= 2.0**.

Over 1901-2019 the super class contains 1972/73, 1982/83, 1997/98 and
2015/16 in the satellite/ERSSTv5-reliable period; 1918/19, 1925/26 and
1940/41 are pre-satellite candidates carried only as a sensitivity test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import xarray as xr

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_LABELS = ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ",
                 "MJJ", "JJA", "JAS", "ASO", "SON", "OND"]


# ---------------------------------------------------------------------
# Parsers for the raw index files
# ---------------------------------------------------------------------
def parse_psl_timeseries(text: str, missing: float | None = None) -> pd.Series:
    """Parse a NOAA/PSL fixed-format monthly time series.

    Layout: a header line ``start_year end_year``, then one line per year
    with 12 monthly values, then a trailing missing-value line.
    Returns a monthly ``pd.Series`` indexed by period-end timestamps.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header = lines[0].split()
    y0, y1 = int(header[0]), int(header[1])
    records: dict[pd.Timestamp, float] = {}
    miss_values = [] if missing is None else [missing]
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 13 and re.fullmatch(r"-?\d{4}", parts[0]):
            year = int(parts[0])
            if not (y0 <= year <= y1):
                continue
            for m, val in enumerate(parts[1:], start=1):
                records[pd.Timestamp(year=year, month=m, day=1)] = float(val)
        elif len(parts) == 1:
            try:
                miss_values.append(float(parts[0]))
            except ValueError:
                continue
    s = pd.Series(records).sort_index()
    s.index.name = "time"
    for mv in miss_values:
        s = s.mask(np.isclose(s.values, mv))
    # PSL sometimes uses -99.99 / -9999 without declaring it
    s = s.mask(s <= -90.0)
    return s.astype("float64")


def parse_cpc_nino34_ascii(text: str) -> pd.DataFrame:
    """Parse the CPC ``detrend.nino34.ascii.txt`` table (YR MON TOTAL ...)."""
    df = pd.read_csv(pd.io.common.StringIO(text), sep=r"\s+")
    df.columns = [c.strip().upper() for c in df.columns]
    time = pd.to_datetime(dict(year=df["YR"], month=df["MON"], day=1))
    out = pd.DataFrame({"nino34": df["TOTAL"].astype(float),
                        "nino34_anom": df["ANOM"].astype(float)}, index=time)
    out.index.name = "time"
    return out


# ---------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------
def sliding_base_period_anomalies(sst: pd.Series, window_years: int = 30,
                                  step_years: int = 5) -> pd.Series:
    """Anomalies against centred 30-yr base periods updated every 5 yr.

    This is the CPC ONI convention. Each 5-year block of the record is
    referenced to the 30-year climatology centred on that block, which
    removes the low-frequency warming of the tropical Pacific without a
    parametric detrend.

    Blocks at the ends of the record cannot be centred and fall back to
    the first / last complete window, which leaves a small residual of
    the trend there. On a 1901-2019 record with 1.2 degC of warming the
    residual is under +-0.15 degC and confined to roughly the first and
    last 15 years: it slightly deflates pre-1916 events and inflates
    post-2005 ones. It matters for exactly one thing in this workflow --
    whether 2015/16 clears the 2.0 degC super threshold partly on the
    strength of a base-period artefact -- so that event's classification
    is worth re-checking against the operational CPC ONI, which step 02
    does when the CPC file is available.
    """
    sst = sst.dropna().sort_index()
    years = sst.index.year
    y0, y1 = int(years.min()), int(years.max())
    half = window_years // 2
    anom = pd.Series(index=sst.index, dtype="float64")

    block_starts = range(y0 - (y0 % step_years), y1 + 1, step_years)
    for bstart in block_starts:
        bend = bstart + step_years - 1
        centre = bstart + step_years / 2.0
        base_lo = int(round(centre - half))
        base_hi = base_lo + window_years - 1
        # clamp to the available record
        if base_lo < y0:
            base_lo, base_hi = y0, y0 + window_years - 1
        if base_hi > y1:
            base_hi, base_lo = y1, y1 - window_years + 1
        base_lo, base_hi = max(base_lo, y0), min(base_hi, y1)

        base = sst[(sst.index.year >= base_lo) & (sst.index.year <= base_hi)]
        clim = base.groupby(base.index.month).mean()
        blk = sst[(sst.index.year >= bstart) & (sst.index.year <= bend)]
        if blk.empty:
            continue
        anom.loc[blk.index] = blk.values - clim.reindex(blk.index.month).values
    return anom.dropna()


def fixed_base_period_anomalies(sst: pd.Series, start: int, end: int) -> pd.Series:
    base = sst[(sst.index.year >= start) & (sst.index.year <= end)]
    clim = base.groupby(base.index.month).mean()
    return sst - clim.reindex(sst.index.month).values


def oni_from_nino34(nino34: pd.Series, method: str = "sliding_30yr",
                    window_years: int = 30, step_years: int = 5,
                    base: tuple[int, int] = (1991, 2020),
                    already_anomaly: bool = False) -> pd.Series:
    """3-month running mean of Nino3.4 anomalies, labelled by central month."""
    if already_anomaly:
        anom = nino34.dropna().sort_index()
    elif method == "sliding_30yr":
        anom = sliding_base_period_anomalies(nino34, window_years, step_years)
    elif method == "fixed":
        anom = fixed_base_period_anomalies(nino34, *base)
    else:
        raise ValueError(f"unknown base-period method: {method}")
    oni = anom.rolling(3, center=True, min_periods=3).mean()
    oni.name = "oni"
    return oni.dropna()


def season_label(month: int) -> str:
    """CPC season label for a 3-month mean centred on ``month``."""
    return SEASON_LABELS[month % 12]


# ---------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------
@dataclass
class ENSOEvent:
    """One El Nino (or La Nina) event on the ONI series."""
    year0: int                 # developing year, e.g. 1997 for 1997/98
    label: str                 # "1997/98"
    phase: str                 # "el_nino" | "la_nina"
    onset: pd.Timestamp        # first month of the qualifying sequence
    end: pd.Timestamp          # last month of the qualifying sequence
    peak_time: pd.Timestamp    # month of the extreme ONI
    peak_oni: float            # extreme ONI value
    djf_oni: float             # ONI of the DJF season (Jan-centred)
    duration_months: int
    amplitude_class: str
    flavour: str = "unknown"   # EP | CP | unknown
    presatellite: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("onset", "end", "peak_time"):
            d[k] = pd.Timestamp(d[k]).strftime("%Y-%m")
        return d


def classify_amplitude(peak: float, classes: dict) -> str:
    """Map |peak ONI| onto the configured amplitude classes."""
    a = abs(float(peak))
    for name, (lo, hi) in classes.items():
        if lo <= a < hi:
            return name
    return "none"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop (inclusive) indices of True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def detect_events(oni: pd.Series, threshold: float = 0.5,
                  min_consecutive_seasons: int = 5,
                  phase: str = "el_nino",
                  classes: dict | None = None,
                  presatellite_before: int = 1950) -> list[ENSOEvent]:
    """Detect ENSO events as runs of the ONI beyond ``threshold``.

    ``min_consecutive_seasons`` counts overlapping 3-month seasons, i.e.
    consecutive monthly values of the running-mean series.
    """
    classes = classes or {"weak": [0.5, 1.0], "moderate": [1.0, 1.5],
                          "strong": [1.5, 2.0], "super": [2.0, 99.0]}
    oni = oni.dropna().sort_index()
    if phase == "el_nino":
        mask = (oni >= threshold).values
        pick = np.argmax
    elif phase == "la_nina":
        mask = (oni <= threshold).values
        pick = np.argmin
    else:
        raise ValueError(phase)

    events: list[ENSOEvent] = []
    for i0, i1 in _runs(mask):
        n = i1 - i0 + 1
        if n < min_consecutive_seasons:
            continue
        seg = oni.iloc[i0:i1 + 1]
        k = int(pick(seg.values))
        peak_time = seg.index[k]
        peak_oni = float(seg.iloc[k])
        # Developing year: events peak in boreal winter, so a peak in
        # Jan-Jul belongs to the preceding calendar year.
        year0 = peak_time.year if peak_time.month >= 8 else peak_time.year - 1
        djf_time = pd.Timestamp(year=year0 + 1, month=1, day=1)
        djf = float(oni.get(djf_time, np.nan))
        events.append(ENSOEvent(
            year0=year0,
            label=f"{year0}/{str(year0 + 1)[-2:]}",
            phase=phase,
            onset=seg.index[0], end=seg.index[-1],
            peak_time=peak_time, peak_oni=peak_oni, djf_oni=djf,
            duration_months=n,
            amplitude_class=classify_amplitude(peak_oni, classes),
            presatellite=year0 < presatellite_before,
        ))
    # Merge events that share a developing year (double-peaked cases)
    merged: dict[int, ENSOEvent] = {}
    for ev in events:
        keep = merged.get(ev.year0)
        if keep is None or abs(ev.peak_oni) > abs(keep.peak_oni):
            merged[ev.year0] = ev
    return [merged[k] for k in sorted(merged)]


def select_event_class(events: pd.DataFrame, klass: str,
                       main_sample_only: bool = True) -> pd.DataFrame:
    """Rows of the event catalogue belonging to one composite class.

    Recognised classes: any amplitude class (``super``, ``strong``,
    ``moderate``, ``weak``), ``reference`` / ``moderate_strong`` (the
    pooled moderate+strong El Ninos the nonlinearity test scales),
    ``all_el_nino``, ``all_la_nina`` and ``la_nina_<class>``.
    """
    ev = events[events["in_main_sample"]] if (
        main_sample_only and "in_main_sample" in events) else events
    if klass == "all_el_nino":
        return ev[ev.phase == "el_nino"]
    if klass == "all_la_nina":
        return ev[ev.phase == "la_nina"]
    if klass.startswith("la_nina_"):
        return ev[(ev.phase == "la_nina") &
                  (ev.amplitude_class == klass.split("_", 2)[2])]
    if klass in ("reference", "moderate_strong"):
        return ev[(ev.phase == "el_nino") &
                  (ev.amplitude_class.isin(["moderate", "strong"]))]
    return ev[(ev.phase == "el_nino") & (ev.amplitude_class == klass)]


def events_to_frame(events: Iterable[ENSOEvent]) -> pd.DataFrame:
    df = pd.DataFrame([e.to_dict() for e in events])
    if not df.empty:
        df = df.sort_values("year0").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# ENSO flavour (Eastern vs Central Pacific)
# ---------------------------------------------------------------------
def ren_jin_indices(nino3: pd.Series, nino4: pd.Series) -> pd.DataFrame:
    """Ren & Jin (2011) transformed NCT / NWP indices.

    ``N_CT = N3 - a*N4``, ``N_WP = N4 - a*N3`` with ``a = 0.4`` when
    ``N3*N4 > 0`` and ``a = 0`` otherwise. NCT-dominant events are
    Eastern-Pacific ("canonical") El Ninos; NWP-dominant events are
    Central-Pacific ("Modoki") events. A cheap, reproducible alternative
    to the EOF-based E/C indices of Takahashi et al. (2011).
    """
    n3, n4 = nino3.align(nino4, join="inner")
    a = np.where(n3.values * n4.values > 0, 0.4, 0.0)
    nct = n3.values - a * n4.values
    nwp = n4.values - a * n3.values
    return pd.DataFrame({"nct": nct, "nwp": nwp}, index=n3.index)


def assign_flavour(events: Sequence[ENSOEvent], rj: pd.DataFrame,
                   season_months: Sequence[int] = (12, 1, 2)) -> list[ENSOEvent]:
    """Label each event EP or CP from the mature-season NCT/NWP ratio."""
    for ev in events:
        win = [pd.Timestamp(year=ev.year0 + (0 if m == 12 else 1), month=m, day=1)
               for m in season_months]
        sub = rj.reindex(win).dropna()
        if sub.empty:
            continue
        ev.flavour = "EP" if abs(sub["nct"].mean()) >= abs(sub["nwp"].mean()) else "CP"
    return list(events)


# ---------------------------------------------------------------------
# Superposed epoch analysis
# ---------------------------------------------------------------------
def event_reference(event, anchor: str = "december") -> pd.Timestamp:
    """Epoch anchor (lag 0) for one event.

    ``december`` (default) anchors every event on December of its
    developing year, so that the ``lag`` axis keeps a fixed seasonal
    meaning across events and the season codes in ``_season_to_lags``
    are exact. ``peak`` anchors on the observed month of maximum ONI
    instead, which maximises composite amplitude but blurs the seasonal
    cycle; it is offered as a sensitivity test.
    """
    year0 = int(event["year0"] if isinstance(event, dict) else event.year0)
    if anchor == "december":
        return pd.Timestamp(year=year0, month=12, day=1)
    if anchor == "peak":
        pt = event["peak_time"] if isinstance(event, dict) else event.peak_time
        return pd.Timestamp(pt)
    raise ValueError(f"unknown epoch anchor: {anchor}")


def lag_months(times: pd.DatetimeIndex, reference: pd.Timestamp) -> np.ndarray:
    """Integer month offset of each time from ``reference``."""
    times = pd.DatetimeIndex(times)
    return ((times.year - reference.year) * 12 +
            (times.month - reference.month)).to_numpy()


def epoch_slice(da: xr.DataArray, reference: pd.Timestamp,
                lag_min: int, lag_max: int, time_dim: str = "time") -> xr.DataArray:
    """Extract ``da`` over [lag_min, lag_max] months around ``reference``.

    The result carries a ``lag`` coordinate in months and is padded with
    NaN where the requested window falls outside the record, so that
    events near the ends of the record still contribute what they can.
    """
    times = pd.DatetimeIndex(da[time_dim].values)
    lags = lag_months(times, pd.Timestamp(reference))
    wanted = np.arange(lag_min, lag_max + 1)
    idx = {int(lg): int(i) for i, lg in enumerate(lags)
           if lag_min <= lg <= lag_max}
    take = [idx.get(int(lg), -1) for lg in wanted]

    present = [t for t in take if t >= 0]
    if not present:
        raise ValueError(f"no overlap between record and epoch around {reference}")
    sub = da.isel({time_dim: present})
    sub = sub.rename({time_dim: "lag"}).assign_coords(
        lag=[lg for lg, t in zip(wanted, take, strict=True) if t >= 0])
    return sub.reindex(lag=wanted)


def superposed_epoch(da: xr.DataArray, references: Sequence[pd.Timestamp],
                     lag_min: int, lag_max: int, time_dim: str = "time",
                     keep_events: bool = False) -> xr.DataArray:
    """Composite ``da`` across events, aligned on each event's peak month.

    Returns the event mean over ``lag`` (or, with ``keep_events=True``,
    the stacked per-event epochs, which the bootstrap and the
    variance-partitioning steps need).
    """
    epochs = []
    labels = []
    for ref in references:
        ref = pd.Timestamp(ref)
        try:
            epochs.append(epoch_slice(da, ref, lag_min, lag_max, time_dim))
            labels.append(ref.strftime("%Y-%m"))
        except ValueError:
            continue
    if not epochs:
        raise ValueError("no usable events for the composite")
    stacked = xr.concat(epochs, dim=xr.DataArray(labels, dims="event",
                                                 name="event"))
    return stacked if keep_events else stacked.mean("event", skipna=True)


def season_mean(epoch: xr.DataArray, season: str) -> xr.DataArray:
    """Mean of an epoch over a named season relative to the event peak.

    Season names follow ``<MONTHS><year-offset>``: ``JJA0`` is the
    developing summer, ``DJF01`` the mature winter, ``MAM1`` the decaying
    spring, ``DJF12`` the following winter. The peak month is taken as
    lag 0 (climatologically December-January).
    """
    lags = _season_to_lags(season)
    return epoch.sel(lag=[lg for lg in lags if lg in epoch.lag.values]).mean("lag")


# First calendar month (1-12) of each 3-month season.
_SEASON_START_MONTH = {
    "JFM": 1, "FMA": 2, "MAM": 3, "AMJ": 4, "MJJ": 5, "JJA": 6,
    "JAS": 7, "ASO": 8, "SON": 9, "OND": 10, "NDJ": 11, "DJF": 12,
}


def _season_to_lags(season: str) -> list[int]:
    """Translate e.g. 'JJA0' / 'DJF01' / 'MAM1' into lag months.

    Lag 0 is the event peak, taken to be December of the developing year
    (``year0``). A calendar month ``m`` of ``year0`` therefore sits at
    lag ``m - 12`` and month ``m`` of the decaying year at lag ``m``:

    ==========  ==========================  ============
    code        months                      lags
    ==========  ==========================  ============
    ``JJA0``    Jun-Aug, developing year    -6 .. -4
    ``SON0``    Sep-Nov, developing year    -3 .. -1
    ``DJF01``   Dec(0)-Feb(1), mature       0 .. 2
    ``MAM1``    Mar-May, decaying year      3 .. 5
    ``JJA1``    Jun-Aug, decaying year      6 .. 8
    ``DJF12``   Dec(1)-Feb(2), following    12 .. 14
    ==========  ==========================  ============
    """
    m = re.fullmatch(r"([A-Z]{3})(\d{1,2})", season.upper())
    if not m:
        raise ValueError(f"unrecognised season code: {season}")
    name, yr = m.group(1), m.group(2)
    if name not in _SEASON_START_MONTH:
        raise ValueError(f"unrecognised season: {name}")
    # '0' / '01' -> developing year; '1' / '12' -> decaying year; etc.
    year_offset = 0 if yr in ("0", "01") else int(yr[0])
    start = _SEASON_START_MONTH[name] - 12 + 12 * year_offset
    return [start, start + 1, start + 2]
