#!/usr/bin/env python3
"""Step 02 - build the ONI and define the super-El-Nino event sample.

What it does
------------
1. Reads Nino3.4 (and Nino3/Nino4 when available) from ``data/raw/enso/``.
2. Converts SST to anomalies against **centred 30-year base periods
   updated every 5 years** (the CPC convention) and takes the 3-month
   running mean -> ONI, 1901-2019.
3. Detects El Nino and La Nina events (ONI beyond +-0.5 degC for >= 5
   consecutive overlapping seasons) and classifies them by peak
   amplitude: weak / moderate / strong / **super (>= 2.0 degC)**.
4. Labels each event Eastern-Pacific or Central-Pacific from the
   Ren & Jin (2011) NCT/NWP indices.
5. Writes the event catalogue that every later step reads, and checks it
   against the events expected from the literature.

Outputs
-------
``data/processed/enso/oni.csv``           monthly ONI (and its inputs)
``data/processed/enso/enso_events.csv``   the event catalogue
``data/processed/enso/event_summary.json`` counts per class, sanity checks

Run:  python workflow/02_define_enso_events.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _common import step_setup, table_path

from enso_inwaters import enso as E
from enso_inwaters.isimip_io import save_table
from enso_inwaters.utils import write_json


def load_nino_series(raw_dir: Path, stem: str, log) -> pd.Series | None:
    """Read a Nino index from a PSL fixed-format file or a generic CSV."""
    psl = raw_dir / f"{stem}.long.data"
    if psl.exists():
        log.info("reading %s (NOAA/PSL format)", psl.name)
        return E.parse_psl_timeseries(psl.read_text())
    csv = raw_dir / f"{stem}.csv"
    if csv.exists():
        log.info("reading %s (generic CSV)", csv.name)
        df = pd.read_csv(csv)
        tcol = df.columns[0]
        vcol = [c for c in df.columns if c != tcol][0]
        s = pd.Series(df[vcol].to_numpy(dtype="float64"),
                      index=pd.to_datetime(df[tcol]))
        s.index.name = "time"
        return s
    return None


def main() -> int:
    cfg, log, args = step_setup("02_define_enso_events", __doc__)
    raw_dir = cfg.path("raw", "enso")
    out_dir = cfg.path("processed", "enso", mkdir=True)
    y0, y1 = int(cfg["period.start"]), int(cfg["period.end"])

    nino34 = load_nino_series(raw_dir, "nino34", log)
    if nino34 is None:
        log.error("no Nino3.4 input found in %s - run step 01 first "
                  "(or tests/make_synthetic_dataset.py for a dry run)", raw_dir)
        return 1

    # ---------------- ONI ------------------------------------------------
    bp = cfg["enso.base_period"]
    already_anom = bool(np.nanmean(np.abs(nino34.values)) < 5.0)  # anomalies, not SST
    log.info("input looks like %s", "anomalies" if already_anom else "absolute SST")
    oni = E.oni_from_nino34(
        nino34, method=bp["method"], window_years=int(bp["window_years"]),
        step_years=int(bp["step_years"]),
        base=(int(cfg["period.climatology_start"]), int(cfg["period.climatology_end"])),
        already_anomaly=already_anom)
    oni = oni[(oni.index.year >= y0) & (oni.index.year <= y1)]
    log.info("ONI: %s to %s, %d months, range %.2f to %.2f",
             oni.index[0].date(), oni.index[-1].date(), len(oni),
             oni.min(), oni.max())

    # ---------------- events --------------------------------------------
    det = cfg["enso.event_detection"]
    classes = {k: tuple(v) for k, v in cfg["enso.classes"].items()}
    el_nino = E.detect_events(oni, threshold=float(det["threshold"]),
                              min_consecutive_seasons=int(det["min_consecutive_seasons"]),
                              phase="el_nino", classes=classes)
    la_nina = E.detect_events(oni, threshold=float(cfg["enso.la_nina_threshold"]),
                              min_consecutive_seasons=int(det["min_consecutive_seasons"]),
                              phase="la_nina", classes=classes)
    log.info("detected %d El Nino and %d La Nina events", len(el_nino), len(la_nina))

    # ---------------- flavour -------------------------------------------
    n3 = load_nino_series(raw_dir, "nino3", log)
    n4 = load_nino_series(raw_dir, "nino4", log)
    if n3 is not None and n4 is not None:
        anom3 = (n3 if float(np.nanmean(np.abs(n3))) < 5 else
                 E.sliding_base_period_anomalies(n3, int(bp["window_years"]),
                                                 int(bp["step_years"])))
        anom4 = (n4 if float(np.nanmean(np.abs(n4))) < 5 else
                 E.sliding_base_period_anomalies(n4, int(bp["window_years"]),
                                                 int(bp["step_years"])))
        rj = E.ren_jin_indices(anom3, anom4)
        el_nino = E.assign_flavour(el_nino, rj)
        la_nina = E.assign_flavour(la_nina, rj)
        log.info("assigned EP/CP flavour from Nino3/Nino4")
    else:
        rj = None
        log.warning("nino3/nino4 not available - EP/CP flavour left as 'unknown'")

    # ---------------- catalogue -----------------------------------------
    events = E.events_to_frame(el_nino + la_nina)
    events["in_analysis_period"] = ((events["year0"] >= y0 + int(cfg["period.discard_spinup_years"]))
                                    & (events["year0"] + 2 <= y1))
    use_pre = bool(cfg["enso.include_presatellite_in_main"])
    events["in_main_sample"] = (events["in_analysis_period"]
                                & (use_pre | ~events["presatellite"]))

    save_table(events, out_dir / "enso_events.csv", "02_define_enso_events", cfg)
    save_table(events, table_path(cfg, "table_S1_enso_events.csv"),
               "02_define_enso_events", cfg)

    idx = pd.DataFrame({"oni": oni})
    if rj is not None:
        idx = idx.join(rj, how="left")
    idx.index.name = "time"
    idx.reset_index().to_csv(out_dir / "oni.csv", index=False)
    log.info("wrote %s", out_dir / "oni.csv")

    # ---------------- checks --------------------------------------------
    supers = events[(events.phase == "el_nino") &
                    (events.amplitude_class == "super") & events.in_main_sample]
    found = sorted(int(y) for y in supers["year0"])
    expected = sorted(int(y) for y in cfg["enso.expected_super_events"])
    log.info("super El Nino events in the main sample: %s", found)
    if found != expected:
        log.warning("super-event sample %s differs from the configured "
                    "expectation %s. Check the SST product, the base-period "
                    "method and the analysis period before continuing.",
                    found, expected)

    counts = (events[events.phase == "el_nino"]["amplitude_class"]
              .value_counts().to_dict())
    summary = {
        "period": [y0, y1],
        "n_el_nino": int((events.phase == "el_nino").sum()),
        "n_la_nina": int((events.phase == "la_nina").sum()),
        "el_nino_class_counts": counts,
        "super_events": found,
        "expected_super_events": expected,
        "matches_expectation": found == expected,
        "mean_peak_oni_super": float(supers["peak_oni"].mean()) if len(supers) else None,
        "oni_source": "sliding 30-yr base periods" if bp["method"] == "sliding_30yr"
                      else "fixed base period",
    }
    write_json(summary, out_dir / "event_summary.json")
    log.info("class counts (El Nino): %s", counts)

    if len(supers) < 3:
        log.warning("only %d super events in the sample - the composite will "
                    "have very little statistical power; report it as such.",
                    len(supers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
