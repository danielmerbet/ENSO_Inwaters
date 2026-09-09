#!/usr/bin/env bash
# End-to-end smoke test of the whole workflow on SYNTHETIC data.
#
#   bash tests/test_workflow_smoke.sh
#
# Generates a small synthetic dataset with a known ENSO teleconnection,
# runs every analysis step, and checks that the pipeline recovers what
# was put in. Takes a few minutes and needs no network.
#
# What it asserts:
#   * step 02 finds exactly the four synthetic super events
#   * the composite recovers the imposed dry/wet pattern
#   * the lag cascade recovers the imposed response delays
#   * every figure and table is produced
set -euo pipefail

cd "$(dirname "$0")/.."
export ENSO_INWATERS_CONFIG=config/config_synthetic.yaml
PYTHON="${PYTHON:-python3}"

echo "=== unit tests ==="
$PYTHON -m pytest tests/ -q

echo
echo "=== generating the synthetic dataset ==="
if [[ ! -f data/raw/enso/nino34.csv ]]; then
  $PYTHON tests/make_synthetic_dataset.py
else
  echo "(already present; delete data/raw to regenerate)"
fi

echo
echo "=== running the workflow ==="
bash workflow/run_all.sh --skip-download

echo
echo "=== checking the results ==="
$PYTHON - <<'PY'
import json, sys
from pathlib import Path

fail = []


def check(ok, message):
    print(("  ok   " if ok else "  FAIL ") + message)
    if not ok:
        fail.append(message)


nums = json.loads(Path("data/processed/paper_numbers.json").read_text())

# 1. the event catalogue
supers = nums["events"]["super_events"]
check(supers == [1972, 1982, 1997, 2015],
      f"super events recovered: {supers}")
check(nums["events"]["n_reference"] >= 5,
      f"reference sample size: {nums['events']['n_reference']}")

# 2. the imposed teleconnection pattern
import numpy as np
import xarray as xr
ds = xr.open_dataset(
    "data/processed/significance/significance_dis_super_obsclim_histsoc.nc")
c = ds["composite"].sel(season="DJF01")
for name, lat, lon, sign in [
        ("eastern Australia", -20, 142, -1),
        ("Amazon", 0, -60, -1),
        ("southern Africa", -20, 25, -1),
        ("southern USA", 32, -100, +1),
        ("central Asia", 40, 60, +1)]:
    v = float(c.sel(lat=lat, lon=lon, method="nearest"))
    check(np.sign(v) == sign and abs(v) > 0.2,
          f"{name}: composite {v:+.2f} (expected sign {sign:+d})")

# 3. the imposed response lags
lags = nums["lag_cascade"]["oni_xcorr_peak_lag"]
expected = {"qtot": 0, "rootmoist": 1, "dis": 2, "qr": 3, "tws": 5,
            "groundwstor": 6}
for var, want in expected.items():
    got = lags.get(var)
    check(got is not None and abs(got - want) <= 1,
          f"{var}: ONI cross-correlation lag {got} (imposed {want})")
order = [v for v in nums["lag_cascade"]["ordering"] if v in expected]
check(order.index("qtot") < order.index("qr") < order.index("groundwstor"),
      f"cascade ordering: {order}")

# 4. outputs exist
for pattern, least in [("results/figures/fig0*.png", 6),
                       ("results/tables/table_*.csv", 10)]:
    n = len(list(Path().glob(pattern)))
    check(n >= least, f"{pattern}: {n} file(s)")
check(Path("paper/manuscript.md").exists(), "manuscript rendered")

print()
if fail:
    print(f"SMOKE TEST FAILED: {len(fail)} check(s)")
    sys.exit(1)
print("SMOKE TEST PASSED - the workflow recovers what was put in")
PY
