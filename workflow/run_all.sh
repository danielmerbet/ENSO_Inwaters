#!/usr/bin/env bash
# =====================================================================
# ENSO_Inwaters - run the whole workflow, in order.
#
#   bash workflow/run_all.sh                       # production config
#   ENSO_INWATERS_CONFIG=config/config_synthetic.yaml \
#       bash workflow/run_all.sh                   # synthetic smoke test
#   bash workflow/run_all.sh --from 07             # resume from a step
#   bash workflow/run_all.sh --only 08,09          # just these steps
#   bash workflow/run_all.sh --skip-download       # data already staged
#
# Every step is idempotent: outputs that exist are skipped unless
# --overwrite is passed through with EXTRA_ARGS.
# =====================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
CONFIG="${ENSO_INWATERS_CONFIG:-config/config.yaml}"
export ENSO_INWATERS_CONFIG="$CONFIG"
PYTHON="${PYTHON:-python3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

FROM=""
ONLY=""
SKIP_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --overwrite) EXTRA_ARGS="$EXTRA_ARGS --overwrite"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

MAIN_CLIM=$($PYTHON - <<'PY'
import sys; sys.path.insert(0, "src")
from enso_inwaters import load_config
c = load_config()
print(c["isimip.main_scenario.climate_scenario"])
PY
)
MAIN_SOC=$($PYTHON - <<'PY'
import sys; sys.path.insert(0, "src")
from enso_inwaters import load_config
c = load_config()
print(c["isimip.main_scenario.soc_scenario"])
PY
)

# step id | script | extra arguments
STEPS=(
  "00|00_check_environment.py|"
  "01|01_fetch_enso_indices.py|DOWNLOAD"
  "02|02_define_enso_events.py|"
  "03|03_fetch_isimip3a.py|DOWNLOAD"
  "04|04_fetch_auxiliary_data.py|DOWNLOAD"
  "05|05_preprocess_harmonize.py|"
  "06|06_anomalies_and_indices.py|"
  "07|07_event_composites.py|"
  "07b|07_event_composites.py|--climate-scenario counterclim --soc-scenario histsoc"
  "07c|07_event_composites.py|--climate-scenario ${MAIN_CLIM} --soc-scenario nosoc"
  "08|08_significance_and_agreement.py|--seasons-only"
  "09|09_nonlinearity.py|"
  "10|10_lag_cascade.py|"
  "11|11_basin_aggregation.py|"
  "12|12_lake_analysis.py|"
  "13|13_extremes.py|"
  "14|14_scenario_contrasts.py|"
  "15|15_exposure.py|"
  "16|16_validation.py|"
  "17|17_uncertainty.py|"
  "20|20_figure1_enso_events.py|"
  "21|21_figure2_global_composites.py|"
  "22|22_figure3_nonlinearity.py|"
  "23|23_figure4_lag_cascade.py|"
  "24|24_figure5_basins_lakes.py|"
  "25|25_figure6_impacts.py|"
  "26|26_supplementary_figures.py|"
  "30|30_assemble_tables.py|"
  "31|31_paper_numbers.py|"
  "40|40_render_manuscript.py|"
)

started=0
[[ -z "$FROM" ]] && started=1
failed=()
t_all=$SECONDS

echo "================================================================"
echo " ENSO_Inwaters workflow"
echo " configuration : $CONFIG"
echo " main scenario : ${MAIN_CLIM}/${MAIN_SOC}"
echo " started       : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

for entry in "${STEPS[@]}"; do
  IFS='|' read -r id script extra <<< "$entry"

  [[ -n "$FROM" && "$id" == "$FROM" ]] && started=1
  [[ $started -eq 0 ]] && continue
  if [[ -n "$ONLY" ]]; then
    [[ ",$ONLY," == *",$id,"* ]] || continue
  fi
  if [[ "$extra" == "DOWNLOAD" ]]; then
    extra=""
    if [[ $SKIP_DOWNLOAD -eq 1 ]]; then
      echo "--- step $id ($script) skipped (--skip-download)"
      continue
    fi
  fi

  echo ""
  echo "--- step $id : $script $extra"
  t0=$SECONDS
  if $PYTHON "workflow/$script" $extra $EXTRA_ARGS; then
    echo "--- step $id OK ($((SECONDS - t0)) s)"
  else
    rc=$?
    echo "--- step $id FAILED (exit $rc) - see logs/${script%.py}.log" >&2
    failed+=("$id:$script")
    # The download steps commonly fail on a machine without access to the
    # data hosts; the analysis steps must not be run on partial data, but
    # the figure steps degrade gracefully, so keep going and report at the end.
  fi
done

echo ""
echo "================================================================"
echo " total wall time: $((SECONDS - t_all)) s"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo " FAILED steps: ${failed[*]}"
  echo " (logs/ holds one log file per step)"
  exit 1
fi
echo " all steps completed"
echo "================================================================"
