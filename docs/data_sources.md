# Data sources

Everything the workflow needs, where it comes from, and what has to be
done by hand. Steps 01, 03 and 04 fetch what can be fetched and print
instructions for the rest.

## 1. ENSO indices (step 01) - no registration

| Dataset | URL | Notes |
|---|---|---|
| Nino3.4 SST, ERSSTv5, 1854- | `https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.data` | primary input; reaches back before 1901 |
| Nino3, Nino4 | `.../nino3.long.data`, `.../nino4.long.data` | needed for the EP/CP classification |
| CPC Nino3.4 ASCII, 1950- | `https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt` | cross-check of the computed ONI |
| HadISST Nino3.4 | `https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data` | independent SST product, sensitivity test |

The workflow computes the ONI itself rather than downloading it, so that
the same base-period convention applies over the whole 1901-2019 record.
Step 02 checks its result against the CPC classification and warns on any
disagreement.

## 2. ISIMIP3a simulations (step 03) - free, registration only for bulk download

Repository: <https://data.isimip.org> - API at `/api/v1`, files on
`files.isimip.org`. Licence: CC BY 4.0 for most models; check each
model's terms in the repository metadata.

**Selection**

* simulation round `ISIMIP3a`, product `OutputData`
* climate forcing `gswp3-w5e5`, 0.5 degree, monthly
* climate scenarios `obsclim` (observed) and `counterclim` (no climate change)
* socio-economic scenarios `histsoc` (time-varying human influence) and
  `nosoc` (none)

**Sectors and variables**

| Sector | Models (subject to availability) | Variables |
|---|---|---|
| `water_global` | CWatM, H08, JULES-W1, MIROC-INTEG-LAND, WaterGAP2-2e | `dis`, `qtot`, `qr`, `tws`, `rootmoist`, `groundwstor`, `evap` |
| `lakes_global` | ALBM, GOTM, SIMSTRAT-UoG, VIC-LAKE, CLM4.5 | `surftemp`, `watertemp`, `lakeicefrac`, `icethick` |

Coverage is uneven -- not every model provides `qr` or `groundwstor`,
and the counterclim/nosoc combinations are sparser than obsclim/histsoc.
**Run `python workflow/03_fetch_isimip3a.py --discover-only` first** and
read `results/tables/table_S2_isimip_availability.csv` before committing
to the download. What is missing changes what the paper can claim.

**Size**: roughly 250-350 GB for the full selection at monthly
resolution. Restrict `isimip.sectors` in the config to work with less.

**Citation**: cite the ISIMIP3a protocol paper, the repository DOI, and
each impact model's own reference. The model references are in the
dataset metadata returned by the API and are worth capturing into the
manifest at download time.

## 3. Auxiliary data (step 04)

| Dataset | Used for | Access |
|---|---|---|
| GRDC Major River Basins | basin aggregation (step 11) | <https://grdc.bafg.de/products/basin_layers/major_river_basins/> - free, one-click data policy |
| HydroLAKES v1.0 | lake polygons and areas (step 12) | <https://www.hydrosheds.org/products/hydrolakes> - free |
| HYDE 3.2 | population 1901-2017, exposure (step 15) | <https://doi.org/10.17026/dans-25g-gez3> - free |
| ISIMIP land-sea mask | common analysis domain | on `files.isimip.org` |

## 4. Observations for evaluation (step 16)

| Dataset | Evaluates | Access |
|---|---|---|
| GRDC monthly discharge | simulated discharge composites | <https://portal.grdc.bafg.de/> - free, request-based; ask for stations with >= 40 years |
| GRACE/GRACE-FO JPL mascons RL06.1 | simulated TWS, 2002- (covers 2015/16) | <https://podaac.jpl.nasa.gov/> - free Earthdata login |
| ESA CCI Lakes v2 LSWT | simulated lake surface temperature | <https://climate.esa.int/en/projects/lakes/data/> - free |
| ERA5-Land monthly | independent runoff/soil-moisture cross-check | <https://cds.climate.copernicus.eu/> - free, CDS API |

The evaluation is the part of the paper most likely to be asked for in
review. The GRACE comparison is the only direct observational constraint
on the simulated storage response and it covers exactly one super event,
which is a limitation worth stating rather than hiding.

### ERA5-Land CDS request template

```python
import cdsapi
cdsapi.Client().retrieve(
    "reanalysis-era5-land-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ["runoff", "volumetric_soil_water_layer_1",
                     "volumetric_soil_water_layer_2"],
        "year": [str(y) for y in range(1950, 2020)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "format": "netcdf",
    },
    "data/raw/obs/era5_land_monthly.nc",
)
```

## 5. Network requirements

The download steps need outbound HTTPS to `data.isimip.org`,
`files.isimip.org`, `psl.noaa.gov`, `origin.cpc.ncep.noaa.gov` and the
auxiliary hosts above. `python workflow/00_check_environment.py` reports
which of these are reachable. On a cluster without direct internet, run
steps 01, 03 and 04 on a login node or stage the files by hand; steps
05 onwards need no network at all.
