# Climate Data and Indices

```{objectives}
- Understand the data sources and simulations
- Know the available climate indices
```

## Data Sources and Simulations

The data for this analysis comes from three long-term simulations produced using the Norwegian Earth System Model (NorESM1-F). These simulations span hundreds of years and include both historical and pre-industrial climate conditions.

1. **Historical Simulation with SLOW Forcing (850-2005 AD):** Accounts for low solar variability, transient volcanic activity, and anthropogenic effects.
2. **Historical Simulation with HIGH Forcing (850-2005 AD):** Similar to the first but uses high solar variability.
3. **Pre-Industrial Control Simulation (PICNTL):** Simulates 1,000 years with constant solar and volcanic forcing, and pre-industrial (1850-level) anthropogenic forcing.

## Climate Indices

This repository works with a wide range of climate indices, which are used to study various aspects of climate variability:

- **Global Surface Temperature Indices:** `glTSann`, `nhTSann`, `shTSann`
- **Sea Surface Temperature (SST) Indices:** `amoSSTann`, `satlSSTann`, `ensoSSTjfm`
- **Sea Ice Concentration (SIC) Indices:** `nhSICmar`, `nhSICsep`, `shSICmar`, `shSICsep`
- **Precipitation Indices:** `nchinaPRjja`, `yrvPRjja`, `ismPRjja`
- **Pressure Indices:** `naoPSLjfm`, `eapPSLjfm`, `scpPSLjfm`
- **Ocean Circulation Indices:** `AMOCann`, `traBO`, `traBS`

A detailed description of each index is provided below.

## Default Predictors

These indices provide historical data used to build the model for predicting future conditions:

**Atmospheric Pressure:**
- `naoPSLjfm`: North Atlantic Oscillation (NAO) pressure anomalies (December-February, boreal winter)
- `eapPSLjfm`: East Atlantic Pattern (EAP) pressure anomalies (December-February, boreal winter)
- `scpPSLjfm`: Scandinavian Pattern (SCP) pressure anomalies (December-February, boreal winter)
- `eurPSLjfm`: Euro-Mediterranean Pattern (EU) pressure anomalies (December-February, boreal winter)

**Sea Ice Concentration:**
- `nhSICmar`: Northern Hemisphere Sea Ice Concentration (March)
- `nhSICsep`: Northern Hemisphere Sea Ice Concentration (September)
- `shSICmar`: Southern Hemisphere Sea Ice Concentration (March)
- `shSICsep`: Southern Hemisphere Sea Ice Concentration (September)
- `atlSICmar`: Atlantic Sea Ice Concentration (March)
- `atlSICsep`: Atlantic Sea Ice Concentration (September)

**Ocean Circulation:**
- `AMOCann`: Atlantic Meridional Overturning Circulation (AMOC) anomaly (annual)

**Atmospheric Circulation (November-December-January, boreal winter):**
- `stratZ50ndj01`: Stratospheric 50 hPa Geopotential Height anomalies (November)
- `stratZ50ndj02`: Stratospheric 50 hPa Geopotential Height anomalies (December)
- `stratZ50ndj03`: Stratospheric 50 hPa Geopotential Height anomalies (January)
- `stratZ50ndj04`: Stratospheric 50 hPa Geopotential Height anomalies (December-January-February)

**Ocean Temperatures:**
- `amoSSTann`: (included as both a predictor and predictand) Annual AMO SST anomaly
- `amoSSTmjjaso`: AMO SST anomaly (May-June-July-August-September, boreal summer)
- `amo1`, `amo2`, `amo3`: These represent additional principal components of the AMO

## Other Available Predictors

| Variable | Description |
|:---------|:------------|
| `glTSann` | annual mean global surface temperature (degC) |
| `nhTSann` | annual mean northern hemisphere surface temperature (degC) |
| `shTSann` | annual mean southern hemisphere surface temperature (degC) |
| `cetTSann` | annual mean Central England surface temperature (degC) |
| `amoSSTann` | annual mean Atlantic (0-60N; 75W-7.5W) sea surface temperature (SST; degC) |
| `satlSSTann` | annual mean South Atlantic (60S-0S; 60W-15E) SST (degC) |
| `ensoSSTjfm` | winter (JFM) mean ENSO (Nino3.4 region) SST index (degC) |
| `neurTSjfm` | winter (JFM) mean northern Europe surface temperature index (degC) |
| `seurTSjfm` | winter (JFM) mean southern Europe surface temperature index (degC) |
| `arcTSann` | annual mean Arctic (north of 60N) surface temperature (degC) |
| `nhSICmar` | northern hemisphere sea ice concentration (%) for March |
| `nhSICsep` | northern hemisphere sea ice concentration (%) for September |
| `shSICmar` | southern hemisphere sea ice concentration (%) for March |
| `shSICsep` | southern hemisphere sea ice concentration (%) for September |
| `nchinaPRjja` | summer (JJA) precipitation over northern China |
| `yrvPRjja` | summer (JJA) precipitation over Yangtze River region in China |
| `ismPRjja` | summer (JJA) precipitation over Indian Monsoon region in Asia |
| `wnorPRjfm` | winter (JFM) precipitation for western Norway |
| `naoPSLjfm` | leading principal component (PC1) from North Atlantic sea level pressure |
| `eapPSLjfm` | second principal component (PC2) from North Atlantic sea level pressure |
| `scpPSLjfm` | third principal component (PC3) from North Atlantic sea level pressure |
| `eurPSLjfm` | fourth principal component (PC4) from North Atlantic sea level pressure |
| `solFORC` | annual total solar irradiance index (W m-2) (only for shigh and slow simulations) |
| `samPSLann` | annular mean Southern annular mode (SAM) pressure index |
| `atlSICmar` | March sea ice concentration (%) for the Atlantic sector |
| `atlSICsep` | September sea ice concentration (%) for the Atlantic sector |
| `pdoSSTjfm` | Pacific Decadal Oscillation (PDO) index as leading PC of North Pacific SST |
| `nppc2SSTjfm` | Second PC of North Pacific SST |
| `nppc3SSTjfm` | Third PC of North Pacific SST |
| `AMOCann` | Annual mean Atlantic meridional overturning circulation (Sv) |
| `traBO` | Volume transport (Sv) Barents Opening |
| `traBS` | Volume transport (Sv) Bering Strait |
| `traCA` | Volume transport (Sv) Canadian Archipelago |
| `traDS` | Volume transport (Sv) Denmark Strait |
| `traDP` | Volume transport (Sv) Drake Passage |
| `traEC` | Volume transport (Sv) English Channel |
| `traEU` | Volume transport (Sv) Equatorial Undercurrent |
| `traFSC` | Volume transport (Sv) Faroe Shetland Channel |
| `traFB` | Volume transport (Sv) Florida - Bahamas |
| `traFS` | Volume transport (Sv) Fram Strait |
| `traIFC` | Volume transport (Sv) Iceland Faroe Channel |
| `traIT` | Volume transport (Sv) Indonesian Throughflow |
| `traMC` | Volume transport (Sv) Mozambique Channel |
| `traTLS` | Volume transport (Sv) Taiwan and Luzon Straits |
| `traWP` | Volume transport (Sv) Windward Passage |
| `alpi` | Aleutian Low Pressure Index |
| `mhfATL30` | meridional heat flux Atlantic Ocean at 30N |
| `mhfATL45` | meridional heat flux Atlantic Ocean at 45N |
| `mhfATL60` | meridional heat flux Atlantic Ocean at 60N |
| `mhfIP30` | meridional heat flux Indian-Pacific Ocean at 30N |
| `tripol1` | annual mean SST index for the tripole 1 Pacific region |
| `tripol2` | annual mean SST index for the tripole 2 Pacific region |
| `tripol3` | annual mean SST index for the tripole 3 Pacific region |
| `amo1` | annual mean SST index for Atlantic Ocean 0-30N |
| `amo2` | annual mean SST index for Atlantic Ocean 30-45N |
| `amo3` | annual mean SST index for Atlantic Ocean 45-60N |
| `labSSTann` | annual mean SST index (degC) over the Labrador Sea |
| `ormenSSTann` | annual mean SST index (degC) over the Ormen Lange region |
| `gyreSSTann` | annual mean SST index (degC) over the Subpolar Gyre |
| `ginwSSTann` | annual mean SST index (degC) over the western Nordic Seas |
| `glomaSSTann` | annual mean AMO-index with global mean (excluding the Atlantic) removed |
| `amoSSTmjjaso` | extended summer AMO-index |
| `stratZ50ndj01` | leading PC for geopotential height at 50 hPa for NH north of 20N |
| `stratZ50ndj02` | second PC for geopotential height at 50 hPa for NH north of 20N |
| `stratZ50ndj03` | third PC for geopotential height at 50 hPa for NH north of 20N |
| `stratZ50ndj04` | fourth PC for geopotential height at 50 hPa for NH north of 20N |

```{keypoints}
- Data comes from NorESM1-F simulations: SLOW, HIGH, and PICNTL
- 65 climate indices are available covering temperature, pressure, ocean, and ice
- Indices can be used as predictors or prediction targets
```
