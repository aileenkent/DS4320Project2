# DS 4320 Project 2: Predicting Fire Magnitude

### Executive Summary:

This is the complete implementation of a wildfire prediciton system built on a MongoDB document database sourced from the USDA Forest Service Fire Program Analysis Fire-Occurance Database. This project tackles the specific problem of predicing the final size in acres and containment duration of a wildfire in the US at the moment of ignition. The project usings weather covariates, fuel model, location, and cause code as features. This project contains tha pipeline that queres MongoDB Atlas, engineers featues, trains gradient-boosted models, and produces visualizations.

### Name: 
Aileen Kent

### NetID:
sbx3sw

### DOI:

### Press Release:
[link]

### Pipeline:
[link]

### License: 

GNU General Public License v3.0 [link](https://github.com/aileenkent/DS4320Project2/blob/498d2d2756bc1ed562846eef81c2c3b5a4c7303f/LICENSE)

## Problem Definition: 

### Initial General Problem:
Predicting wildfire risk

### Specific Refined Problem:
Predicting the number of acres burned and containment duration of US wildfires at the moment of ignition by applying a regression model based on weather conditions, vegitation tyep, ignition source, etc. of historical fire incident reports

### Motivation: 
Wildfires are some of the most destructive natural disasers. Many places in the US that are frequently hit with wildfires often struggle becuase they have a limited amount of resources to contain those fires. This limited amount of resources often causes issues becuase in order to prevent mass destruction, the government must pre-position their resources so that the fire can be quickly contained. Yet if the strength and growth potential of a fire is unknown, there could very easily be the case of giving a lot of resources to a smaller fire while larger ones do not have enough containment resources dedicated to them. A model preicitng fire growth and containment time could serve as a guiding for where to put resources.

### Rationale:
The refinment was made to decide what exactly about wildfire risk should be looked into. There are many different aspects of wildfire risk that could be explored, including what areas are at high risk for a wildfire to start or the direction of the wildfire. However it is not feasible to build an accurate model to cover every single aspect of wildfires. The size of the fire and containment duration were selected in order to have results that could help guide actions and focus on lowering the uncertainty of resource allocation during a wildfire. The goal was to provide some benefit to those who would be suffering or in dange from the wildfire and size and duration are the factors that determine how long people's lives are disrupted.

### Press Release Headline:
**ML Model predict Wildfire size at Ignition, Gives Citical Hours to Pre-Position** [link]

## Domain Exposition

### Terminology:
 Term / KPI | Definition |
|------------|-----------|
| **Acre-Burn Rate** | Acres consumed per hour during active fire spread; key operational KPI |
| **Containment (%)** | Percentage of the fire perimeter that has an effective control line; 100% = fully contained |
| **Containment Duration** | Elapsed days from discovery date to 100% containment; primary prediction target |
| **Final Fire Size (acres)** | Total acres within the fire perimeter at 100% containment; primary regression target |
| **Incident Complexity (Type 1–5)** | NIMS classification of resource requirements; Type 1 = most complex, Type 5 = smallest |
| **Initial Attack** | First suppression response dispatched immediately after discovery; its success largely determines final size |
| **Fuel Model** | NFFL/Scott-Burgan classification of surface and canopy fuel type (e.g., grass, shrub, timber litter) |
| **ERC (Energy Release Component)** | NFDRS index measuring the potential heat energy released per unit area; high ERC → high fire intensity |
| **KBDI (Keetch-Byram Drought Index)** | Soil moisture deficit index; higher values indicate drier fuels and greater fire spread potential |
| **Ignition Cause** | ICS-209 classification of fire origin: lightning, debris burning, equipment use, arson, etc. |
| **NIFC** | National Interagency Fire Center; federal clearinghouse for wildfire statistics and incident data |
| **FPA FOD** | USDA Forest Service Fire-Program Analysis Fire-Occurrence Database; 1992–present SQLite/JSON fire records |
| **ICS-209** | Incident Command System form 209; the standardized situation report filed for every large fire |
| **RMSE (log-acres)** | Root Mean Squared Error on log-transformed final size; primary regression evaluation KPI |
| **Macro F1** | Macro-averaged F1 score across containment-duration classes; primary classification KPI |

### Domain:
This project is in the domain of environmental science, with a specific focus on wildland fire management. There is a coordinated Incident Command Center that assess the complexity and potential resources needed at all levels of the government. These assessments are typically based off of a combination of fuel, weather, and toporapy.

### Background Reading Folder
[link](https://drive.google.com/drive/folders/1fm5EBDBnkbeFRDTooP5Vqit3n_lqSoNZ?usp=sharing)

| Title | Brief Description | Link |
|-------|------------------|-----------|
|Short et al. (2022) — *Spatial database of wildfires in the US, 1992–2020* | The FPA FOD dataset paper; explains data collection, fields, and known biases | [link](https://drive.google.com/file/d/19BKvQEiXXr2-ODt3YObOs43u_m4Mp_jT/view?usp=sharing) |
| Preisler et al. (2004) — *Probability-based models for wildland fire occurrence* | Foundational statistical framework for ignition and spread modeling | [link](https://drive.google.com/file/d/12S7TlNk0UXr0z1Aql6xAwc10NTZ0E5Ji/view?usp=sharing) |
| Jain et al. (2020) — *A review of ML for wildfire risk prediction* | Survey of 50+ ML approaches applied to fire size, spread, and risk | [link](https://drive.google.com/file/d/1KwHMg1K6sIvRphVhMudTc0HiNI7F8dyV/view?usp=sharing) |
| NIFC (2024) — *Wildland Fire Statistics* | Annual federal summary of acres burned, fire counts, and suppression costs | [link](https://drive.google.com/file/d/18tlNk6XKyTUgXhgrgAufsmHlsVQAGy7Y/view?usp=sharing) |
| Scott & Burgan (2005) — *Standard Fire Behavior Fuel Models* | USDA technical reference defining the 40 fuel models used in fire behavior modeling | [link](https://drive.google.com/file/d/1_g_z1C0ub20ikCheNFpT98EaCWv4PNZQ/view?usp=sharing) |

## Data Creation:

The source of the data is the USDA Forest Service Fire Program Analysis Fire-Occurrence Database (FPA FOD), which is maintained by karen Short and publically available at the Forest Service Research Data Archive. The dataset contains every reported wildfire in the US from 1992-2020, with the records being compiled from the fire-reporting systems at the various levels of government. The data was orginally found in an SQLite format, but I exported them to JSON to be able to use MongoDB Atlas. The data includes the tatitude, longitude, discovery date, containment date, final size, and ingition cause code for each fire.

| File | Description | Link |
|------|-------------|------|
| `ds4320extract.ipynb` | Reads the raw SQLite FPA FOD file, filters to CONUS records with valid geometry, exports one JSON document per fire incident | [link](https://github.com/aileenkent/DS4320Project2/blob/10b7392ed7752b4ea528fce943e8840bad8ba051/ds4320extract.ipynb) |
| `ds4320fetch.ipynb` | Calls the Copernicus CDS API to download ERA5 daily reanalysis tiles for each year; samples grid cells at each fire's coordinates and discovery date | [link](https://github.com/aileenkent/DS4320Project2/blob/10b7392ed7752b4ea528fce943e8840bad8ba051/ds4320fetch.ipynb) |
| `ds4320join.ipynb` | Merges fire records with weather covariates on (fire_id, date), adds computed fields (containment_days, size_class), and upserts documents into MongoDB Atlas | [link](https://github.com/aileenkent/DS4320Project2/blob/10b7392ed7752b4ea528fce943e8840bad8ba051/ds4320join.ipynb) |
| `ds4320validate.ipynb` | Runs post-load validation: checks field completeness, flags outliers (runtime > 3 years, size < 0), and writes a quality report to `logs/` | [link](https://github.com/aileenkent/DS4320Project2/blob/10b7392ed7752b4ea528fce943e8840bad8ba051/ds4320validate.ipynb) |

### Bias Identification:
Bias could be introducted into the data collection process in a few ways, but the primary concern for this datset is reporting bias. Fires in public land or very populated areas are more likely to be reported than those in say remot areas or solely on private land. There is also the concern of the sparser weather data in the years prior to 2000 becuase the data gaps in stations increasing uncertainty.

### Bias Mitigation: 
In order to mitigate the likely reporting bias, the model is trained on a weighted filtered subset of the dataset. This subset is a group of agencies with consistent reporting history. The weighting is inverse to the agency reporting completeness. In order to mitigate the uncertainty in the data prior to 2000, there is a feature added called ERA5 ensemble spread, which allows the model to discount the records missing data.

### Rationale:
There were a few judgement calls in creating the dataset and attempting to mitigate bias, but likely the most cruical one was the definition of the prediction targets. The final size of the fire was log-transformed to prevent allowing the rare case of fires over 10,000 acres to have too large of an emphasize on the model. The containment duration was simplified into four groups (<1 day, 1-7 days, 8-30 days, and > 30 days) in order to convery the relevant order-of-magnitude and not have the model get stuck on the precise day.

## Metadata:

### Implicit Schema:
This is the general schema of my document structure, however some of the documents do not have all of the optional fields. Dates are stored as `Date` objects, numberic fields use `double`, interger counts use `int32`. Nested sub-documents group related covariates are not flattened.
```
{
  "_id"             : <string>   REQUIRED  — FPA FOD fire_id (e.g. "CA-LPF-002345")
  "fire_name"       : <string>   REQUIRED  — official incident name, UPPER CASE
  "discovery_date"  : <ISODate>  REQUIRED  — UTC datetime of fire discovery
  "containment_date": <ISODate>  OPTIONAL  — UTC datetime of 100% containment
  "final_size_acres": <double>   REQUIRED  — total acres at containment (>= 0)
  "containment_days": <double>   OPTIONAL  — (containment_date - discovery_date).days
  "size_log"        : <double>   REQUIRED  — log10(final_size_acres + 1)
  "size_class"      : <string>   REQUIRED  — one of ["A","B","C","D","E","F","G"]
  "ignition_cause"  : <string>   REQUIRED  — ICS code: "Lightning"|"Human"|"Unknown"
  "location": {                  REQUIRED
    "type"        : "Point"      REQUIRED  — GeoJSON type (always "Point")
    "coordinates" : [<lon>, <lat>] REQUIRED — WGS-84, longitude first
  }
  "state"           : <string>   REQUIRED  — 2-letter USPS abbreviation
  "gacc_region"     : <string>   OPTIONAL  — NIFC Geographic Area Coordination Center code
  "weather": {                   OPTIONAL  — ERA5 covariates at ignition
    "max_temp_c"      : <double> — daily max 2 m temperature (°C)
    "wind_speed_ms"   : <double> — daily mean 10 m wind speed (m/s)
    "relative_humidity": <double> — daily mean RH (%)
    "kbdi"            : <double> — Keetch-Byram Drought Index (0–800)
    "erc"             : <double> — Energy Release Component
  }
  "fuel_model"      : <string>   OPTIONAL  — Scott-Burgan 40 fuel model code
  "incident_type"   : <int>      OPTIONAL  — NIMS complexity type 1–5
}
```

### Data Summary: 
| Table / Collection | Description |
|--------------------|-------------|
| `wildfires` (MongoDB) | One document per fire incident; primary modelling dataset |
| `fires_raw.csv` | Unprocessed FPA FOD export — one row per fire, original USDA fields |
| `weather_covariates.csv` | ERA5-joined weather features, one row per fire_id |
| `fires_final.csv` | Merged, cleaned, feature-engineered dataset used for model training |

### Data Dictionary:
| Feature | Data Type | Description | Example |
|---------|-----------|-------------|--------|
| `_id` | string | Unique FPA FOD fire identifier | `"CA-LPF-002345"` |
| `fire_name` | string | Official NWCG incident name | `"BOBCAT"` |
| `discovery_date` | ISODate | UTC timestamp of first report | `2020-09-06T00:00:00Z` |
| `containment_date` | ISODate | UTC timestamp of 100% containment | `2020-11-16T00:00:00Z` |
| `final_size_acres` | double | Acres within perimeter at containment | `115796.0` |
| `containment_days` | double | Days from discovery to containment | `71.0` |
| `size_log` | double | log₁₀(final_size_acres + 1) | `5.064` |
| `size_class` | string | NWCG size class A–G | `"G"` |
| `ignition_cause` | string | Broad cause category | `"Human"` |
| `location.coordinates` | array[double] | [longitude, latitude] in WGS-84 | `[-117.89, 34.20]` |
| `state` | string | 2-letter state abbreviation | `"CA"` |
| `gacc_region` | string | NIFC Geographic Area code | `"OSCC"` |
| `weather.max_temp_c` | double | Daily max 2 m air temperature (°C) | `38.4` |
| `weather.wind_speed_ms` | double | Daily mean 10 m wind speed (m/s) | `6.2` |
| `weather.relative_humidity` | double | Daily mean relative humidity (%) | `14.5` |
| `weather.kbdi` | double | Keetch-Byram Drought Index (0–800) | `612` |
| `weather.erc` | double | Energy Release Component index | `87.3` |
| `fuel_model` | string | Scott-Burgan 40 fuel model code | `"TL3"` |
| `incident_type` | int | NIMS complexity type (1=largest) | `1` |

| Feature | Min | Max | Mean | Std Dev | Notes on Uncertainty |
|---------|-----|-----|------|---------|---------------------|
| `final_size_acres` | 0.1 | ~640,000 | ~1,200 | ~12,000 | Log-normal; measurement error ±5% from aerial perimeter mapping |
| `containment_days` | 0 | ~400 | ~7 | ~20 | Reporting lag of 1–3 days common; open-ended fires coded as missing |
| `size_log` | 0 | ~5.8 | ~1.9 | ~1.2 | Derived from `final_size_acres`; propagates same ±5% |
| `weather.max_temp_c` | -20 | 55 | 28 | 10 | ERA5 grid spacing ~31 km introduces ±2°C local uncertainty |
| `weather.wind_speed_ms` | 0 | 25 | 4.1 | 2.8 | ERA5 10 m wind ±0.8 m/s vs. station observations |
| `weather.relative_humidity` | 2 | 100 | 35 | 22 | ERA5 RH ±5–8% absolute in arid regimes |
| `weather.kbdi` | 0 | 800 | 310 | 210 | Computed from ERA5 precip and temp; cumulative drift ±30 index units |
| `weather.erc` | 0 | 120 | 45 | 25 | Sensitive to fuel model choice; ±10 units uncertainty when fuel model is missing |
