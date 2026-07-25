# Data Directory

This directory holds the dataset used by the analysis pipeline.
The dataset is **not committed** to this repository. Download it
from Mendeley Data as described below.

---

## Dataset Record

**Title:** Longitudinal Indoor Air Quality Dataset Collected Using a
Low-Cost Multi-Sensor IoT Monitoring Platform

**Author:** Siddique, Md Abubakar (2026)

**Repository:** Mendeley Data, V1

**DOI:** https://doi.org/10.17632/b5jvs7kykn.1

**License:** CC BY 4.0

---

## Download Instructions

1. Go to https://doi.org/10.17632/b5jvs7kykn.1
2. Open the **Research_Subsets** folder
3. Download `TVOC_Filtering_Subset.csv`
4. Place it in this `data/` directory
5. Confirm `DATA_PATH` in `TVOC_Complete_Final.py` matches the filename

---

## File Used by This Study

**`Research_Subsets/TVOC_Filtering_Subset.csv`**

| Column | Unit | Source sensor |
|---|---|---|
| Timestamp | datetime (5-second intervals) | System clock |
| Temperature (C) | °C | Bosch BME688 |
| Humidity (%) | % | Bosch BME688 |
| Pressure (hPa) | hPa | Bosch BME688 |
| Gas Resistance (Ohms) | Ohms | Bosch BME688 |
| TVOC (ppb) | ppb | Sensirion SGP30 |
| eCO2 (ppm) | ppm | Sensirion SGP30 |

**Preprocessing notes:**

- Row 2 (and a few subsequent rows) contains boot-sequence values
  where Temperature is missing and TVOC = 0, eCO2 = 400. These are
  removed in Cell 2 of the analysis script using `dropna(subset=['Temperature (C)'])`.
- TVOC values of 60,000 ppb are SGP30 firmware saturation artifacts
  and are forward-filled in preprocessing.
- Gas Resistance values of 1.02E+08 Ohms (102,400,000) represent
  the BME688 clean-air ceiling — these are valid readings, not errors.

---

## Complete Dataset

The **Master_Dataset** folder on the same Mendeley record contains
`air_quality_log.csv` with all 13 sensor channels across the full
16 calendar days of deployment. This master file supports two
additional companion studies:

| Study | Subset used |
|---|---|
| PM2.5 adaptive control | PM2.5, Temperature, Humidity, Pressure, Gas Resistance |
| IAQ classification | All 13 columns, Days 2–10 only |

---

## Citation

When using this dataset, cite:

```text
Siddique, Md Abubakar (2026). Longitudinal Indoor Air Quality Dataset
Collected Using a Low-Cost Multi-Sensor IoT Monitoring Platform.
Mendeley Data, V1. https://doi.org/10.17632/b5jvs7kykn.1
```
