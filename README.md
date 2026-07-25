# GPO-Tuned Kalman Filtering for Indoor TVOC Signal Enhancement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Research Code](https://img.shields.io/badge/status-research%20code-orange.svg)](#project-status)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-blue.svg)](https://doi.org/10.17632/b5jvs7kykn.1)

## Overview

This repository provides the analysis and deployment code for a
Gaussian-process-optimisation (GPO) tuned Kalman filtering framework
for indoor total volatile organic compound (TVOC) measurements. The
study uses a longitudinal indoor air-quality dataset collected with a
low-cost, Raspberry Pi 5-based sensing platform and evaluates signal
enhancement, environmental decoupling, temporal cross-validation, and
anomaly detection.

**Associated manuscript (under preparation):**

> *Enhancement of Indoor TVOC Measurements Using Kalman Filtering
> with Gaussian Process Hyperparameter Optimisation:
> A 15-Day Longitudinal Validation Study*
>
> Md Abubakar Siddique et al., IMSE, The University of Texas at El Paso

A journal DOI and Zenodo archive DOI will be added here once the
manuscript is accepted and the code release is finalised.

---

## Main Contributions

- Gaussian process optimisation of Kalman filter noise covariance parameters Q and R
- Balanced objective function that prevents degenerate MSE-minimisation solutions
- Environmental decoupling: non-significant innovation correlations with temperature and humidity
- 10-fold time-series cross-validation with bootstrap confidence intervals
- Wilcoxon signed-rank tests on a balanced signal-quality objective
- Ljung-Box innovation whiteness diagnostics
- Innovation-based TVOC anomaly detection
- Lightweight edge deployment on Raspberry Pi 5

---

## Repository Structure

```text
tvoc-gpo-kalman/
├── README.md
├── LICENSE
├── requirements.txt
├── TVOC_Complete_Final.py          ← complete analysis pipeline
├── data/
│   └── README.md                   ← dataset download instructions
├── deployment/
│   └── tvoc_edge_deployment.py     ← Raspberry Pi 5 real-time pipeline
├── figures/
│   ├── fig1_15day_overview.png
│   ├── fig2_gpo_convergence.png
│   ├── fig3_cv_results.png
│   ├── fig4_innovation_acf.png
│   ├── fig5_metrics_summary.png
│   ├── fig6_longitudinal_analysis.png
│   └── fig7_anomaly_detection.png
└── outputs/
    └── (generated tables and results — not committed)
```

---

## Dataset

The dataset is deposited in Mendeley Data. This repository contains
only the analysis code; the data file must be downloaded separately.

**Mendeley Data record:**
> Siddique, Md Abubakar (2026). *Longitudinal Indoor Air Quality Dataset
> Collected Using a Low-Cost Multi-Sensor IoT Monitoring Platform.*
> Mendeley Data, V1. https://doi.org/10.17632/b5jvs7kykn.1

**To use the dataset with this code:**

1. Go to https://doi.org/10.17632/b5jvs7kykn.1
2. Open the `Research_Subsets` folder
3. Download `TVOC_Filtering_Subset.csv`
4. Place the file in the `data/` directory of this repository
5. Confirm that `DATA_PATH` in `TVOC_Complete_Final.py` points to the
   correct filename

The subset contains: Timestamp, Temperature (C), Humidity (%),
Pressure (hPa), Gas Resistance (Ohms), TVOC (ppb), eCO2 (ppm).
The complete master dataset (all 13 channels) is available in the
`Master_Dataset` folder of the same Mendeley record.

Do not commit the CSV file to this repository.

---

## Hardware Platform

| Component | Model | Interface | Role |
|---|---|---|---|
| Edge processor | Raspberry Pi 5 (4 GB) | — | Data acquisition and pipeline |
| TVOC / eCO₂ sensor | Sensirion SGP30 | I²C | Primary gas sensing |
| Environmental sensor | Bosch BME688 | I²C | Temperature, humidity, pressure, gas resistance |

---

## Software Requirements

Python 3.10 or later. A standard laptop or desktop is sufficient for
the offline analysis. Raspberry Pi 5 is required only for the optional
edge deployment.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/YOUR-USERNAME/tvoc-gpo-kalman.git
cd tvoc-gpo-kalman
pip install -r requirements.txt
```

Download `TVOC_Filtering_Subset.csv` from Mendeley Data (see above),
place it in `data/`, then run:

```bash
python TVOC_Complete_Final.py
```

For interactive cell-by-cell execution, open in JupyterLab. Each
`# ── CELL N` comment marks a logical cell boundary:

```bash
jupyter lab
```

The GPO optimisation (Cell 5) and 10-fold cross-validation (Cell 9)
are the most computationally intensive stages.

---

## Analysis Workflow

1. Configuration and reproducibility settings (`SEED = 42`)
2. Data loading and timestamp parsing
3. Missing-value handling and saturation artifact capping
4. Descriptive and autocorrelation analysis
5. Filter function definitions
6. **GPO optimisation of Kalman parameters** (Cell 5)
7. Full-dataset filtering with all methods
8. Signal-quality and environmental-coupling assessment
9. **10-fold time-series cross-validation** (Cell 9)
10. Bootstrap confidence intervals and Wilcoxon tests
11. Ljung-Box innovation whiteness diagnostics
12. Final results table and publication figures
13. Anomaly-event detection and temporal characterisation
14. Figure generation (Cells 12–19)

---

## Reproducibility

All stochastic procedures use a fixed random seed (`SEED = 42`).
Chronological train-validation splits are preserved throughout.
Figures are saved at 300 dpi in both PNG and PDF formats.

Numerical results may vary slightly across operating systems and
package versions. Tag the exact commit used for manuscript submission.

---

## Project Status

This repository contains research code associated with a manuscript
currently in preparation. Code and results may change before the
final public release.

**Before archiving a versioned release:**

1. Verify the script runs to completion from a clean environment
2. Confirm the Mendeley DOI resolves: https://doi.org/10.17632/b5jvs7kykn.1
3. Remove any private paths, credentials, or local system details
4. Confirm every result is reproducible from the released code
5. Create a versioned GitHub release (e.g., `v1.0.0`)
6. Archive that release through Zenodo
7. Add the Zenodo DOI badge to this README

---

## Citation

### Software (before Zenodo DOI is available)

```text
Siddique, M.A. et al. (2026). tvoc-gpo-kalman: GPO-tuned Kalman
filtering for indoor TVOC signal enhancement [Computer software].
GitHub. https://github.com/YOUR-USERNAME/tvoc-gpo-kalman
Commit: <insert commit hash>
```

### Dataset

```text
Siddique, Md Abubakar (2026). Longitudinal Indoor Air Quality Dataset
Collected Using a Low-Cost Multi-Sensor IoT Monitoring Platform.
Mendeley Data, V1. https://doi.org/10.17632/b5jvs7kykn.1
```

### Article

The associated manuscript is in preparation. A citable reference
will be added here after acceptance and DOI assignment.

---

## License

Source code: [MIT License](LICENSE)

Dataset: CC BY 4.0 (as stated on the Mendeley Data record). The
software license does not apply to the dataset.

---

## Responsible Use

This code is for research and educational purposes only. It is not a
certified measurement or calibration system and should not be used as
the sole basis for health, safety, or regulatory decisions.

---

## Contact

Open a GitHub Issue for reproducibility problems or code questions.
Author contact details will be added after the manuscript is finalised
for public release.
