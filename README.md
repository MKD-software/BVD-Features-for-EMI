# UTTOP Corrosion Test — BVD Feature Extraction

Accelerated corrosion experiment conducted in France by Mads. Three standard steel sensor probes (UTTOP design, no concrete) were corroded under controlled conditions: ~3% salt water electrolyte, 7 V applied voltage, 100 mA per sample. Electromechanical impedance (EMI) was measured repeatedly over the test duration.

The primary goal is to use **BVD circuit fitting as a feature extraction step** — mapping each impedance sweep to five physically interpretable parameters — and then use those features for machine learning-based corrosion/mass-loss prediction.

---

## Quickstart

### 1. Understand the data
Open [Plot_Corrosion_Data.ipynb](Plot_Corrosion_Data.ipynb) to visualise raw impedance sweeps, phase, and mass-loss over time for all three samples.

### 2. Apply the data correction (always required)
The raw CSVs contain a known scaling bug. Apply these corrections immediately after loading:

```python
df['Impedance (ohms)'] = df['Impedance (ohms)'] * 10e4
df['Phase (Radians)'] = df['Phase (Radians)'] * -100
```

### 3. Fit the BVD model
Open [BVD_AllSamples.ipynb](BVD_AllSamples.ipynb). This fits a 1st-order BVD equivalent circuit to the dominant resonant peak (~200 kHz) of the real-part impedance for every session and saves results to `BVD_Fits/`.

The BVD model: $Z = R_0 + (Z_{mot} \| Z_{sta})$ where $Z_{mot} = R_1 + j\omega L_1 + \frac{1}{j\omega C_1}$ and $Z_{sta} = \frac{1}{j\omega C_0}$

| Parameter | Physical meaning in corrosion context |
|-----------|--------------------------------------|
| `R0` | Series resistance (lead resistance, contact) |
| `C0` | Static capacitance of the PZT |
| `R1` | Mechanical damping — increases as surface roughens |
| `L1` | Acoustic mass — changes as corrosion product accumulates |
| `C1` | Mechanical compliance (1/stiffness) |
| `f_r = 1/(2π√L1C1)` | Resonant frequency — drops with mass increase (Sauerbrey analog) |

Fitted parameters are saved as CSVs in `BVD_Fits/` (see versioned files; `_final.csv` is the recommended result).

### 4. Run machine learning
Navigate to `ML/` and open [ML/ML_Alt.ipynb](ML/ML_Alt.ipynb) or [ML/PINN_ML.ipynb](ML/PINN_ML.ipynb). These take BVD parameters (and their relative changes from a healthy baseline) as inputs and predict cumulative mass loss.

---

## Repository Layout

```
UTTOPTest/
├── Corrosion_Dataset/          # Raw impedance measurements (time-series, per session)
├── UTTOP_Sweep/                # Baseline sweep (healthy state, free-air PZT reference)
├── BVD_Fits/                   # BVD parameter CSVs from fitting runs
├── BVD_Fits_Force/             # BVD fits for force-loading variant experiments
├── ML/                         # Machine learning notebooks and utilities
├── Extra/                      # Alternative modelling methods (Bhalla, Mason)
├── figures/                    # Output plots
├── Temp/                       # Scratch/temporary files
│
├── Plot_Initial_Data.ipynb     # Explore UTTOP_Sweep baseline data
├── Plot_Corrosion_Data.ipynb   # Explore corrosion dataset (START HERE)
├── BVD-Model.ipynb             # BVD circuit model derivation and sanity checks
├── BVD_AllSamples.ipynb        # Batch BVD fitting for all corrosion sessions
├── BVD_AllSamples_Force.ipynb  # BVD fitting for force-loading data
├── Fit_BVD_Free_PZT.ipynb      # BVD fit on free-air (unconstrained) PZT reference
│
├── bvd_utils.py                # Core BVD model, fitting helpers, peak preprocessing
├── bvd_fitting_v2.py           # Improved objective function with L1 penalty & weights
├── pzt_analysis_utils.py       # Impedance/admittance calculation, MATLAB data loading
└── apc_materials.json          # APC piezoelectric material constants
```

---

## Dataset Overview

### `Corrosion_Dataset/`
Time-series of impedance sweeps taken throughout the accelerated corrosion test.

- **Samples:** `sample_0`, `sample_5`, `sample_10` — three steel probes with PZT patches bonded to the surface.
- **Session folders:** `YYYY-MM-DD_HH-MM-SS_T.TC_H.HRH/` — date/time + ambient temperature + relative humidity recorded at measurement time.
- **File columns:** `Frequency (Hz)`, `Impedance (ohms)`*, `Phase (Radians)`*, `Temperature (C)`, `Humidity (%)`
- **`calibrated_mass_loss.csv`** — Faraday-law corrected cumulative mass loss per sample (ground truth for ML).
- **`mass_loss.csv` / `uptime.csv`** — raw gravimetric and uptime logs.

*Scaling correction required — see Quickstart step 2.*

### `UTTOP_Sweep/`
Single sweep taken before the corrosion test began. Provides the healthy-state baseline for each sample and a free-air PZT reference.

- **`dev7343_demods_0_sample_*.csv`** — raw Zurich Instruments demodulator output (2000 samples per chunk).
- **`dev7343_demods_0_sample_header_*.csv`** — metadata (timing, data quality, system configuration).

### `BVD_Fits/`
Versioned CSVs of fitted BVD parameters. The recommended file is `bvd_params_optimized_real_smaller_freq_final.csv`. Earlier versions (`V2`–`V12`) reflect iterative improvements to the fitting strategy (frequency window, penalty weights, etc.).

### `ML/`
Machine learning experiments using BVD features.

| Notebook | Description |
|----------|-------------|
| `ML_Alt.ipynb` | Main ML pipeline — BVD features → mass loss |
| `ML_Alt_V2.ipynb` | Updated version with relative-change features |
| `PINN_ML.ipynb` | Physics-Informed Neural Network with monotonicity and Sauerbrey constraints |
| `ML_Impedance_Based.ipynb` | Direct impedance (no BVD) baseline comparison |
| `ML_Alt_CNN.ipynb` | CNN on raw impedance spectra |
| `Physical_model.ipynb` | Analytical / closed-form corrosion model |
| `Calibrate_mass_loss.ipynb` | Faraday-law mass-loss calibration |
| `ML_Functions.py` | Shared ML utility functions |

---

## Critical Notes

- **Real-part fitting only:** The BVD fit targets `Re(Z)` because it is more sensitive to damage-driven changes than magnitude or phase representations.
- **Relative features for ML:** Always compute `ΔX/X₀` relative to each sample's own healthy baseline before training. This removes sensor-to-sensor offsets and isolates corrosion-driven drift.
- **Do not use absolute time as an ML input.** Corrosion rate was stable in this controlled experiment, but will vary in field conditions — time would not generalise.
- **Mass can only increase (monotonicity).** Enforce this constraint in any physics-informed model.
- **BVD trends observed:** `L1` and `R1` both decrease over the test duration, consistent with surface roughening (damping) and accumulation of corrosion products reducing effective acoustic mass.

---

## Alternative Methods (`Extra/`)
| Notebook | Description |
|----------|-------------|
| `Bhalla-method.ipynb` | Bhalla EMI structural health index approach |
| `Masons-Model.ipynb` | Mason equivalent circuit (more detailed piezo model) |
| `DMG-Requirements.ipynb` | Damage detection requirements analysis |
