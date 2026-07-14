"""
bvd_utils.py
------------
Shared helpers for simple 1st-order BVD circuit fitting.

Model: Z = R0 + (Z_mot ‖ Z_sta)
    Z_mot = R1 + jωL1 + 1/(jωC1)
    Z_sta = 1/(jωC0)
"""

import numpy as np
import matplotlib.pyplot as plt
import os, re, subprocess, sys
import pandas as pd
from datetime import datetime
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import r2_score

# ── Default bounds (wide fallback search space) ───────────────────────────────
DEFAULT_BOUNDS = [
    (0.1,   5000),   # R0  (Ω)
    (1e-15, 10),     # C0  (F)
    (0.01,  5000),   # R1  (Ω)
    (1e-15, 10),     # L1  (H)
    (1e-15, 10),     # C1  (F)
]

FOLDER_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_([\d.]+)C_([\d.]+)RH$'
)

# Pattern for force-data folders (format: YYYY-MM-DD_HH-MM)
FORCE_FOLDER_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})$')

# ── Circuit model ─────────────────────────────────────────────────────────────

def bvd_1st_order(frequency, R0, C0, R1, L1, C1):
    """Return complex impedance of a simple 1st-order BVD circuit."""
    omega  = 2 * np.pi * frequency
    Z_mot  = R1 + 1j*omega*L1 + 1.0/(1j*omega*C1)
    Z_sta  = 1.0/(1j*omega*C0)
    Z_para = 1.0/(1.0/Z_mot + 1.0/Z_sta)
    return R0 + Z_para

def resonant_frequency(L1, C1):
    return 1.0 / (2 * np.pi * np.sqrt(L1 * C1))

def quality_factor(f_res, L1, R1):
    return 2 * np.pi * f_res * L1 / R1

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_dominant_peak(frequency, z_complex, peak_span_hz=12000):
    """Smooth background, preserve dominant Re(Z) peak in ±peak_span_hz window."""
    peaks, props = find_peaks(np.real(z_complex), height=0)
    if len(peaks) == 0:
        return frequency, z_complex, None, (0, len(frequency) - 1)

    dominant_idx = peaks[np.argmax(props['peak_heights'])]
    f_peak = frequency[dominant_idx]

    idx_lo = max(0,                np.searchsorted(frequency, f_peak - peak_span_hz))
    idx_hi = min(len(frequency)-1, np.searchsorted(frequency, f_peak + peak_span_hz, side='right') - 1)

    n  = len(frequency)
    sw = max(11, n // 8)
    if sw % 2 == 0: sw += 1
    sw = min(sw, n - 2 if (n-2) % 2 == 1 else n-3)

    zr_sm = savgol_filter(np.real(z_complex), sw, polyorder=1)
    zi_sm = savgol_filter(np.imag(z_complex), sw, polyorder=1)

    tp     = max(3, (idx_hi - idx_lo) // 10)
    zr_out = zr_sm.copy()
    zi_out = zi_sm.copy()
    zr_out[idx_lo:idx_hi+1] = np.real(z_complex[idx_lo:idx_hi+1])
    zi_out[idx_lo:idx_hi+1] = np.imag(z_complex[idx_lo:idx_hi+1])

    for k in range(tp):
        i = idx_lo - tp + k
        if i < 0: continue
        a = 0.5*(1 - np.cos(np.pi*k/tp))
        zr_out[i] = (1-a)*zr_sm[i] + a*np.real(z_complex[i])
        zi_out[i] = (1-a)*zi_sm[i] + a*np.imag(z_complex[i])

    for k in range(tp):
        i = idx_hi + 1 + k
        if i >= n: break
        a = 0.5*(1 + np.cos(np.pi*k/tp))
        zr_out[i] = (1-a)*zr_sm[i] + a*np.real(z_complex[i])
        zi_out[i] = (1-a)*zi_sm[i] + a*np.imag(z_complex[i])

    return frequency, zr_out + 1j*zi_out, f_peak, (idx_lo, idx_hi)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_corrosion_data(base_path, start_freq, end_freq, samples,
                            start_date=None, end_date=None):
    """Scan dated folders in base_path and return a combined DataFrame.

    Parameters
    ----------
    start_date, end_date : str or datetime-like, optional
        Inclusive date/time bounds applied to folder timestamps.
    """
    all_data = []
    folders  = sorted(
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and FOLDER_PATTERN.match(f)
    )
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    print(f"Found {len(folders)} measurement folders.")

    for folder_name in folders:
        m = FOLDER_PATTERN.match(folder_name)
        ts_str, temp_str, hum_str = m.groups()
        timestamp   = datetime.strptime(ts_str, '%Y-%m-%d_%H-%M-%S')

        if start_date is not None and timestamp < start_date:
            continue
        if end_date is not None and timestamp > end_date:
            continue

        temperature = float(temp_str)
        humidity    = float(hum_str)
        folder_path = os.path.join(base_path, folder_name)

        for sample_name in samples:
            path = os.path.join(folder_path, f"{sample_name}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['condition']     = f"{temperature}°C, {humidity}%RH"
            df['timestamp']     = timestamp
            df['folder']        = folder_name
            df['sample']        = sample_name
            df['temperature_C'] = temperature
            df['humidity_RH']   = humidity
            df['Impedance (ohms)'] = df['Impedance (ohms)'] * 10e4
            df['Phase (Radians)']  = df['Phase (Radians)'] * -100
            df['Z_real'] = df['Impedance (ohms)'] * np.cos(df['Phase (Radians)'])
            df['Z_imag'] = df['Impedance (ohms)'] * np.sin(df['Phase (Radians)'])
            df_f = df[(df['Frequency (Hz)'] >= start_freq) & (df['Frequency (Hz)'] <= end_freq)]
            if not df_f.empty:
                all_data.append(df_f)

    if not all_data:
        return pd.DataFrame()
    combined = pd.concat(all_data, ignore_index=True)
    combined.sort_values(['timestamp', 'sample'], inplace=True)
    return combined

# ── Data loading for Force data ──────────────────────────────────────────────────────────────
def list_force_folders(base_path, start_date=None, end_date=None):
    """Return a sorted list of (folder_name, timestamp) tuples for force-data folders.

    Applies optional inclusive date filtering. Call this once to discover all
    folders, then iterate with load_single_force_folder to process them one at
    a time without loading everything into memory.
    """
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    result = []
    for folder_name in sorted(os.listdir(base_path)):
        if not os.path.isdir(os.path.join(base_path, folder_name)):
            continue
        m = FORCE_FOLDER_PATTERN.match(folder_name)
        if not m:
            continue
        timestamp = datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M')
        if start_date is not None and timestamp < start_date:
            continue
        if end_date is not None and timestamp > end_date:
            continue
        result.append((folder_name, timestamp))
    return result


def load_single_force_folder(base_path, folder_name, start_freq, end_freq, samples):
    """Load force data for ONE folder and return a DataFrame.

    Loads only the specified samples within the frequency window.
    Returns an empty DataFrame when no data is found.
    CSV columns are mislabeled (Impedance = Re(Z), Phase = Im(Z));
    this function remaps them so the rest of the pipeline still works.
    """
    m = FORCE_FOLDER_PATTERN.match(folder_name)
    if not m:
        return pd.DataFrame()

    timestamp   = datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M')
    folder_path = os.path.join(base_path, folder_name)
    all_data    = []

    for sample_name in samples:
        path = os.path.join(folder_path, f"{sample_name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['Z_real'] = df['Impedance (ohms)']
        df['Z_imag'] = df['Phase (Radians)']
        Z = df['Z_real'] + 1j * df['Z_imag']
        df['Impedance (ohms)'] = np.abs(Z)
        df['Phase (Radians)']  = np.angle(Z)
        df['timestamp'] = timestamp
        df['folder']    = folder_name
        df['sample']    = sample_name
        df_f = df[(df['Frequency (Hz)'] >= start_freq) & (df['Frequency (Hz)'] <= end_freq)]
        if not df_f.empty:
            all_data.append(df_f)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def load_all_force_data(base_path, start_freq, end_freq, samples,
                        start_date=None, end_date=None):
    """Scan dated folders in base_path and return a combined DataFrame.

    Folder format: YYYY-MM-DD_HH-MM

    NOTE:
    The CSV columns are mislabeled:
        - 'Impedance (ohms)'  -> actually Re(Z)
        - 'Phase (Radians)'   -> actually Im(Z)

    This function remaps them so the rest of the pipeline still works.
    """

    all_data = []

    folders = sorted(
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
        and FORCE_FOLDER_PATTERN.match(f)
    )

    if start_date is not None:
        start_date = pd.to_datetime(start_date)

    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    print(f"Found {len(folders)} measurement folders.")

    for folder_name in folders:

        m = FORCE_FOLDER_PATTERN.match(folder_name)
        ts_str = m.group(1)

        timestamp = datetime.strptime(ts_str, '%Y-%m-%d_%H-%M')

        if start_date is not None and timestamp < start_date:
            continue

        if end_date is not None and timestamp > end_date:
            continue

        folder_path = os.path.join(base_path, folder_name)

        for sample_name in samples:

            path = os.path.join(folder_path, f"{sample_name}.csv")

            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)

            # -----------------------------------------------------------------
            # CSV columns are mislabeled:
            # "Impedance (ohms)" = Re(Z)
            # "Phase (Radians)"  = Im(Z)
            # -----------------------------------------------------------------

            df['Z_real'] = df['Impedance (ohms)']
            df['Z_imag'] = df['Phase (Radians)']

            # Reconstruct actual complex impedance
            Z = df['Z_real'] + 1j * df['Z_imag']

            # Recreate proper impedance magnitude and phase
            # so the rest of your pipeline still works
            df['Impedance (ohms)'] = np.abs(Z)
            df['Phase (Radians)'] = np.angle(Z)

            # Metadata
            df['timestamp'] = timestamp
            df['folder'] = folder_name
            df['sample'] = sample_name

            # Frequency filtering
            df_f = df[
                (df['Frequency (Hz)'] >= start_freq) &
                (df['Frequency (Hz)'] <= end_freq)
            ]

            if not df_f.empty:
                all_data.append(df_f)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    combined.sort_values(
        ['timestamp', 'sample'],
        inplace=True
    )

    return combined

# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_bvd_fast(frequency, z_complex, peak_weight=20.0, bounds=None, maxiter=300,
                 r2_threshold=0.5, max_retries=2, optimize_for='real',
                 previous_params=None, max_L1_increase=0.0, n_starts=4,
                 jitter_scale=0.08, l1_change_penalty=0.0):
    """Fit simple BVD using a fast bounded local least-squares solve.

    This is intended as a faster alternative to differential evolution for the
    streaming workflow. It uses a warm-start when available and a few bounded
    local restarts when the first solution is poor.
    """

    L1_BASELINE = 0.020  # H

    use_custom_bounds = bounds is not None
    first_fit = previous_params is None
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    def _apply_l1_constraint(bds):
        bds = list(bds)
        if first_fit:
            bds[3] = (L1_BASELINE, L1_BASELINE)
            return bds

        if previous_params is not None:
            prev_L1 = previous_params[3]
            low, high = bds[3]
            max_allowed = prev_L1 * (1 + max_L1_increase)
            high = min(high, max_allowed)
            low = min(low, high)
            bds[3] = (low, high)
        return bds

    if first_fit:
        bounds = list(bounds)
        bounds[3] = (L1_BASELINE, L1_BASELINE)
    elif previous_params is not None:
        bounds = _apply_l1_constraint(bounds)

    freq_p, z_p, f_peak, (idx_lo, idx_hi) = preprocess_dominant_peak(frequency, z_complex)
    if f_peak is None:
        return None, {'success': False}

    weights = np.ones(len(freq_p))
    weights[idx_lo:idx_hi+1] = peak_weight
    sqrt_weights = np.sqrt(weights)
    norm = np.max(np.abs(z_p)) or 1.0
    z_n = z_p / norm
    l1_reference = None if previous_params is None else max(abs(previous_params[3]), 1e-18)

    def _build_seed(bds):
        if previous_params is not None:
            seed = np.asarray(previous_params, dtype=float)
        else:
            vals = []
            for low, high in bds:
                if np.isclose(low, high):
                    vals.append(low)
                elif low > 0 and high > 0:
                    vals.append(float(np.exp((np.log(low) + np.log(high)) / 2.0)))
                else:
                    vals.append((low + high) / 2.0)
            seed = np.asarray(vals, dtype=float)

        lower = np.asarray([low for low, _ in bds], dtype=float)
        upper = np.asarray([high for _, high in bds], dtype=float)
        return np.clip(seed, lower, upper)

    def _l1_penalty_residual(p):
        if l1_change_penalty <= 0.0 or l1_reference is None:
            return np.empty(0, dtype=float)
        relative_change = (p[3] - previous_params[3]) / l1_reference
        return np.array([np.sqrt(l1_change_penalty) * relative_change], dtype=float)

    def _residuals(p):
        Z_m = bvd_1st_order(freq_p, *p)
        if not np.all(np.isfinite(Z_m)):
            return np.full(len(freq_p) + (1 if l1_change_penalty > 0.0 and l1_reference is not None else 0), 1e6, dtype=float)

        Z_n = Z_m / norm
        if optimize_for == 'real':
            data_residual = sqrt_weights * (np.real(Z_n) - np.real(z_n))
        elif optimize_for == 'imag':
            data_residual = sqrt_weights * (np.imag(Z_n) - np.imag(z_n))
        elif optimize_for == 'magnitude':
            data_residual = sqrt_weights * (np.abs(Z_n) - np.abs(z_n))
        elif optimize_for == 'phase':
            data_residual = sqrt_weights * (np.angle(Z_n) - np.angle(z_n))
        else:
            raise ValueError(f"Invalid optimize_for value: {optimize_for}")

        penalty_residual = _l1_penalty_residual(p)
        if penalty_residual.size:
            return np.concatenate([data_residual, penalty_residual])
        return data_residual

    def _quality(p, retried=False, retries=0):
        Z_fit = bvd_1st_order(frequency, *p)
        return {
            'success': True,
            'retried': retried,
            'retries': retries,
            'r2_magnitude': r2_score(np.abs(z_complex), np.abs(Z_fit)),
            'r2_real': r2_score(np.real(z_complex), np.real(Z_fit)),
            'r2_imag': r2_score(np.imag(z_complex), np.imag(Z_fit)),
            'f_peak_hz': f_peak,
            'peak_window': (frequency[idx_lo], frequency[idx_hi]),
        }

    def _run(bds, tries, seed_offset=0):
        bds = sanitize_bounds(bds)
        lower = np.asarray([low for low, _ in bds], dtype=float)
        upper = np.asarray([high for _, high in bds], dtype=float)
        base_seed = _build_seed(bds)
        fixed_mask = np.isclose(lower, upper)
        free_mask = ~fixed_mask
        rng = np.random.default_rng(1234 + seed_offset)

        if not np.any(free_mask):
            class _FixedResult:
                def __init__(self, x, cost):
                    self.x = x
                    self.cost = cost

            residual = _residuals(base_seed)
            return _FixedResult(base_seed, 0.5 * np.dot(residual, residual))

        lower_free = lower[free_mask]
        upper_free = upper[free_mask]
        base_seed_free = base_seed[free_mask]

        def _merge_free_params(free_values):
            full = base_seed.copy()
            full[fixed_mask] = lower[fixed_mask]
            full[free_mask] = free_values
            return full

        def _residuals_free(free_values):
            return _residuals(_merge_free_params(free_values))

        best_result = None
        best_loss = np.inf

        for attempt in range(max(1, tries)):
            if attempt == 0:
                x0_free = base_seed_free.copy()
            else:
                noise = rng.normal(0.0, 1.0, size=base_seed_free.shape)
                span_free = np.maximum(upper_free - lower_free, 1e-18)
                x0_free = np.clip(base_seed_free + noise * span_free * jitter_scale, lower_free, upper_free)

            result = least_squares(
                _residuals_free,
                x0=x0_free,
                bounds=(lower_free, upper_free),
                max_nfev=maxiter,
                method='trf',
                loss='soft_l1',
                f_scale=0.1,
            )

            result.x = _merge_free_params(result.x)

            if result.cost < best_loss:
                best_loss = result.cost
                best_result = result

        return best_result

    result = _run(bounds, tries=n_starts)
    if result is None or not np.all(np.isfinite(result.x)):
        return None, {'success': False}

    p = result.x
    qual = _quality(p)

    if qual['r2_real'] < r2_threshold and use_custom_bounds and max_retries > 0:
        best_p = p
        best_qual = qual
        current_bounds = bounds
        retries = 0

        for retry_idx in range(max_retries):
            retries += 1
            current_bounds = expand_bounds(current_bounds, factor=0.25)
            current_bounds = _apply_l1_constraint(current_bounds)
            result2 = _run(current_bounds, tries=max(2, n_starts - 1), seed_offset=retry_idx + 1)

            if result2 is not None and np.all(np.isfinite(result2.x)):
                p2 = result2.x
                qual2 = _quality(p2, retried=True, retries=retries)
                if qual2['r2_real'] > best_qual['r2_real']:
                    best_p, best_qual = p2, qual2

            if best_qual['r2_real'] >= r2_threshold:
                break

        if best_qual['r2_real'] > qual['r2_real']:
            return best_p, best_qual

        qual['retries'] = retries
        return p, qual

    return p, qual

def fit_bvd(frequency, z_complex, peak_weight=20.0, bounds=None, maxiter=6000,
            r2_threshold=0.5, max_retries=3, optimize_for='real',
            previous_params=None, max_L1_increase=0.0, l1_change_penalty=0.0):
    """
    Fit simple BVD (R0, C0, R1, L1, C1) via differential evolution.

    Parameters
    ----------
    bounds        : list of (min, max) per parameter; falls back to DEFAULT_BOUNDS.
    r2_threshold  : if the initial fit's r2_real is below this value the fit is
                    automatically retried with DEFAULT_BOUNDS (full search space).
                    Set to -np.inf to disable the retry.
    max_retries   : maximum number of retry attempts with DEFAULT_BOUNDS when the
                    fit quality is below r2_threshold. Default is 1.
    optimize_for  : which component to optimize for ('real', 'imag', 'magnitude', 'phase'). Default is 'real'.

    Returns
    -------
    params  : array [R0, C0, R1, L1, C1] or None
    quality : dict with 'success', r2 scores, peak info and 'retried' flag
    """

    L1_BASELINE = 0.020  # H

    use_custom_bounds = bounds is not None
    first_fit = previous_params is None
    #first_fit = 1 # This is to lock L1
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    
    def _apply_l1_constraint(bds):
        """Enforce L1 trend constraint: first fit fixed baseline, later fits no increase."""
        bds = list(bds)
        if first_fit:
            bds[3] = (L1_BASELINE, L1_BASELINE)
            return bds

        if previous_params is not None:
            prev_L1 = previous_params[3]
            low, high = bds[3]
            max_allowed = prev_L1 * (1 + max_L1_increase)
            high = min(high, max_allowed)
            # If custom bounds are tighter than prev_L1 and invert the interval,
            # collapse to a fixed bound to avoid sanitize_bounds swapping it.
            low = min(low, high)
            bds[3] = (low, high)
        return bds

    if first_fit:
        # Force L1 to baseline value
        bounds = list(bounds)
        bounds[3] = (L1_BASELINE, L1_BASELINE)

    elif previous_params is not None:
        bounds = _apply_l1_constraint(bounds)

    freq_p, z_p, f_peak, (idx_lo, idx_hi) = preprocess_dominant_peak(frequency, z_complex)
    if f_peak is None:
        return None, {'success': False}

    weights = np.ones(len(frequency))
    weights[idx_lo:idx_hi+1] = peak_weight
    l1_reference = None if previous_params is None else max(abs(previous_params[3]), 1e-18)

    def objective(p):
        Z_m = bvd_1st_order(freq_p, *p)
        if not np.all(np.isfinite(Z_m)):
            return 1e6

        norm = np.max(np.abs(z_p)) or 1.0
        Z_n  = Z_m / norm
        z_n  = z_p / norm

        rmsd_re = np.sqrt(np.mean(weights * (np.real(Z_n) - np.real(z_n))**2))
        rmsd_im = np.sqrt(np.mean(weights * (np.imag(Z_n) - np.imag(z_n))**2))
        rmsd_mag = np.sqrt(np.mean(weights * (np.abs(Z_n) - np.abs(z_n))**2))
        rmsd_phase = np.sqrt(np.mean(weights * (np.angle(Z_n) - np.angle(z_n))**2))
        
        if optimize_for == 'real':
            score = rmsd_re
        elif optimize_for == 'imag':
            score = rmsd_im
        elif optimize_for == 'magnitude':
            score = rmsd_mag
        elif optimize_for == 'phase':
            score = rmsd_phase
        else:
            raise ValueError(f"Invalid optimize_for value: {optimize_for}")

        if l1_change_penalty > 0.0 and l1_reference is not None:
            relative_change = (p[3] - previous_params[3]) / l1_reference
            score += np.sqrt(l1_change_penalty) * abs(relative_change)

        return score
        
    def _run(bds):
        bds = sanitize_bounds(bds)
        return differential_evolution(objective, bds, maxiter=maxiter, tol=1e-4, polish=True, popsize=10, seed=1234)

    def _quality(p, retried=False, retries=0):
        Z_fit = bvd_1st_order(frequency, *p)
        return {
            'success':      True,
            'retried':      retried,
            'retries':      retries,
            'r2_magnitude': r2_score(np.abs(z_complex),  np.abs(Z_fit)),
            'r2_real':      r2_score(np.real(z_complex), np.real(Z_fit)),
            'r2_imag':      r2_score(np.imag(z_complex), np.imag(Z_fit)),
            'f_peak_hz':    f_peak,
            'peak_window':  (frequency[idx_lo], frequency[idx_hi]),
        }

    result = _run(bounds)
    if not result.success:
        return None, {'success': False}

    p    = result.x
    qual = _quality(p)

    if qual['r2_real'] < r2_threshold and use_custom_bounds and max_retries > 0:

        best_p    = p
        best_qual = qual
        retries   = 0
        current_bounds = bounds

        for i in range(max_retries):

            retries += 1

            current_bounds = expand_bounds(current_bounds, factor=0.25)
            current_bounds = _apply_l1_constraint(current_bounds)

            result2 = _run(current_bounds)

            if result2.success:
                p2    = result2.x
                qual2 = _quality(p2, retried=True)

                if qual2['r2_real'] > best_qual['r2_real']:
                    best_p, best_qual = p2, qual2

            if best_qual['r2_real'] >= r2_threshold:
                break

        best_qual['retries'] = retries

        if best_qual['r2_real'] > qual['r2_real']:
            return best_p, best_qual
        # if retries did not improve
        else:
            qual['retries'] = qual.get('retries', 0)
            return p, qual
    else:
        return p, qual
    



def free_onedrive_folder(folder_path):
    """Free local disk space for a OneDrive-synced folder.

    Uses ``attrib +U -P`` to mark the folder's contents as cloud-only so that
    OneDrive evicts them from local storage (equivalent to right-clicking →
    "Free up space" in Explorer).  Safe to call on non-OneDrive paths — the
    ``attrib`` call will simply have no visible effect there.

    Only runs on Windows; silently skips on other platforms.
    """
    if sys.platform != 'win32':
        return
    try:
        subprocess.run(
            ['attrib', '+U', '-P', folder_path, '/S', '/D'],
            capture_output=True,
            check=False,   # don't raise on non-zero exit — e.g. network paths
        )
    except FileNotFoundError:
        pass  # attrib not found — unlikely on Windows, ignore gracefully


def expand_bounds(bounds, factor=0.2):

    new_bounds = []

    for low, high in bounds:

        width = high - low
        expand = width * factor

        new_low = max(1e-18, low - expand)
        new_high = high + expand

        new_bounds.append((new_low, new_high))

    return new_bounds

def sanitize_bounds(bounds):
    """Ensure all bounds are valid (low <= high)."""
    clean = []
    for low, high in bounds:
        if low > high:
            low, high = high, low
        clean.append((low, high))
    return clean

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_bvd_fit(frequency, z_complex, params, title="BVD Fit"):
    """Four-panel plot: |Z|, phase, Re(Z), Im(Z) — raw vs preprocessed vs fit."""
    Z_fit = bvd_1st_order(frequency, *params)
    _, z_p, _, (idx_lo, idx_hi) = preprocess_dominant_peak(frequency, z_complex)

    R0, C0, R1, L1, C1 = params
    f_res  = resonant_frequency(L1, C1)
    Q      = quality_factor(f_res, L1, R1)
    r2_mag = r2_score(np.abs(z_complex), np.abs(Z_fit))
    r2_re  = r2_score(np.real(z_complex), np.real(Z_fit))

    kHz  = frequency / 1000
    f_lo = frequency[idx_lo] / 1000
    f_hi = frequency[idx_hi] / 1000

    sub = (f"f_res={f_res/1000:.3f} kHz  Q={Q:.0f}  R1={R1:.1f} Ω  "
           f"L1={L1*1e3:.3f} mH  C1={C1*1e15:.2f} fF\n"
           f"R0={R0:.1f} Ω  C0={C0*1e9:.4f} nF  "
           f"R²_mag={r2_mag:.4f}  R²_real={r2_re:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{title}\n{sub}", fontsize=10)

    labels = ['Original', 'Preprocessed', 'BVD fit']
    styles = [('b-', 2, 1.0), ('g--', 1.5, 0.8), ('r-', 2, 1.0)]

    def shade(ax):
        ax.axvspan(f_lo, f_hi, alpha=0.08, color='orange', label='Peak window')

    def prow(ax, yo, yp, yf, ylabel, ttl):
        for (st, lw, al), y, lb in zip(styles, [yo, yp, yf], labels):
            ax.plot(kHz, y, st, lw=lw, alpha=al, label=lb)
        shade(ax)
        ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel(ylabel)
        ax.set_title(ttl); ax.legend(fontsize=8); ax.grid()

    prow(axes[0,0], np.abs(z_complex),             np.abs(z_p),             np.abs(Z_fit),             '|Z| (Ω)',    'Magnitude')
    prow(axes[0,1], np.angle(z_complex, deg=True), np.angle(z_p, deg=True), np.angle(Z_fit, deg=True), 'Phase (°)', 'Phase')
    prow(axes[1,0], np.real(z_complex),            np.real(z_p),            np.real(Z_fit),            'Re(Z) (Ω)', 'Real Part')
    prow(axes[1,1], np.imag(z_complex),            np.imag(z_p),            np.imag(Z_fit),            'Im(Z) (Ω)', 'Imaginary Part')

    plt.tight_layout()
    plt.show()
