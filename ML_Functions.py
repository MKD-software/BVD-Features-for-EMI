import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch


def _safe_split_timestamp(data_sorted, split_idx):
    """Return a valid split timestamp or None when split_idx is unusable."""
    if split_idx is None or data_sorted is None or len(data_sorted) == 0:
        return None

    if isinstance(split_idx, float) and not split_idx.is_integer():
        return None

    try:
        idx = int(split_idx)
    except (TypeError, ValueError):
        return None

    if idx < 0 or idx >= len(data_sorted):
        return None

    return data_sorted.iloc[idx]['timestamp']

def calculate_mass_loss(current_df, uptime_df):

    # Calculate estimated mass loss using Faraday's law
    # Constants
    F = 96485  # Faraday constant (C/mol)
    M_Fe = 55.845  # Atomic mass of iron (g/mol) 
    n = 2  # Electrons for Fe -> Fe2+

    # Calculate cumulative mass loss for each sample
    mass_loss_data = []

    for sample in ['sample_0', 'sample_5', 'sample_10']:
        sample_currents = current_df[current_df['sample'] == sample].copy()
        sample_currents = sample_currents.sort_values('timestamp')
        
        cumulative_mass_loss = 0
        
        for idx, row in sample_currents.iterrows():
            # Find corresponding uptime for this date
            date_key = row['timestamp'].date()
            uptime_match = uptime_df[uptime_df['Date'] == date_key]
            
            if not uptime_match.empty:
                current_mA = row['current_mA']
                current_A = current_mA / 1000  # Convert mA to A
                time_s = uptime_match.iloc[0]['Uptime_hours'] * 3600  # Convert hours to seconds
                
                # Faraday's law: mass = (I * t * M) / (n * F)
                daily_mass_loss_g = (current_A * time_s * M_Fe) / (n * F)
                cumulative_mass_loss += daily_mass_loss_g
                
                mass_loss_data.append({
                    'Date': date_key,
                    'Sample': sample,
                    'Daily_Mass_Loss_g': daily_mass_loss_g,  # Convert to mg
                    'Cumulative_Mass_Loss_g': cumulative_mass_loss
                })

    mass_loss_df = pd.DataFrame(mass_loss_data)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Corrosion_Dataset", "mass_loss.csv")
    mass_loss_df.to_csv(out_path, index=False)


# ===============================================================================

def read_and_plot_current_data(corrosion_dataset_path, conditions_sorted):

    # Read and plot current data over time
    current_data = []

    for condition in conditions_sorted:
        currents_file = os.path.join(corrosion_dataset_path, condition['folder'], 'currents.txt')
        
        if os.path.exists(currents_file):
            with open(currents_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line and ';' in line:
                    parts = line.split(';')
                    if len(parts) == 2:
                        sample_name = parts[0].strip()
                        current_str = parts[1].strip()
                        
                        # Extract current value (remove 'mA')
                        current_value = float(current_str.replace('mA', ''))
                        
                        # Normalize sample names to consistent format
                        if sample_name in ['sample1', 'sample_5', 'sample5']:
                            sample_name = 'sample_5'
                        elif sample_name in ['sample2', 'sample_0', 'sample0']:
                            sample_name = 'sample_0'
                        elif sample_name in ['sample3', 'sample_10', 'sample10']:
                            sample_name = 'sample_10'
                        
                        current_data.append({
                            'timestamp': condition['timestamp'],
                            'temperature_C': condition['temperature_C'],
                            'humidity_RH': condition['humidity_RH'],
                            'sample': sample_name,
                            'current_mA': current_value,
                            'condition': f"{condition['temperature_C']}°C, {condition['humidity_RH']}%RH"
                        })

    # Convert to DataFrame
    current_df = pd.DataFrame(current_data)


# =====================================================

def evaluate_and_plot(model,
                      X_tr, X_vl,
                      y_train, y_val,
                      history,
                      data_sorted,
                      scaler,
                      FEATURE_COLS,
                      sample_map,
                      weight_data,
                      split_idx):

    split_ts = _safe_split_timestamp(data_sorted, split_idx)
    
    # ── Predictions ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_tr_pred = model(X_tr).numpy().flatten()
        y_vl_pred = model(X_vl).numpy().flatten()

    # ── Plot settings ────────────────────────────────────────────────────────
    plt.rcParams['figure.figsize'] = (22, 14)
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 150

    fig, axes = plt.subplots(3, 3, figsize=(22, 14))

    # Keep train/val arrays as numpy for metrics and diagnostic plots
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # Bootstrap RMSE variance by resampling time-indexed points with replacement.
    def _bootstrap_rmse_stats(y_true, y_pred, n_bootstrap=1000, seed=42):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)

        if n < 2:
            return {
                'variance': np.nan,
                'std': np.nan,
                'mean': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'samples': np.array([])
            }

        rng = np.random.default_rng(seed)
        rmse_samples = np.empty(n_bootstrap, dtype=float)

        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            rmse_samples[i] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))

        return {
            'variance': np.var(rmse_samples, ddof=1),
            'std': np.std(rmse_samples, ddof=1),
            'mean': np.mean(rmse_samples),
            'ci_low': np.percentile(rmse_samples, 2.5),
            'ci_high': np.percentile(rmse_samples, 97.5),
            'samples': rmse_samples
        }

    # Core metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_tr_pred))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_vl_pred))
    mae_train = mean_absolute_error(y_train, y_tr_pred)
    mae_val = mean_absolute_error(y_val, y_vl_pred)
    r2_train = r2_score(y_train, y_tr_pred)
    r2_val = r2_score(y_val, y_vl_pred)

    # Bootstrap uncertainty over time points for RMSE
    boot_train = _bootstrap_rmse_stats(y_train, y_tr_pred, n_bootstrap=1000, seed=42)
    boot_val = _bootstrap_rmse_stats(y_val, y_vl_pred, n_bootstrap=1000, seed=4242)

    # ── Plot 1: training loss history ───────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['train_task'], label='Task loss (train)')
    ax.plot(history['epoch'], history['train_mono'], label='Monotonicity loss', linestyle='--')
    ax.plot(history['epoch'], history['train_phys'], label='Sauerbrey physics loss', linestyle='-.')
    ax.plot(history['epoch'], history['val_task'], label='Task loss (val)', linestyle=':')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: predicted vs actual ─────────────────────────────────────────
    ax = axes[0, 1]
    lim = max(y_train.max(), y_val.max()) * 1.1

    ax.scatter(y_train, y_tr_pred, alpha=0.7, s=40,
               label=f'Train  R²={r2_train:.3f}')

    ax.scatter(y_val, y_vl_pred, alpha=0.7, s=40, marker='^',
               label=f'Val    R²={r2_val:.3f}')

    ax.plot([0, lim], [0, lim], 'k--', lw=1)

    ax.set_xlabel('Actual mass loss (g)')
    ax.set_ylabel('Predicted mass loss (g)')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: per-sample timeline ─────────────────────────────────────────
    ax = axes[0, 2]

    colors_s = {
        'sample_0': 'tab:blue',
        'sample_5': 'tab:red',
        'sample_10': 'tab:green'
    }

    wt_colors = {
        'Sample0': 'tab:blue',
        'Sample5': 'tab:red',
        'Sample10': 'tab:green'
    }

    for sample_key, color in colors_s.items():

        sub = data_sorted[data_sorted['sample'] == sample_key].sort_values('timestamp')
        if sub.empty:
            continue

        Xs = torch.tensor(
            scaler.transform(sub[FEATURE_COLS].values.astype(float)),
            dtype=torch.float32
        )

        with torch.no_grad():
            pred = model(Xs).numpy().flatten()

        ax.plot(sub['timestamp'], sub['Cumulative_Mass_Loss_g_piecewise_cal'],
                color=color, linestyle='--', alpha=0.5,
                label=f'{sample_key} Faraday')

        ax.plot(sub['timestamp'], pred,
                color=color, alpha=0.9,
                label=f'{sample_key} predicted')

    for sample_key, (sample_wt, loss_col) in sample_map.items():

        ax.scatter(weight_data['Date_parsed'], weight_data[loss_col],
                   color=wt_colors.get(sample_wt, 'black'),
                   marker='*', s=200, zorder=10,
                   label=f'{sample_wt} (scale)')

    if split_ts is not None:
        ax.axvline(split_ts,
                   color='grey', lw=1.5, linestyle=':',
                   label='Train/Val split')

    ax.set_xlabel('Date')
    ax.set_ylabel('Mass Loss (g)')
    ax.set_title('Mass Loss Over Time')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plots 4–5: residuals per sample ────────────────────────────────────
    ax = axes[1, 0]
    all_true = np.concatenate([y_train, y_val])
    all_pred = np.concatenate([y_tr_pred, y_vl_pred])
    residuals = all_pred - all_true
    ax.scatter(all_true, residuals, alpha=0.5, s=30)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('Actual mass loss (g)')
    ax.set_ylabel('Residual (pred − actual)')
    ax.set_title('Residuals vs Actual')
    ax.grid(True, alpha=0.3)

    # Per-sample prediction error over time
    ax = axes[1, 1]
    for sample_key, color in colors_s.items():
        sub = data_sorted[data_sorted['sample'] == sample_key].sort_values('timestamp')
        if sub.empty:
            continue
        Xs = torch.tensor(
            scaler.transform(sub[FEATURE_COLS].values.astype(float)),
            dtype=torch.float32
        )
        with torch.no_grad():
            pred = model(Xs).numpy().flatten()
        err = pred - sub['Cumulative_Mass_Loss_g_piecewise_cal'].values
        ax.plot(sub['timestamp'], err, 'o-', color=color, label=sample_key, markersize=3)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Prediction error (g)')
    ax.set_title('Prediction Error Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 6: absolute error distribution ─────────────────────────────────
    ax = axes[1, 2]
    abs_err_train = np.abs(y_tr_pred - y_train)
    abs_err_val = np.abs(y_vl_pred - y_val)
    ax.hist(abs_err_train, bins=30, alpha=0.6, label='Train', color='tab:blue')
    ax.hist(abs_err_val, bins=30, alpha=0.6, label='Val', color='tab:orange')
    ax.set_xlabel('Absolute error (g)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Helper: smooth diagnostic curves by binning x and averaging y
    def _binned_curve(x, y, n_bins=8):
        x = np.asarray(x)
        y = np.asarray(y)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(x) < 3:
            return np.array([]), np.array([])
        bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
        idx = np.digitize(x, bins, right=False)
        x_mean, y_mean = [], []
        for b in range(1, len(bins) + 1):
            mask = idx == b
            if np.any(mask):
                x_mean.append(np.mean(x[mask]))
                y_mean.append(np.mean(y[mask]))
        return np.array(x_mean), np.array(y_mean)

    # ── Plot 7: calibration plot ────────────────────────────────────────────
    ax = axes[2, 0]
    lim_cal = max(np.max(y_train), np.max(y_val), np.max(y_tr_pred), np.max(y_vl_pred)) * 1.05

    tr_xm, tr_ym = _binned_curve(y_train, y_tr_pred, n_bins=8)
    vl_xm, vl_ym = _binned_curve(y_val, y_vl_pred, n_bins=8)

    if len(tr_xm) > 0:
        ax.plot(tr_xm, tr_ym, 'o-', color='tab:blue', label='Train (binned mean)')
    if len(vl_xm) > 0:
        ax.plot(vl_xm, vl_ym, 's-', color='tab:orange', label='Val (binned mean)')
    ax.plot([0, lim_cal], [0, lim_cal], 'k--', lw=1, label='Perfect calibration')
    ax.set_xlabel('Actual mass loss (g)')
    ax.set_ylabel('Mean predicted mass loss (g)')
    ax.set_title('Calibration Plot')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 8: error vs mass-loss curve ────────────────────────────────────
    ax = axes[2, 1]
    all_true = np.concatenate([y_train, y_val])
    all_pred = np.concatenate([y_tr_pred, y_vl_pred])
    abs_err_all = np.abs(all_pred - all_true)

    ax.scatter(all_true, abs_err_all, alpha=0.4, s=22, color='tab:purple', label='Absolute error')
    curve_x, curve_y = _binned_curve(all_true, abs_err_all, n_bins=10)
    if len(curve_x) > 0:
        ax.plot(curve_x, curve_y, 'o-', color='black', lw=2, label='Binned mean error')
    ax.set_xlabel('Actual mass loss (g)')
    ax.set_ylabel('Absolute prediction error (g)')
    ax.set_title('Error vs Mass-Loss Curve')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 9: metric summary panel ───────────────────────────────────────
    ax = axes[2, 2]
    ax.axis('off')
    summary_lines = [
        'Performance Summary',
        '',
        f'Train RMSE: {rmse_train:.5f} g',
        f'Val   RMSE: {rmse_val:.5f} g',
        f'Train RMSE var (boot): {boot_train["variance"]:.3e}',
        f'Val   RMSE var (boot): {boot_val["variance"]:.3e}',
        f'Train MAE : {mae_train:.5f} g',
        f'Val   MAE : {mae_val:.5f} g',
        f'Train R2  : {r2_train:.4f}',
        f'Val   R2  : {r2_val:.4f}',
    ]
    ax.text(
        0.02,
        0.98,
        '\n'.join(summary_lines),
        va='top',
        ha='left',
        fontsize=11,
        family='monospace',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.9)
    )

    plt.suptitle('PINN — Corrosion: Mass Loss Prediction + Physical Indicators',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ── Metrics ─────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Train  RMSE : {rmse_train:.5f} g")
    print(f"Val    RMSE : {rmse_val:.5f} g")
    print(f"Train  RMSE var (bootstrap over time points): {boot_train['variance']:.6e}")
    print(f"Val    RMSE var (bootstrap over time points): {boot_val['variance']:.6e}")
    print(f"Train  RMSE 95% CI (bootstrap): [{boot_train['ci_low']:.5f}, {boot_train['ci_high']:.5f}] g")
    print(f"Val    RMSE 95% CI (bootstrap): [{boot_val['ci_low']:.5f}, {boot_val['ci_high']:.5f}] g")
    print(f"Train  MAE  : {mae_train:.5f} g")
    print(f"Val    MAE  : {mae_val:.5f} g")
    print(f"Train  R²   : {r2_train:.4f}")
    print(f"Val    R²   : {r2_val:.4f}")
    print("=" * 55)


# =====================================================


def plot_per_sample_analysis(model,
                             data_sorted,
                             scaler,
                             FEATURE_COLS,
                             weight_data,
                             split_idx):

    # ── Per-sample: mass loss prediction + error ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex='col')
    fig.suptitle('Per-Sample: Predicted Mass Loss & Prediction Error',
                 fontsize=13, fontweight='bold')

    samples = ['sample_0', 'sample_5', 'sample_10']

    split_ts = _safe_split_timestamp(data_sorted, split_idx)

    for col_idx, sample_key in enumerate(samples):

        sub = data_sorted[data_sorted['sample'] == sample_key].sort_values('timestamp')

        Xs = torch.tensor(
            scaler.transform(sub[FEATURE_COLS].values.astype(float)),
            dtype=torch.float32
        )

        with torch.no_grad():
            pred = model(Xs).numpy().flatten()

        # Row 0: Actual vs predicted mass loss
        ax = axes[0, col_idx]
        ax.plot(sub['timestamp'], sub['Cumulative_Mass_Loss_g_piecewise_cal'],
                'o--', label='Actual (Faraday)', color='steelblue')
        ax.plot(sub['timestamp'], pred,
                's-', label='Predicted', color='tomato')

        wt_label_map = {
            'sample_0': 'Sample0',
            'sample_5': 'Sample5',
            'sample_10': 'Sample10'
        }
        wt_name = wt_label_map[sample_key]
        loss_col = f'{wt_name}_loss'
        if loss_col in weight_data.columns:
            ax.scatter(weight_data['Date_parsed'],
                       weight_data[loss_col],
                       marker='*', s=200, color='gold',
                       edgecolors='k', zorder=10,
                       label='Scale measurement')
        ax.legend(fontsize=8)
        ax.set_ylabel('Cumulative Mass Loss (g)', fontsize=9)
        ax.set_title(sample_key, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if split_ts is not None:
            ax.axvline(split_ts, color='grey', lw=1.5, linestyle=':', alpha=0.7)

        # Row 1: Prediction error over time
        ax = axes[1, col_idx]
        err = pred - sub['Cumulative_Mass_Loss_g_piecewise_cal'].values
        ax.plot(sub['timestamp'], err, 'o-', color='darkorange', markersize=3)
        ax.axhline(0, color='grey', lw=0.8, linestyle='--')
        ax.set_ylabel('Prediction error (g)', fontsize=9)
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)

        if split_ts is not None:
            ax.axvline(split_ts, color='grey', lw=1.5, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # ── Stiffness panel ─────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 4), sharey=False)

    fig2.suptitle(
        'Stiffness Indicator per Sample (−rel ΔC1 = Δstiffness/stiffness₀)',
        fontsize=12,
        fontweight='bold'
    )

    samples = ['sample_0', 'sample_5', 'sample_10']

    for col_idx, sample_key in enumerate(samples):

        sub = data_sorted[data_sorted['sample'] == sample_key].sort_values('timestamp')

        axes2[col_idx].plot(sub['timestamp'], -sub['rel_C1'],
                            'o-', color='mediumpurple')

        axes2[col_idx].axhline(0, color='grey', lw=0.8, linestyle='--')

        if split_ts is not None:
            axes2[col_idx].axvline(split_ts,
                                   color='grey',
                                   lw=1.5,
                                   linestyle=':',
                                   alpha=0.7)

        axes2[col_idx].set_title(sample_key, fontweight='bold')
        axes2[col_idx].set_xlabel('Date')
        axes2[col_idx].set_ylabel('−rel ΔC1 (↑ = stiffness increase)')
        axes2[col_idx].tick_params(axis='x', rotation=30)
        axes2[col_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()