# ============================================================
# Author : Mads Kofod Dahl
# License: MIT License (2026)
# ============================================================


"""
BVD Circuit GUI Fitter
======================
Interactive tool for fitting a Butterworth-Van Dyke (BVD) circuit model
to measured impedance data.

Usage
-----
    python bvd_gui_fitter.py

Data format
-----------
CSV or Excel file with columns (case-insensitive matching):
    - Frequency (Hz)
    - Impedance (ohms)   [raw values are scaled by 1e5, matching the dataset convention]
    - Phase (Radians)    [raw values are multiplied by -100, matching the dataset convention]
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# BVD model
# ---------------------------------------------------------------------------

def bvd_impedance(frequency_hz, R0, R1, L1, C1, C0):
    """Return complex impedance of the BVD equivalent circuit."""
    w = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    jw = 1j * w
    z_c0 = 1.0 / (jw * C0)
    z_motional = R1 + jw * L1 + 1.0 / (jw * C1)
    z_parallel = 1.0 / (1.0 / z_motional + 1.0 / z_c0)
    return R0 + z_parallel


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class BVDFitterApp:

    PARAMS = ["R0", "R1", "L1", "C1", "C0"]

    DEFAULTS = {
        "R0": 140.0,
        "R1": 1955.0,
        "L1": 0.020,
        "C1": 2.45e-11,
        "C0": 7.96e-10,
    }

    DEFAULT_RANGES = {
        "R0": (0.0,    2000.0),
        "R1": (0.0,   10000.0),
        "L1": (0.001,  0.1),
        "C1": (1e-12,  1e-10),
        "C0": (1e-11,  1e-8),
    }

    N_STEPS = 1000  # slider internal resolution

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BVD Circuit Fitter")

        # Raw loaded data arrays
        self.freq_raw: np.ndarray | None = None
        self.z_real_raw: np.ndarray | None = None

        # Frequency-filtered view (what is plotted)
        self.freq: np.ndarray | None = None
        self.z_real: np.ndarray | None = None

        # Current BVD parameter values
        self.params: dict[str, float] = dict(self.DEFAULTS)

        # Slider range limits [min, max] — user-editable
        self.ranges: dict[str, list[float]] = {k: list(v) for k, v in self.DEFAULT_RANGES.items()}

        # Widget references populated during build
        self.sliders: dict[str, ttk.Scale] = {}
        self.value_labels: dict[str, tk.StringVar] = {}
        self.min_vars: dict[str, tk.StringVar] = {}
        self.max_vars: dict[str, tk.StringVar] = {}

        self._block_update: bool = False  # guard against recursive slider updates

        self._build_ui()
        self._draw_placeholder()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ---- Row 0: top bar (file + frequency range + RMSD) ----
        top = ttk.Frame(self.root, padding=(6, 5))
        top.grid(row=0, column=0, sticky="ew")

        ttk.Button(top, text="Load File…", command=self._load_file).pack(side=tk.LEFT)

        self._file_label = ttk.Label(top, text="No file loaded", width=42, anchor="w")
        self._file_label.pack(side=tk.LEFT, padx=(6, 10))

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=(0, 8))

        ttk.Label(top, text="Freq min (Hz):").pack(side=tk.LEFT)
        self._fmin_var = tk.StringVar(value="180000")
        ttk.Entry(top, textvariable=self._fmin_var, width=9).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(top, text="Freq max (Hz):").pack(side=tk.LEFT)
        self._fmax_var = tk.StringVar(value="250000")
        ttk.Entry(top, textvariable=self._fmax_var, width=9).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Button(top, text="Apply", command=self._apply_range).pack(side=tk.LEFT, padx=(0, 14))

        self._rmsd_var = tk.StringVar(value="RMSD Re{Z}: —")
        ttk.Label(top, textvariable=self._rmsd_var, width=22,
                  foreground="navy", font=("TkDefaultFont", 10, "bold")).pack(side=tk.RIGHT, padx=8)

        # ---- Row 1: plot (left) + sliders panel (right) ----
        content = ttk.Frame(self.root)
        content.grid(row=1, column=0, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        # -- Plot --
        plot_frame = ttk.Frame(content)
        plot_frame.grid(row=0, column=0, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(7, 4.8), dpi=90)
        self.fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.12)
        self._canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # -- Sliders panel --
        panel = ttk.LabelFrame(content, text="BVD Parameters", padding=10)
        panel.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=6)

        # Header row
        col_cfg = [("Param", 5), ("Current value", 14), ("Slider", 20), ("Min", 10), ("Max", 10)]
        for col, (hdr, w) in enumerate(col_cfg):
            ttk.Label(panel, text=hdr, width=w, anchor="center",
                      font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=col, padx=2, pady=(0, 6))

        # One row per BVD parameter
        for row_idx, param in enumerate(self.PARAMS, start=1):
            ttk.Label(panel, text=param, width=5, anchor="e").grid(
                row=row_idx, column=0, padx=(2, 4), pady=5)

            val_var = tk.StringVar(value=self._fmt(self.params[param]))
            self.value_labels[param] = val_var
            ttk.Label(panel, textvariable=val_var, width=14, anchor="e",
                      foreground="darkred").grid(row=row_idx, column=1, padx=4)

            slider = ttk.Scale(
                panel, from_=0, to=self.N_STEPS, orient=tk.HORIZONTAL, length=210,
                command=lambda v, p=param: self._on_slider(p, float(v)),
            )
            slider.set(self._to_slider(param, self.params[param]))
            slider.grid(row=row_idx, column=2, padx=4)
            self.sliders[param] = slider

            lo, hi = self.ranges[param]

            min_var = tk.StringVar(value=str(lo))
            self.min_vars[param] = min_var
            me = ttk.Entry(panel, textvariable=min_var, width=10)
            me.grid(row=row_idx, column=3, padx=2)
            me.bind("<Return>",   lambda e, p=param: self._apply_range_param(p))
            me.bind("<FocusOut>", lambda e, p=param: self._apply_range_param(p))

            max_var = tk.StringVar(value=str(hi))
            self.max_vars[param] = max_var
            xe = ttk.Entry(panel, textvariable=max_var, width=10)
            xe.grid(row=row_idx, column=4, padx=2)
            xe.bind("<Return>",   lambda e, p=param: self._apply_range_param(p))
            xe.bind("<FocusOut>", lambda e, p=param: self._apply_range_param(p))

        ttk.Button(panel, text="Reset to defaults",
                   command=self._reset_params).grid(
            row=len(self.PARAMS) + 1, column=0, columnspan=5,
            pady=(10, 0), sticky="ew")

        # Small hint label
        hint = (
            "Tip: edit Min / Max fields and press Enter to rescale a slider's range.\n"
            "Use scientific notation for small values, e.g.  1e-12"
        )
        ttk.Label(panel, text=hint, foreground="#666666",
                  font=("TkDefaultFont", 8)).grid(
            row=len(self.PARAMS) + 2, column=0, columnspan=5,
            pady=(6, 0), sticky="w")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(value: float) -> str:
        """Compact display for both large and tiny numbers."""
        if value == 0.0:
            return "0"
        mag = abs(value)
        if mag < 1e-4 or mag >= 1e6:
            return f"{value:.4e}"
        return f"{value:.6g}"

    def _to_slider(self, param: str, value: float) -> float:
        lo, hi = self.ranges[param]
        if hi == lo:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo) * self.N_STEPS, 0.0, self.N_STEPS))

    def _from_slider(self, param: str, slider_val: float) -> float:
        lo, hi = self.ranges[param]
        return lo + (slider_val / self.N_STEPS) * (hi - lo)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_slider(self, param: str, slider_val: float) -> None:
        if self._block_update:
            return
        value = self._from_slider(param, slider_val)
        self.params[param] = value
        self.value_labels[param].set(self._fmt(value))
        self._update_plot()

    def _apply_range_param(self, param: str) -> None:
        """Re-scale a slider when the user edits its Min / Max fields."""
        try:
            new_min = float(self.min_vars[param].get())
            new_max = float(self.max_vars[param].get())
        except ValueError:
            return
        if new_min >= new_max:
            return
        self.ranges[param] = [new_min, new_max]
        clamped = float(np.clip(self.params[param], new_min, new_max))
        self.params[param] = clamped
        self.value_labels[param].set(self._fmt(clamped))
        self._block_update = True
        self.sliders[param].set(self._to_slider(param, clamped))
        self._block_update = False
        self._update_plot()

    def _reset_params(self) -> None:
        for param in self.PARAMS:
            self.params[param] = self.DEFAULTS[param]
            self.value_labels[param].set(self._fmt(self.DEFAULTS[param]))
            self._block_update = True
            self.sliders[param].set(self._to_slider(param, self.DEFAULTS[param]))
            self._block_update = False
        self._update_plot()

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select impedance data file",
            filetypes=[
                ("Excel / CSV", "*.xlsx *.xls *.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        freq_col  = self._find_col(df, ["Frequency (Hz)", "Frequency", "freq", "f(Hz)", "f"])
        imp_col   = self._find_col(df, ["Impedance (ohms)", "Impedance", "Z_mag", "|Z|", "Z"])
        phase_col = self._find_col(df, ["Phase (Radians)", "Phase", "phase"])

        if None in (freq_col, imp_col, phase_col):
            messagebox.showerror(
                "Column error",
                f"Could not identify the required columns.\n\nDetected columns:\n{list(df.columns)}\n\n"
                "Expected: Frequency (Hz), Impedance (ohms), Phase (Radians)",
            )
            return

        freq  = df[freq_col].to_numpy(dtype=float)
        # Apply same scaling as original analysis script
        imp   = df[imp_col].to_numpy(dtype=float) * 10e4
        phase = df[phase_col].to_numpy(dtype=float) * -100

        z_real = imp * np.cos(phase)

        self.freq_raw   = freq
        self.z_real_raw = z_real

        self._file_label.config(text=os.path.basename(path))
        self._apply_range()

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        # Exact match first
        for c in candidates:
            if c in df.columns:
                return c
        # Case-insensitive exact
        lower_map = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        # Substring match
        for c in candidates:
            for col_low, col_orig in lower_map.items():
                if c.lower() in col_low:
                    return col_orig
        return None

    # ------------------------------------------------------------------
    # Frequency range
    # ------------------------------------------------------------------

    def _apply_range(self) -> None:
        if self.freq_raw is None:
            return
        try:
            f_min = float(self._fmin_var.get())
            f_max = float(self._fmax_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric frequency limits.")
            return

        mask = (self.freq_raw >= f_min) & (self.freq_raw <= f_max)
        self.freq   = self.freq_raw[mask]
        self.z_real = self.z_real_raw[mask]

        if self.freq.size == 0:
            messagebox.showwarning(
                "Empty range",
                f"No data points found between {f_min:.0f} Hz and {f_max:.0f} Hz.",
            )
        self._update_plot()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _draw_placeholder(self) -> None:
        self.ax.set_facecolor("#f5f5f5")
        self.ax.text(0.5, 0.5, "Load a data file to begin",
                     transform=self.ax.transAxes,
                     ha="center", va="center", fontsize=13, color="#aaaaaa",
                     style="italic")
        self.ax.set_title("BVD Circuit Fitter", fontsize=11)
        self.ax.tick_params(labelbottom=False, labelleft=False)
        self._canvas.draw_idle()

    def _update_plot(self) -> None:
        if self._block_update:
            return

        self.ax.clear()

        if self.freq is not None and self.freq.size > 0:
            f_khz = self.freq / 1e3

            # Measured data
            self.ax.plot(f_khz, self.z_real, color="steelblue",
                         linewidth=1.8, label="Measured Re{Z}")

            # BVD model overlay
            try:
                z_model    = bvd_impedance(self.freq, **self.params)
                z_model_re = np.real(z_model)
                self.ax.plot(f_khz, z_model_re, color="crimson", linestyle="--",
                             linewidth=1.8, label="BVD Model Re{Z}")

                rmsd = np.sqrt(np.mean((self.z_real - z_model_re) ** 2))
                self._rmsd_var.set(f"RMSD Re{{Z}}: {rmsd:.3f} Ω")
            except Exception:
                self._rmsd_var.set("RMSD Re{Z}: error")

            self.ax.set_xlabel("Frequency (kHz)", fontsize=10)
            self.ax.set_ylabel("Re{Z} (Ω)", fontsize=10)
            self.ax.set_title("BVD Circuit Fit — Real Part of Impedance", fontsize=11)
            self.ax.legend(fontsize=9)
            self.ax.grid(True, alpha=0.3)
            self.fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.12)
        else:
            self._draw_placeholder()
            return

        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.geometry("1230x650")
    root.minsize(950, 520)
    BVDFitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
