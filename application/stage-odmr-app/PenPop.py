from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
from scipy.linalg import expm


H = 6.62607015e-34  # J.s
C = 299792458.0       # m/s


def rate_matrix(p: dict[str, float]) -> np.ndarray:
    """Return the 5x5 rate matrix A for dN/dt = A N.

    State order:
        [S0, S1, Tx, Ty, Tz]
    """
    k01 = p["k01"]
    k10 = p["k10"]
    kisc = p["kISC"]

    px, py, pz = p["Px"], p["Py"], p["Pz"]
    kx, ky, kz = p["kx"], p["ky"], p["kz"]

    wxy, wyx = p["wxy"], p["wyx"]
    wxz, wzx = p["wxz"], p["wzx"]
    wyz, wzy = p["wyz"], p["wzy"]

    # Effective incoherent microwave mixing.
    # The user enters the Rabi frequency Ω/2π in MHz.
    # In this rate-equation approximation, ΓMW = Ω = 2π fR
    # is added symmetrically to both directions of the driven pair.
    gxy = p["gamma_xy"]
    gxz = p["gamma_xz"]
    gyz = p["gamma_yz"]

    wxy_eff, wyx_eff = wxy + gxy, wyx + gxy
    wxz_eff, wzx_eff = wxz + gxz, wzx + gxz
    wyz_eff, wzy_eff = wyz + gyz, wzy + gyz

    return np.array(
        [
            [-k01,          k10,                  kx,                  ky,                  kz],
            [ k01, -(k10 + kisc),                 0.0,                 0.0,                 0.0],
            [ 0.0, kisc * px, -(kx + wxy_eff + wxz_eff),              wyx_eff,              wzx_eff],
            [ 0.0, kisc * py,                  wxy_eff, -(ky + wyx_eff + wyz_eff),              wzy_eff],
            [ 0.0, kisc * pz,                  wxz_eff,              wyz_eff, -(kz + wzx_eff + wzy_eff)],
        ],
        dtype=float,
    )


def steady_state(A: np.ndarray) -> np.ndarray:
    """Solve A n = 0 together with sum(n)=1."""
    M = A.copy()
    b = np.zeros(5)

    # Replace one dependent rate equation with normalization.
    M[-1, :] = 1.0
    b[-1] = 1.0

    n = np.linalg.solve(M, b)
    n[np.abs(n) < 1e-14] = 0.0
    return n


class PopulationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pentacene population dynamics — Eq. (2)")
        self.geometry("1500x900")
        self.minsize(1200, 720)

        self.vars: dict[str, tk.StringVar] = {}
        self.last_data: tuple[np.ndarray, np.ndarray] | None = None
        self.last_stationary: np.ndarray | None = None
        self.plot_vars: dict[str, tk.BooleanVar] = {}
        self.last_parameters: dict[str, float] | None = None
        self.last_initial_state: np.ndarray | None = None

        self._build_ui()
        self._set_defaults()
        self.run_simulation()

    def _entry(self, parent, row, label, key, unit="", width=13):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, padx=4, pady=2)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", padx=4, pady=2)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        # Scrollable control panel on the left.
        controls_container = ttk.Frame(outer)
        controls_container.pack(side="left", fill="y")

        controls_canvas = tk.Canvas(
            controls_container,
            width=285,
            highlightthickness=0,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls_scrollbar.pack(side="right", fill="y")
        controls_canvas.pack(side="left", fill="y", expand=False)

        controls = ttk.Frame(controls_canvas, padding=8)
        controls_window = controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )

        def update_scroll_region(_event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def resize_controls_width(event):
            controls_canvas.itemconfigure(controls_window, width=event.width)

        controls.bind("<Configure>", update_scroll_region)
        controls_canvas.bind("<Configure>", resize_controls_width)

        # Mouse-wheel scrolling over the control panel.
        def on_mousewheel(event):
            if event.delta:
                controls_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif event.num == 4:
                controls_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                controls_canvas.yview_scroll(1, "units")

        def bind_mousewheel(_event):
            controls_canvas.bind_all("<MouseWheel>", on_mousewheel)
            controls_canvas.bind_all("<Button-4>", on_mousewheel)
            controls_canvas.bind_all("<Button-5>", on_mousewheel)

        def unbind_mousewheel(_event):
            controls_canvas.unbind_all("<MouseWheel>")
            controls_canvas.unbind_all("<Button-4>")
            controls_canvas.unbind_all("<Button-5>")

        controls_canvas.bind("<Enter>", bind_mousewheel)
        controls_canvas.bind("<Leave>", unbind_mousewheel)

        plot_frame = ttk.Frame(outer, padding=4)
        plot_frame.pack(side="right", fill="both", expand=True)

        # Optical parameters
        optical = ttk.LabelFrame(controls, text="Optical pumping", padding=6)
        optical.pack(fill="x", pady=4)
        self._entry(optical, 0, "Cross section σ", "sigma", "cm²")
        self._entry(optical, 1, "Laser intensity I", "intensity", "W/cm²")
        self._entry(optical, 2, "Laser wavelength λ", "wavelength_nm", "nm")

        ttk.Label(optical, text="Calculated k01").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.k01_label = ttk.Label(optical, text="—")
        self.k01_label.grid(row=3, column=1, columnspan=2, sticky="w", padx=4, pady=2)

        # Intrinsic rates
        rates = ttk.LabelFrame(controls, text="Intrinsic rates", padding=6)
        rates.pack(fill="x", pady=4)

        rows = [
            ("k10", "k10", "s⁻¹"),
            ("kISC", "kISC", "s⁻¹"),
            ("kx", "kx", "s⁻¹"),
            ("ky", "ky", "s⁻¹"),
            ("kz", "kz", "s⁻¹"),
            ("wxy", "wxy", "s⁻¹"),
            ("wyx", "wyx", "s⁻¹"),
            ("wxz", "wxz", "s⁻¹"),
            ("wzx", "wzx", "s⁻¹"),
            ("wyz", "wyz", "s⁻¹"),
            ("wzy", "wzy", "s⁻¹"),
        ]
        for i, (label, key, unit) in enumerate(rows):
            self._entry(rates, i, label, key, unit)

        # Branching probabilities
        probs = ttk.LabelFrame(controls, text="ISC branching probabilities", padding=6)
        probs.pack(fill="x", pady=4)
        self._entry(probs, 0, "Px", "Px", "")
        self._entry(probs, 1, "Py", "Py", "")
        self._entry(probs, 2, "Pz", "Pz", "")

        mw = ttk.LabelFrame(controls, text="Microwave drives", padding=6)
        mw.pack(fill="x", pady=4)
        ttk.Label(
            mw,
            text=("Enter Ω/2π. The rate model uses symmetric mixing "
                  "ΓMW = 2π(Ω/2π). Very large values make adaptive ODE "
                  "solvers stiff; Matrix exponential is recommended."),
            wraplength=245,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))
        self._entry(mw, 1, "Txy drive Ωxy/2π", "rabi_xy_mhz", "MHz")
        self._entry(mw, 2, "Txz drive Ωxz/2π", "rabi_xz_mhz", "MHz")
        self._entry(mw, 3, "Tyz drive Ωyz/2π", "rabi_yz_mhz", "MHz")

        plot_options = ttk.LabelFrame(controls, text="Curves shown", padding=6)
        plot_options.pack(fill="x", pady=4)
        for index, label in enumerate(["S0", "S1", "Tx", "Ty", "Tz", "T1 total"]):
            var = tk.BooleanVar(value=True)
            self.plot_vars[label] = var
            ttk.Checkbutton(
                plot_options,
                text=label,
                variable=var,
                command=self.update_plot,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=4, pady=2)

        ttk.Button(
            plot_options,
            text="Show all",
            command=lambda: self.set_all_curves(True),
        ).grid(row=3, column=0, sticky="ew", padx=4, pady=(5, 2))
        ttk.Button(
            plot_options,
            text="Hide all",
            command=lambda: self.set_all_curves(False),
        ).grid(row=3, column=1, sticky="ew", padx=4, pady=(5, 2))

        # Simulation settings
        sim = ttk.LabelFrame(controls, text="Simulation and initial populations", padding=6)
        sim.pack(fill="x", pady=4)

        ttk.Label(
            sim,
            text="The five initial populations below are editable and must sum to 1.",
            wraplength=245,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))
        self._entry(sim, 1, "Start time", "t_start", "s")
        self._entry(sim, 2, "End time", "t_end", "s")
        self._entry(sim, 3, "Number of points", "n_points", "")
        self._entry(sim, 4, "Initial S0", "S0_0", "")
        self._entry(sim, 5, "Initial S1", "S1_0", "")
        self._entry(sim, 6, "Initial Tx", "Tx_0", "")
        self._entry(sim, 7, "Initial Ty", "Ty_0", "")
        self._entry(sim, 8, "Initial Tz", "Tz_0", "")

        ttk.Label(sim, text="Integrator").grid(row=9, column=0, sticky="w", padx=4, pady=2)
        self.method_var = tk.StringVar(value="Matrix exponential")
        ttk.Combobox(
            sim,
            textvariable=self.method_var,
            values=["Matrix exponential", "Radau", "BDF", "LSODA", "RK45"],
            state="readonly",
            width=11,
        ).grid(row=9, column=1, padx=4, pady=2)

        self.log_time_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sim, text="Logarithmic time axis", variable=self.log_time_var).grid(
            row=10, column=0, columnspan=3, sticky="w", padx=4, pady=2
        )

        button_frame = ttk.Frame(controls)
        button_frame.pack(fill="x", pady=8)
        ttk.Button(button_frame, text="Run", command=self.run_simulation).pack(side="left", padx=3)
        ttk.Button(button_frame, text="Normalize initial", command=self.normalize_initial_populations).pack(
            side="left", padx=3
        )
        ttk.Button(button_frame, text="Reset defaults", command=self.reset_defaults).pack(side="left", padx=3)
        ttk.Button(
            button_frame,
            text="Export population PNG",
            command=lambda: self.open_export_options("png"),
        ).pack(side="left", padx=3)
        ttk.Button(
            button_frame,
            text="Export population CSV",
            command=lambda: self.open_export_options("csv"),
        ).pack(side="left", padx=3)
        ttk.Button(
            controls,
            text="Export population graph (combined PNG)",
            command=self.open_population_graph_export,
        ).pack(fill="x", pady=(0, 4))

        ttk.Button(
            controls,
            text="Open intensity sweep / ΔPL/PL tool",
            command=self.open_intensity_sweep_tool,
        ).pack(fill="x", pady=(0, 4))

        ttk.Button(
            controls,
            text="Open ΔPL/PL versus time tool",
            command=self.open_contrast_time_tool,
        ).pack(fill="x", pady=(0, 8))

        stationary = ttk.LabelFrame(controls, text="Stationary populations", padding=6)
        stationary.pack(fill="x", pady=(4, 8))

        self.stationary_vars = {}
        for row, label in enumerate(["S0", "S1", "Tx", "Ty", "Tz", "T1 total"]):
            ttk.Label(stationary, text=label).grid(
                row=row, column=0, sticky="w", padx=4, pady=2
            )
            var = tk.StringVar(value="—")
            self.stationary_vars[label] = var
            ttk.Label(
                stationary,
                textvariable=var,
                width=16,
                anchor="e",
            ).grid(row=row, column=1, sticky="e", padx=4, pady=2)

        ttk.Button(
            stationary,
            text="Copy stationary values",
            command=self.copy_stationary_values,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 2))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=255,
        ).pack(fill="x", pady=(0, 8))

        # Run with Enter from any editable field.
        self.bind("<Return>", lambda _event: self.run_simulation())

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.summary = tk.Text(plot_frame, height=9, wrap="word")
        self.summary.pack(fill="x", pady=(4, 0))

    def _set_defaults(self) -> None:
        defaults = {
            # Optical values: deliberately editable assumptions
            "sigma": "1e-16",
            "intensity": "100",
            "wavelength_nm": "520",

            # Reference values from Fig. S2
            "k10": "4.2e7",
            "kISC": "6.9e7",
            "kx": "2.8e4",
            "ky": "0.6e4",
            "kz": "0.2e4",
            "wxy": "0.4e4",
            "wyx": "0.4e4",
            "wxz": "1.1e4",
            "wzx": "1.1e4",
            "wyz": "2.2e4",
            "wzy": "2.2e4",
            "Px": "0.76",
            "Py": "0.16",
            "Pz": "0.08",
            "rabi_xy_mhz": "0",
            "rabi_xz_mhz": "0",
            "rabi_yz_mhz": "0",

            # Simulation
            "t_start": "1e-10",
            "t_end": "5e-3",
            "n_points": "1500",
            "S0_0": "1",
            "S1_0": "0",
            "Tx_0": "0",
            "Ty_0": "0",
            "Tz_0": "0",
        }
        for key, value in defaults.items():
            self.vars[key].set(value)

    def normalize_initial_populations(self) -> None:
        """Normalize the five editable initial populations so that they sum to one."""
        try:
            keys = ["S0_0", "S1_0", "Tx_0", "Ty_0", "Tz_0"]
            values = np.array([self._read_float(key) for key in keys], dtype=float)
            if np.any(values < 0):
                raise ValueError("Initial populations must be non-negative.")
            total = values.sum()
            if total <= 0:
                raise ValueError("At least one initial population must be greater than zero.")
            values /= total
            for key, value in zip(keys, values):
                self.vars[key].set(f"{value:.10g}")
        except Exception as exc:
            messagebox.showerror("Normalization error", str(exc))

    def set_all_curves(self, visible: bool) -> None:
        for var in self.plot_vars.values():
            var.set(visible)
        self.update_plot()

    def update_plot(self) -> None:
        if self.last_data is None:
            return

        t, y = self.last_data
        curves = {
            "S0": y[0],
            "S1": y[1],
            "Tx": y[2],
            "Ty": y[3],
            "Tz": y[4],
            "T1 total": y[2] + y[3] + y[4],
        }

        self.ax.clear()
        selected_values = []

        for label, values in curves.items():
            if not self.plot_vars.get(label, tk.BooleanVar(value=True)).get():
                continue
            if label == "T1 total":
                self.ax.plot(t, values, "--", linewidth=2, label="T1 = Tx + Ty + Tz")
            else:
                self.ax.plot(t, values, label=label)
            selected_values.append(values)

        if self.log_time_var.get() and np.all(t > 0):
            self.ax.set_xscale("log")
        else:
            self.ax.set_xscale("linear")

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Population")
        self.ax.set_title("Population dynamics from Eq. (2)")
        self.ax.grid(False)

        if selected_values:
            values = np.concatenate(selected_values)
            finite = values[np.isfinite(values)]
            if finite.size:
                ymin = float(np.min(finite))
                ymax = float(np.max(finite))
                span = ymax - ymin
                margin = 0.06 * span if span > 0 else max(0.02, 0.06 * abs(ymax))
                lower = max(0.0, ymin - margin)
                upper = ymax + margin
                if upper <= lower:
                    upper = lower + 0.05
                self.ax.set_ylim(lower, upper)
            self.ax.legend(ncol=2)
        else:
            self.ax.text(
                0.5, 0.5, "No population curve selected",
                transform=self.ax.transAxes,
                ha="center", va="center",
            )
            self.ax.set_ylim(0.0, 1.0)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def copy_stationary_values(self) -> None:
        if self.last_stationary is None:
            messagebox.showwarning("No stationary state", "Run a simulation first.")
            return

        ss = self.last_stationary
        text = (
            f"S0={ss[0]:.10g}\n"
            f"S1={ss[1]:.10g}\n"
            f"Tx={ss[2]:.10g}\n"
            f"Ty={ss[3]:.10g}\n"
            f"Tz={ss[4]:.10g}\n"
            f"T1_total={ss[2] + ss[3] + ss[4]:.10g}"
        )
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Stationary populations copied to clipboard.")

    def reset_defaults(self) -> None:
        self._set_defaults()
        self.run_simulation()

    def _read_float(self, key: str) -> float:
        try:
            value = float(self.vars[key].get())
        except ValueError as exc:
            raise ValueError(f"Invalid numerical value for {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value

    def _collect_parameters(self) -> tuple[dict[str, float], np.ndarray, float, float, int]:
        sigma = self._read_float("sigma")
        intensity = self._read_float("intensity")
        wavelength_nm = self._read_float("wavelength_nm")

        if sigma < 0 or intensity < 0 or wavelength_nm <= 0:
            raise ValueError("σ and I must be non-negative; laser wavelength must be positive.")

        wavelength_m = wavelength_nm * 1e-9
        photon_energy = H * C / wavelength_m

        # Photon flux = I/E_photon, with I in W/cm² and σ in cm².
        # Therefore k01 = σ I / E_photon = σ I λ / (h c).
        k01 = sigma * intensity / photon_energy

        p = {"k01": k01}
        for key in ["k10", "kISC", "kx", "ky", "kz", "wxy", "wyx", "wxz", "wzx", "wyz", "wzy"]:
            p[key] = self._read_float(key)
            if p[key] < 0:
                raise ValueError(f"{key} must be non-negative.")

        for key in ["Px", "Py", "Pz"]:
            p[key] = self._read_float(key)
            if p[key] < 0:
                raise ValueError(f"{key} must be non-negative.")

        for pair, key in [
            ("xy", "rabi_xy_mhz"),
            ("xz", "rabi_xz_mhz"),
            ("yz", "rabi_yz_mhz"),
        ]:
            rabi_mhz = self._read_float(key)
            if rabi_mhz < 0:
                raise ValueError(f"{key} must be non-negative.")
            p[f"gamma_{pair}"] = 2.0 * math.pi * rabi_mhz * 1e6

        prob_sum = p["Px"] + p["Py"] + p["Pz"]
        if prob_sum <= 0:
            raise ValueError("At least one of Px, Py or Pz must be greater than zero.")

        # Treat Px, Py, Pz as relative branching weights and normalize them.
        # Thus entries such as 0.333, 0.333, 0.333 are accepted.
        p["Px"] /= prob_sum
        p["Py"] /= prob_sum
        p["Pz"] /= prob_sum

        y0 = np.array(
            [
                self._read_float("S0_0"),
                self._read_float("S1_0"),
                self._read_float("Tx_0"),
                self._read_float("Ty_0"),
                self._read_float("Tz_0"),
            ],
            dtype=float,
        )
        if np.any(y0 < 0):
            raise ValueError("Initial populations must be non-negative.")
        initial_sum = y0.sum()
        if initial_sum <= 0:
            raise ValueError("At least one initial population must be greater than zero.")

        # Normalize automatically so the user can enter either populations
        # or relative weights.
        y0 /= initial_sum

        t_start = self._read_float("t_start")
        t_end = self._read_float("t_end")
        n_points = int(self._read_float("n_points"))

        if t_start < 0 or t_end <= t_start:
            raise ValueError("Require 0 ≤ start time < end time.")
        if n_points < 20:
            raise ValueError("Use at least 20 time points.")

        return p, y0, t_start, t_end, n_points

    @staticmethod
    def propagate_linear_system(
        A: np.ndarray,
        y0: np.ndarray,
        times: np.ndarray,
        t0: float,
    ) -> np.ndarray:
        """Exact propagation for the constant linear system dN/dt = A N.

        For this 5x5 problem, evaluating exp[A(t-t0)] is fast and avoids
        the tiny adaptive time steps caused by very large microwave rates.
        """
        result = np.empty((len(y0), len(times)), dtype=float)
        for index, time_value in enumerate(times):
            dt = float(time_value - t0)
            if dt <= 0:
                result[:, index] = y0
            else:
                result[:, index] = expm(A * dt) @ y0

        # Remove only roundoff-level artifacts.
        result[np.abs(result) < 1e-14] = 0.0
        return result

    def run_simulation(self) -> None:
        try:
            p, y0, t_start, t_end, n_points = self._collect_parameters()
            self.last_parameters = p.copy()
            self.last_initial_state = y0.copy()

            # Display the actual normalized values used in the calculation.
            for key in ["Px", "Py", "Pz"]:
                self.vars[key].set(f"{p[key]:.10g}")
            for key, value in zip(["S0_0", "S1_0", "Tx_0", "Ty_0", "Tz_0"], y0):
                self.vars[key].set(f"{value:.10g}")

            A = rate_matrix(p)

            if self.log_time_var.get() and t_start > 0:
                t_eval = np.geomspace(t_start, t_end, n_points)
            else:
                t_eval = np.linspace(t_start, t_end, n_points)

            method = self.method_var.get()

            if method == "Matrix exponential":
                time_values = t_eval
                y = self.propagate_linear_system(A, y0, time_values, t_start)
            else:
                sol = solve_ivp(
                    fun=lambda t, state: A @ state,
                    t_span=(t_start, t_end),
                    y0=y0,
                    method=method,
                    t_eval=t_eval,
                    rtol=1e-8,
                    atol=1e-11,
                )
                if not sol.success:
                    raise RuntimeError(sol.message)
                time_values = sol.t
                y = sol.y
                y[np.abs(y) < 1e-13] = 0.0

            ss = steady_state(A)
            self.last_stationary = ss.copy()

            stationary_display = {
                "S0": ss[0],
                "S1": ss[1],
                "Tx": ss[2],
                "Ty": ss[3],
                "Tz": ss[4],
                "T1 total": ss[2] + ss[3] + ss[4],
            }
            for label, value in stationary_display.items():
                self.stationary_vars[label].set(f"{value:.8f}")

            eigvals = np.linalg.eigvals(A)
            nonzero = [ev for ev in eigvals if abs(ev) > 1e-10]
            slowest_tau = max((-1.0 / ev.real for ev in nonzero if ev.real < 0), default=float("nan"))

            self.last_data = (time_values.copy(), y.copy())
            self.k01_label.config(text=f"{p['k01']:.4e} s⁻¹")

            self.update_plot()

            summary_lines = [
                f"k01 = σIλ/(hc) = {p['k01']:.6e} s⁻¹",
                f"ΓMW,xy = {p['gamma_xy']:.6e} s⁻¹",
                f"ΓMW,xz = {p['gamma_xz']:.6e} s⁻¹",
                f"ΓMW,yz = {p['gamma_yz']:.6e} s⁻¹",
                "",
                "Stationary populations:",
                f"S0 = {ss[0]:.8f}",
                f"S1 = {ss[1]:.8f}",
                f"Tx = {ss[2]:.8f}",
                f"Ty = {ss[3]:.8f}",
                f"Tz = {ss[4]:.8f}",
                f"T1 = {ss[2] + ss[3] + ss[4]:.8f}",
                f"Sum = {ss.sum():.12f}",
                "",
                f"Approximate slowest relaxation time = {slowest_tau:.4e} s",
                f"Population conservation error during integration = {np.max(np.abs(y.sum(axis=0) - 1)):.3e}",
            ]
            self.summary.delete("1.0", tk.END)
            self.summary.insert(tk.END, "\n".join(summary_lines))
            max_mw_rate = max(p["gamma_xy"], p["gamma_xz"], p["gamma_yz"])
            intrinsic_scale = max(
                p["k10"], p["kISC"], p["kx"], p["ky"], p["kz"],
                p["wxy"], p["wyx"], p["wxz"], p["wzx"], p["wyz"], p["wzy"],
            )
            stiffness_ratio = max_mw_rate / intrinsic_scale if intrinsic_scale > 0 else 0.0
            self.status_var.set(
                f"Simulation updated with {method}. "
                f"Maximum MW/intrinsic rate ratio: {stiffness_ratio:.3e}."
            )

        except Exception as exc:
            self.status_var.set(f"Simulation not updated: {exc}")
            messagebox.showerror("Simulation error", str(exc))

    def _state_at_time(
        self,
        p: dict[str, float],
        y0: np.ndarray,
        t_start: float,
        t_eval: float,
        method: str,
    ) -> np.ndarray:
        """Return the state at one chosen time.

        The matrix exponential is used for this constant linear system,
        independently of the plotting integrator. This makes intensity
        sweeps fast even for very large microwave mixing rates.
        """
        if t_eval <= t_start:
            return y0.copy()

        A = rate_matrix(p)
        state = expm(A * float(t_eval - t_start)) @ y0
        state[np.abs(state) < 1e-14] = 0.0
        return state

    def open_intensity_sweep_tool(self) -> None:
        """Open a full-parameter intensity-sweep tool.

        The sweep window contains the same physical parameters as the main
        population window. The only parameter varied automatically is the
        laser intensity.
        """
        try:
            p_main, y0_main, t_start_main, t_end_main, _ = self._collect_parameters()
        except Exception as exc:
            messagebox.showerror("Parameter error", str(exc))
            return

        window = tk.Toplevel(self)
        window.title("Laser-intensity sweep — ΔPL/PL")
        window.geometry("1450x900")
        window.minsize(1150, 720)

        outer = ttk.Frame(window)
        outer.pack(fill="both", expand=True)

        # Scrollable parameter panel, matching the main application style.
        controls_container = ttk.Frame(outer)
        controls_container.pack(side="left", fill="y")

        controls_canvas = tk.Canvas(
            controls_container,
            width=330,
            highlightthickness=0,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls_scrollbar.pack(side="right", fill="y")
        controls_canvas.pack(side="left", fill="y", expand=False)

        controls = ttk.Frame(controls_canvas, padding=8)
        controls_window = controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )

        def update_scroll_region(_event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def resize_controls_width(event):
            controls_canvas.itemconfigure(controls_window, width=event.width)

        controls.bind("<Configure>", update_scroll_region)
        controls_canvas.bind("<Configure>", resize_controls_width)

        def on_mousewheel(event):
            if event.delta:
                controls_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif event.num == 4:
                controls_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                controls_canvas.yview_scroll(1, "units")

        def bind_mousewheel(_event):
            controls_canvas.bind_all("<MouseWheel>", on_mousewheel)
            controls_canvas.bind_all("<Button-4>", on_mousewheel)
            controls_canvas.bind_all("<Button-5>", on_mousewheel)

        def unbind_mousewheel(_event):
            controls_canvas.unbind_all("<MouseWheel>")
            controls_canvas.unbind_all("<Button-4>")
            controls_canvas.unbind_all("<Button-5>")

        controls_canvas.bind("<Enter>", bind_mousewheel)
        controls_canvas.bind("<Leave>", unbind_mousewheel)

        plot_frame = ttk.Frame(outer, padding=4)
        plot_frame.pack(side="right", fill="both", expand=True)

        fields: dict[str, tk.StringVar] = {}

        def add_field(parent, row, label, key, default, unit="", width=14):
            ttk.Label(parent, text=label).grid(
                row=row, column=0, sticky="w", padx=4, pady=2
            )
            var = tk.StringVar(value=str(default))
            fields[key] = var
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=1, padx=4, pady=2
            )
            ttk.Label(parent, text=unit).grid(
                row=row, column=2, sticky="w", padx=4, pady=2
            )

        # Optical parameters
        optical = ttk.LabelFrame(controls, text="Optical parameters", padding=6)
        optical.pack(fill="x", pady=4)
        add_field(optical, 0, "Cross section σ", "sigma", self.vars["sigma"].get(), "cm²")
        add_field(optical, 1, "Laser wavelength λ", "wavelength_nm", self.vars["wavelength_nm"].get(), "nm")

        # Intensity sweep settings
        sweep = ttk.LabelFrame(controls, text="Intensity sweep", padding=6)
        sweep.pack(fill="x", pady=4)
        current_i = self._read_float("intensity")
        add_field(sweep, 0, "Minimum intensity", "i_min", max(current_i / 100, 1e-9), "W/cm²")
        add_field(sweep, 1, "Maximum intensity", "i_max", current_i * 10, "W/cm²")
        add_field(sweep, 2, "Number of intensities", "n_i", 120, "")
        add_field(sweep, 3, "Start time", "t_start", self.vars["t_start"].get(), "s")
        add_field(sweep, 4, "Evaluation time", "eval_time", self.vars["t_end"].get(), "s")

        log_i_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sweep,
            text="Logarithmic intensity axis",
            variable=log_i_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=3)

        # Intrinsic rates
        rates = ttk.LabelFrame(controls, text="Intrinsic rates", padding=6)
        rates.pack(fill="x", pady=4)
        rate_rows = [
            ("k10", "k10"),
            ("kISC", "kISC"),
            ("kx", "kx"),
            ("ky", "ky"),
            ("kz", "kz"),
            ("wxy", "wxy"),
            ("wyx", "wyx"),
            ("wxz", "wxz"),
            ("wzx", "wzx"),
            ("wyz", "wyz"),
            ("wzy", "wzy"),
        ]
        for row, (label, key) in enumerate(rate_rows):
            add_field(rates, row, label, key, self.vars[key].get(), "s⁻¹")

        # Branching probabilities
        probs = ttk.LabelFrame(controls, text="ISC branching probabilities", padding=6)
        probs.pack(fill="x", pady=4)
        add_field(probs, 0, "Px", "Px", self.vars["Px"].get())
        add_field(probs, 1, "Py", "Py", self.vars["Py"].get())
        add_field(probs, 2, "Pz", "Pz", self.vars["Pz"].get())

        # Microwave drives
        mw = ttk.LabelFrame(controls, text="Microwave drives", padding=6)
        mw.pack(fill="x", pady=4)
        ttk.Label(
            mw,
            text=(
                "Driven calculation uses these Ω/2π values. "
                "Control calculation uses Ω = 0 for all three pairs."
            ),
            wraplength=285,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))
        add_field(mw, 1, "Txy drive Ωxy/2π", "rabi_xy_mhz", self.vars["rabi_xy_mhz"].get(), "MHz")
        add_field(mw, 2, "Txz drive Ωxz/2π", "rabi_xz_mhz", self.vars["rabi_xz_mhz"].get(), "MHz")
        add_field(mw, 3, "Tyz drive Ωyz/2π", "rabi_yz_mhz", self.vars["rabi_yz_mhz"].get(), "MHz")

        # Initial populations
        initial = ttk.LabelFrame(controls, text="Initial populations", padding=6)
        initial.pack(fill="x", pady=4)
        ttk.Label(
            initial,
            text="These values are normalized automatically.",
            wraplength=285,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))
        for row, (label, key) in enumerate(
            [
                ("Initial S0", "S0_0"),
                ("Initial S1", "S1_0"),
                ("Initial Tx", "Tx_0"),
                ("Initial Ty", "Ty_0"),
                ("Initial Tz", "Tz_0"),
            ],
            start=1,
        ):
            add_field(initial, row, label, key, self.vars[key].get())

        # Plot options
        plot_options = ttk.LabelFrame(controls, text="Plot options", padding=6)
        plot_options.pack(fill="x", pady=4)
        show_s1_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            plot_options,
            text="Show S1 driven and control",
            variable=show_s1_var,
        ).pack(anchor="w", padx=4, pady=3)

        # Result area
        result_text = tk.Text(controls, width=39, height=15, wrap="word")
        result_text.pack(fill="x", pady=6)

        fig = plt.Figure(figsize=(9.0, 7.0))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        last_sweep: dict[str, np.ndarray] = {}

        def read_float(key: str) -> float:
            try:
                value = float(fields[key].get())
            except ValueError as exc:
                raise ValueError(f"Invalid numerical value for {key}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            return value

        def collect_sweep_parameters():
            sigma = read_float("sigma")
            wavelength_nm = read_float("wavelength_nm")
            if sigma < 0 or wavelength_nm <= 0:
                raise ValueError("σ must be non-negative and λ must be positive.")

            p = {}
            for key in [
                "k10", "kISC", "kx", "ky", "kz",
                "wxy", "wyx", "wxz", "wzx", "wyz", "wzy",
            ]:
                p[key] = read_float(key)
                if p[key] < 0:
                    raise ValueError(f"{key} must be non-negative.")

            for key in ["Px", "Py", "Pz"]:
                p[key] = read_float(key)
                if p[key] < 0:
                    raise ValueError(f"{key} must be non-negative.")

            prob_sum = p["Px"] + p["Py"] + p["Pz"]
            if prob_sum <= 0:
                raise ValueError("At least one of Px, Py or Pz must be positive.")
            p["Px"] /= prob_sum
            p["Py"] /= prob_sum
            p["Pz"] /= prob_sum

            for pair, key in [
                ("xy", "rabi_xy_mhz"),
                ("xz", "rabi_xz_mhz"),
                ("yz", "rabi_yz_mhz"),
            ]:
                rabi_mhz = read_float(key)
                if rabi_mhz < 0:
                    raise ValueError(f"{key} must be non-negative.")
                p[f"gamma_{pair}"] = 2.0 * math.pi * rabi_mhz * 1e6

            y0 = np.array(
                [
                    read_float("S0_0"),
                    read_float("S1_0"),
                    read_float("Tx_0"),
                    read_float("Ty_0"),
                    read_float("Tz_0"),
                ],
                dtype=float,
            )
            if np.any(y0 < 0):
                raise ValueError("Initial populations must be non-negative.")
            total = y0.sum()
            if total <= 0:
                raise ValueError("At least one initial population must be positive.")
            y0 /= total

            i_min = read_float("i_min")
            i_max = read_float("i_max")
            n_i = int(read_float("n_i"))
            t_start = read_float("t_start")
            eval_time = read_float("eval_time")

            if i_min < 0 or i_max <= i_min:
                raise ValueError("Require 0 ≤ minimum intensity < maximum intensity.")
            if n_i < 2:
                raise ValueError("Use at least two intensity points.")
            if t_start < 0:
                raise ValueError("Start time must be non-negative.")
            if eval_time < t_start:
                raise ValueError("Evaluation time must be later than the start time.")

            return p, y0, sigma, wavelength_nm, i_min, i_max, n_i, t_start, eval_time

        def run_sweep():
            try:
                (
                    p_base,
                    y0,
                    sigma,
                    wavelength_nm,
                    i_min,
                    i_max,
                    n_i,
                    t_start,
                    eval_time,
                ) = collect_sweep_parameters()

                if log_i_var.get():
                    if i_min <= 0:
                        raise ValueError(
                            "Minimum intensity must be positive for a logarithmic sweep."
                        )
                    intensities = np.geomspace(i_min, i_max, n_i)
                else:
                    intensities = np.linspace(i_min, i_max, n_i)

                photon_energy = H * C / (wavelength_nm * 1e-9)

                contrast = np.empty(n_i)
                s1_drive = np.empty(n_i)
                s1_control = np.empty(n_i)

                for idx, intensity in enumerate(intensities):
                    p_drive = p_base.copy()
                    p_drive["k01"] = sigma * intensity / photon_energy

                    p_control = p_drive.copy()
                    p_control["gamma_xy"] = 0.0
                    p_control["gamma_xz"] = 0.0
                    p_control["gamma_yz"] = 0.0

                    y_drive = self._state_at_time(
                        p_drive,
                        y0,
                        t_start,
                        eval_time,
                        "Matrix exponential",
                    )
                    y_control = self._state_at_time(
                        p_control,
                        y0,
                        t_start,
                        eval_time,
                        "Matrix exponential",
                    )

                    s1_drive[idx] = y_drive[1]
                    s1_control[idx] = y_control[1]

                    if abs(s1_control[idx]) < 1e-30:
                        contrast[idx] = np.nan
                    else:
                        contrast[idx] = (
                            s1_drive[idx] - s1_control[idx]
                        ) / s1_control[idx]

                last_sweep.clear()
                last_sweep.update(
                    intensity=intensities,
                    contrast=contrast,
                    s1_drive=s1_drive,
                    s1_control=s1_control,
                )

                ax.clear()
                ax.plot(intensities, 100.0 * contrast, label="ΔPL/PL")

                if log_i_var.get():
                    ax.set_xscale("log")
                else:
                    ax.set_xscale("linear")

                ax.set_xlabel("Laser intensity (W/cm²)")
                ax.set_ylabel("ΔPL/PL (%)")
                ax.grid(False)

                if show_s1_var.get():
                    ax2 = ax.twinx()
                    ax2.plot(intensities, s1_drive, "--", label="S1 driven")
                    ax2.plot(intensities, s1_control, ":", label="S1 control")
                    ax2.set_ylabel("S1 population")
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2)
                else:
                    ax.legend()

                finite = np.isfinite(contrast)
                result = [
                    f"Start time: {t_start:.6e} s",
                    f"Evaluation time: {eval_time:.6e} s",
                    f"σ: {sigma:.6e} cm²",
                    f"λ: {wavelength_nm:.6g} nm",
                    "",
                    f"Ωxy/2π: {read_float('rabi_xy_mhz'):.6g} MHz",
                    f"Ωxz/2π: {read_float('rabi_xz_mhz'):.6g} MHz",
                    f"Ωyz/2π: {read_float('rabi_yz_mhz'):.6g} MHz",
                ]

                if np.any(finite):
                    imax = np.nanargmax(contrast)
                    imin = np.nanargmin(contrast)
                    result.extend(
                        [
                            "",
                            "Largest positive contrast:",
                            f"I = {intensities[imax]:.6e} W/cm²",
                            f"ΔPL/PL = {100.0 * contrast[imax]:.6e} %",
                            "",
                            "Most negative contrast:",
                            f"I = {intensities[imin]:.6e} W/cm²",
                            f"ΔPL/PL = {100.0 * contrast[imin]:.6e} %",
                        ]
                    )
                else:
                    result.extend(
                        [
                            "",
                            "S1(control) was zero or numerically negligible "
                            "over the entire sweep."
                        ]
                    )

                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "\n".join(result))

                fig.tight_layout()
                canvas.draw_idle()

            except Exception as exc:
                messagebox.showerror("Sweep error", str(exc), parent=window)

        def export_sweep_csv():
            if not last_sweep:
                messagebox.showwarning("No sweep", "Run the intensity sweep first.", parent=window)
                return
            folder = filedialog.askdirectory(parent=window, title="Choose folder for intensity-sweep CSV files")
            if not folder:
                return
            out = Path(folder)
            curves = {
                "deltaPL_over_PL": last_sweep["contrast"],
                "S1_control": last_sweep["s1_control"],
                "S1_driven": last_sweep["s1_drive"],
            }
            for name, values in curves.items():
                np.savetxt(
                    out / f"intensity_{name}.csv",
                    np.column_stack([last_sweep["intensity"], values]),
                    delimiter=",",
                    header=f"intensity_W_cm2,{name}",
                    comments="",
                )
            messagebox.showinfo("CSV export complete", f"Three separate CSV files saved in:\n{folder}", parent=window)

        def export_sweep_png():
            if not last_sweep:
                messagebox.showwarning("No sweep", "Run the intensity sweep first.", parent=window)
                return
            folder = filedialog.askdirectory(parent=window, title="Choose folder for intensity-sweep PNG files")
            if not folder:
                return
            out = Path(folder)
            curves = [
                ("deltaPL_over_PL", 100.0 * last_sweep["contrast"], "ΔPL/PL (%)"),
                ("S1_control", last_sweep["s1_control"], "S1 population"),
                ("S1_driven", last_sweep["s1_drive"], "S1 population"),
            ]
            for name, values, ylabel in curves:
                export_fig, export_ax = plt.subplots(figsize=(9, 6))
                export_ax.plot(last_sweep["intensity"], values, label=name.replace("_", " "))
                export_ax.set_xscale("log" if log_i_var.get() else "linear")
                export_ax.set_xlabel("Laser intensity (W/cm²)")
                export_ax.set_ylabel(ylabel)
                export_ax.set_title(name.replace("_", " "))
                export_ax.grid(False)
                export_ax.legend()
                export_fig.tight_layout()
                export_fig.savefig(out / f"intensity_{name}.png", dpi=300, bbox_inches="tight")
                plt.close(export_fig)
            messagebox.showinfo("PNG export complete", f"Three separate PNG files saved in:\n{folder}", parent=window)

        def export_delta_intensity_graph():
            if not last_sweep:
                messagebox.showwarning("No sweep", "Run the intensity sweep first.", parent=window)
                return

            options = tk.Toplevel(window)
            options.title("ΔPL/PL graph export options")
            options.transient(window)
            options.resizable(False, False)
            frame = ttk.Frame(options, padding=12)
            frame.pack(fill="both", expand=True)

            fields = [
                ("Graph title", export_title_var),
                ("Title font size", export_title_size_var),
                ("Axis-title font size", export_axis_size_var),
                ("Tick-label font size", export_tick_size_var),
                ("Legend font size", export_legend_size_var),
                ("PNG resolution (dpi)", export_dpi_var),
            ]
            for row, (label, var) in enumerate(fields):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
                ttk.Entry(frame, textvariable=var, width=42 if row == 0 else 12).grid(
                    row=row, column=1, sticky="ew", padx=4, pady=4
                )

            def save_graph():
                try:
                    title_size = float(export_title_size_var.get())
                    axis_size = float(export_axis_size_var.get())
                    tick_size = float(export_tick_size_var.get())
                    legend_size = float(export_legend_size_var.get())
                    dpi = int(float(export_dpi_var.get()))
                    if min(title_size, axis_size, tick_size, legend_size, dpi) <= 0:
                        raise ValueError("All font sizes and the DPI must be positive.")
                    path = filedialog.asksaveasfilename(
                        parent=options,
                        title="Export ΔPL/PL intensity graph",
                        defaultextension=".png",
                        filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                        initialfile="deltaPL_over_PL_vs_intensity.png",
                    )
                    if not path:
                        return
                    export_fig, export_ax = plt.subplots(figsize=(10, 7))
                    export_ax.plot(
                        last_sweep["intensity"],
                        100.0 * last_sweep["contrast"],
                        label="ΔPL/PL",
                    )
                    export_ax.axhline(0.0, linewidth=0.8)
                    export_ax.set_xscale("log" if log_i_var.get() else "linear")
                    export_ax.set_xlabel("Laser intensity (W/cm²)", fontsize=axis_size)
                    export_ax.set_ylabel("ΔPL/PL (%)", fontsize=axis_size)
                    graph_title = export_title_var.get().strip()
                    if graph_title:
                        export_ax.set_title(graph_title, fontsize=title_size)
                    export_ax.tick_params(axis="both", labelsize=tick_size)
                    export_ax.legend(fontsize=legend_size)
                    export_ax.grid(False)
                    export_fig.tight_layout()
                    export_fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close(export_fig)
                    messagebox.showinfo("PNG export complete", f"Saved:\n{path}", parent=options)
                    options.destroy()
                except Exception as exc:
                    messagebox.showerror("Export error", str(exc), parent=options)

            ttk.Button(frame, text="Choose file and export graph", command=save_graph).grid(
                row=len(fields), column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 2)
            )
            frame.columnconfigure(1, weight=1)

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Run intensity sweep", command=run_sweep).pack(side="left", padx=3)
        ttk.Button(buttons, text="Export ΔPL/PL graph", command=export_delta_intensity_graph).pack(side="left", padx=3)
        ttk.Button(buttons, text="Export PNG curves", command=export_sweep_png).pack(side="left", padx=3)
        ttk.Button(buttons, text="Export CSV curves", command=export_sweep_csv).pack(side="left", padx=3)

        window.bind("<Return>", lambda _event: run_sweep())
        run_sweep()


    def open_export_options(self, export_format: str) -> None:
        """Export each selected population curve separately as PNG or CSV."""
        if self.last_data is None:
            messagebox.showwarning("No data", "Run a simulation first.")
            return

        window = tk.Toplevel(self)
        window.title(f"Population {export_format.upper()} export options")
        window.transient(self)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        title_var = tk.StringVar(value="Population dynamics from Eq. (2)")
        title_size_var = tk.StringVar(value="18")
        axis_size_var = tk.StringVar(value="16")
        tick_size_var = tk.StringVar(value="12")
        legend_size_var = tk.StringVar(value="12")
        dpi_var = tk.StringVar(value="300")

        fields = [
            ("Graph title", title_var),
            ("Title font size", title_size_var),
            ("Axis-title font size", axis_size_var),
            ("Tick-label font size", tick_size_var),
            ("Legend font size", legend_size_var),
            ("PNG resolution (dpi)", dpi_var),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(frame, textvariable=var, width=35 if row == 0 else 12).grid(
                row=row, column=1, sticky="ew", padx=4, pady=4
            )

        ttk.Label(
            frame,
            text=(f"Each selected population curve will be exported as its own "
                  f"{export_format.upper()} file. PNG and CSV exports are intentionally separate."),
            wraplength=430,
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", padx=4, pady=(8, 4))

        def do_export():
            try:
                title_size = float(title_size_var.get())
                axis_size = float(axis_size_var.get())
                tick_size = float(tick_size_var.get())
                legend_size = float(legend_size_var.get())
                dpi = int(float(dpi_var.get()))
                if min(title_size, axis_size, tick_size, legend_size, dpi) <= 0:
                    raise ValueError("All font sizes and the DPI must be positive.")

                folder = filedialog.askdirectory(parent=window, title=f"Choose folder for {export_format.upper()} files")
                if not folder:
                    return
                t, y = self.last_data
                curves = {
                    "S0": y[0], "S1": y[1], "Tx": y[2], "Ty": y[3], "Tz": y[4],
                    "T1_total": y[2] + y[3] + y[4],
                }
                selected = {
                    key: values for key, values in curves.items()
                    if self.plot_vars["T1 total" if key == "T1_total" else key].get()
                }
                if not selected:
                    raise ValueError("Select at least one population curve before exporting.")
                out = Path(folder)
                for name, values in selected.items():
                    if export_format == "csv":
                        np.savetxt(
                            out / f"population_{name}.csv",
                            np.column_stack([t, values]),
                            delimiter=",",
                            header=f"time_s,{name}",
                            comments="",
                        )
                    elif export_format == "png":
                        export_fig, export_ax = plt.subplots(figsize=(9, 6))
                        export_ax.plot(t, values, label=name.replace("_", " "))
                        export_ax.set_xscale("log" if self.log_time_var.get() and np.all(t > 0) else "linear")
                        export_ax.set_xlabel("Time (s)", fontsize=axis_size)
                        export_ax.set_ylabel("Population", fontsize=axis_size)
                        graph_title = title_var.get().strip()
                        if graph_title:
                            export_ax.set_title(f"{graph_title} — {name.replace('_', ' ')}", fontsize=title_size)
                        export_ax.tick_params(axis="both", labelsize=tick_size)
                        export_ax.legend(fontsize=legend_size)
                        export_ax.grid(False)
                        export_fig.tight_layout()
                        export_fig.savefig(out / f"population_{name}.png", dpi=dpi, bbox_inches="tight")
                        plt.close(export_fig)
                    else:
                        raise ValueError("Unknown export format.")
                messagebox.showinfo(
                    "Export complete",
                    f"{len(selected)} separate {export_format.upper()} file(s) saved in:\n{folder}",
                    parent=window,
                )
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Export error", str(exc), parent=window)

        ttk.Button(frame, text=f"Choose folder and export {export_format.upper()}", command=do_export).grid(
            row=len(fields)+1, column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 2)
        )
        frame.columnconfigure(1, weight=1)

    def open_population_graph_export(self) -> None:
        """Export one combined PNG containing all currently selected population curves."""
        if self.last_data is None:
            messagebox.showwarning("No data", "Run a simulation first.")
            return

        window = tk.Toplevel(self)
        window.title("Population graph export")
        window.transient(self)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)

        title_var = tk.StringVar(value="Évolution des Populations des Niveaux dans le Temps")
        title_size_var = tk.StringVar(value="18")
        axis_size_var = tk.StringVar(value="16")
        tick_size_var = tk.StringVar(value="12")
        legend_size_var = tk.StringVar(value="12")
        dpi_var = tk.StringVar(value="300")

        fields = [
            ("Graph title", title_var),
            ("Title font size", title_size_var),
            ("Axis-title font size", axis_size_var),
            ("Tick-label font size", tick_size_var),
            ("Legend font size", legend_size_var),
            ("PNG resolution (dpi)", dpi_var),
        ]
        for row, (label, var) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(frame, textvariable=var, width=48 if row == 0 else 12).grid(
                row=row, column=1, sticky="ew", padx=4, pady=4
            )

        ttk.Label(
            frame,
            text="The exported graph contains all population curves currently selected in the main window.",
            wraplength=470,
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", padx=4, pady=(8, 4))

        def export_graph():
            try:
                title_size = float(title_size_var.get())
                axis_size = float(axis_size_var.get())
                tick_size = float(tick_size_var.get())
                legend_size = float(legend_size_var.get())
                dpi = int(float(dpi_var.get()))
                if min(title_size, axis_size, tick_size, legend_size, dpi) <= 0:
                    raise ValueError("All font sizes and the DPI must be positive.")

                path = filedialog.asksaveasfilename(
                    parent=window,
                    title="Export population graph",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                    initialfile="population_dynamics.png",
                )
                if not path:
                    return

                t, y = self.last_data
                curves = {
                    "S0": y[0],
                    "S1": y[1],
                    "Tx": y[2],
                    "Ty": y[3],
                    "Tz": y[4],
                    "T1 total": y[2] + y[3] + y[4],
                }
                selected = [(name, values) for name, values in curves.items() if self.plot_vars[name].get()]
                if not selected:
                    raise ValueError("Select at least one population curve before exporting.")

                export_fig, export_ax = plt.subplots(figsize=(10, 7))
                for name, values in selected:
                    if name == "T1 total":
                        export_ax.plot(t, values, "--", linewidth=2, label="T1 = Tx + Ty + Tz")
                    else:
                        export_ax.plot(t, values, label=name)

                export_ax.set_xscale(
                    "log" if self.log_time_var.get() and np.all(t > 0) else "linear"
                )
                export_ax.set_xlabel("Time (s)", fontsize=axis_size)
                export_ax.set_ylabel("Population", fontsize=axis_size)
                graph_title = title_var.get().strip()
                if graph_title:
                    export_ax.set_title(graph_title, fontsize=title_size)
                export_ax.tick_params(axis="both", labelsize=tick_size)
                export_ax.legend(fontsize=legend_size)
                export_ax.grid(False)
                export_fig.tight_layout()
                export_fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(export_fig)
                messagebox.showinfo("PNG export complete", f"Saved:\n{path}", parent=window)
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Export error", str(exc), parent=window)

        ttk.Button(frame, text="Choose file and export combined PNG", command=export_graph).grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 2)
        )
        frame.columnconfigure(1, weight=1)

    def open_contrast_time_tool(self) -> None:
        """Plot transient ΔPL/PL using the same parameters as the population tool.

        The driven trace uses the current microwave rates. The control trace uses
        the same parameters with all microwave drives set to zero.
        """
        try:
            p, y0, t_start, t_end, n_points = self._collect_parameters()
        except Exception as exc:
            messagebox.showerror("Parameter error", str(exc))
            return

        window = tk.Toplevel(self)
        window.title("Transient ΔPL/PL")
        window.geometry("1100x760")

        top = ttk.Frame(window, padding=8)
        top.pack(fill="x")
        title_var = tk.StringVar(value="Transient ODMR contrast")
        title_size_var = tk.StringVar(value="18")
        axis_size_var = tk.StringVar(value="16")
        tick_size_var = tk.StringVar(value="12")
        legend_size_var = tk.StringVar(value="12")
        dpi_var = tk.StringVar(value="300")
        ttk.Label(top, text="Title").grid(row=0, column=0, padx=4, pady=3)
        ttk.Entry(top, textvariable=title_var, width=32).grid(row=0, column=1, padx=4, pady=3)
        ttk.Label(top, text="Title size").grid(row=0, column=2, padx=4, pady=3)
        ttk.Entry(top, textvariable=title_size_var, width=7).grid(row=0, column=3, padx=4, pady=3)
        ttk.Label(top, text="Axis-title size").grid(row=0, column=4, padx=4, pady=3)
        ttk.Entry(top, textvariable=axis_size_var, width=7).grid(row=0, column=5, padx=4, pady=3)
        ttk.Label(top, text="Tick size").grid(row=0, column=6, padx=4, pady=3)
        ttk.Entry(top, textvariable=tick_size_var, width=7).grid(row=0, column=7, padx=4, pady=3)
        ttk.Label(top, text="Legend size").grid(row=0, column=8, padx=4, pady=3)
        ttk.Entry(top, textvariable=legend_size_var, width=7).grid(row=0, column=9, padx=4, pady=3)

        fig, ax = plt.subplots(figsize=(9, 6))
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        last = {}

        def calculate():
            try:
                p_now, y0_now, t0, t1, points = self._collect_parameters()
                times = np.geomspace(t0, t1, points) if self.log_time_var.get() and t0 > 0 else np.linspace(t0, t1, points)
                p_control = p_now.copy()
                p_control["gamma_xy"] = p_control["gamma_xz"] = p_control["gamma_yz"] = 0.0
                driven = self.propagate_linear_system(rate_matrix(p_now), y0_now, times, t0)
                control = self.propagate_linear_system(rate_matrix(p_control), y0_now, times, t0)
                denom = control[1]
                contrast = np.full_like(denom, np.nan, dtype=float)
                valid = np.abs(denom) > 1e-30
                contrast[valid] = (driven[1, valid] - denom[valid]) / denom[valid]
                last.clear()
                last.update(time=times, contrast=contrast, s1_driven=driven[1], s1_control=control[1])

                ax.clear()
                ax.plot(times, 100.0 * contrast, label="ΔPL/PL")
                ax.axhline(0.0, linewidth=0.8)
                ax.set_xscale("log" if self.log_time_var.get() and np.all(times > 0) else "linear")
                ax.set_xlabel("Time (s)", fontsize=float(axis_size_var.get()))
                ax.set_ylabel("ΔPL/PL (%)", fontsize=float(axis_size_var.get()))
                if title_var.get().strip():
                    ax.set_title(title_var.get().strip(), fontsize=float(title_size_var.get()))
                ax.tick_params(axis="both", labelsize=float(tick_size_var.get()))
                ax.grid(False)
                ax.legend(fontsize=float(legend_size_var.get()))
                fig.tight_layout()
                canvas.draw_idle()
            except Exception as exc:
                messagebox.showerror("ΔPL/PL error", str(exc), parent=window)

        def export_delta_time_graph():
            if not last:
                messagebox.showwarning("No data", "Calculate the curves first.", parent=window)
                return
            try:
                title_size = float(title_size_var.get())
                axis_size = float(axis_size_var.get())
                tick_size = float(tick_size_var.get())
                legend_size = float(legend_size_var.get())
                dpi = int(float(dpi_var.get()))
                if min(title_size, axis_size, tick_size, legend_size, dpi) <= 0:
                    raise ValueError("All font sizes and the DPI must be positive.")
                path = filedialog.asksaveasfilename(
                    parent=window,
                    title="Export ΔPL/PL time graph",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                    initialfile="deltaPL_over_PL_vs_time.png",
                )
                if not path:
                    return
                export_fig, export_ax = plt.subplots(figsize=(10, 7))
                export_ax.plot(last["time"], 100.0 * last["contrast"], label="ΔPL/PL")
                export_ax.axhline(0.0, linewidth=0.8)
                export_ax.set_xscale(
                    "log" if self.log_time_var.get() and np.all(last["time"] > 0) else "linear"
                )
                export_ax.set_xlabel("Time (s)", fontsize=axis_size)
                export_ax.set_ylabel("ΔPL/PL (%)", fontsize=axis_size)
                graph_title = title_var.get().strip()
                if graph_title:
                    export_ax.set_title(graph_title, fontsize=title_size)
                export_ax.tick_params(axis="both", labelsize=tick_size)
                export_ax.legend(fontsize=legend_size)
                export_ax.grid(False)
                export_fig.tight_layout()
                export_fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(export_fig)
                messagebox.showinfo("PNG export complete", f"Saved:\n{path}", parent=window)
            except Exception as exc:
                messagebox.showerror("Export error", str(exc), parent=window)

        def export_time_csv():
            if not last:
                messagebox.showwarning("No data", "Calculate the curves first.", parent=window)
                return
            folder = filedialog.askdirectory(parent=window, title="Choose folder for transient CSV files")
            if not folder:
                return
            out = Path(folder)
            curves = {
                "deltaPL_over_PL": last["contrast"],
                "S1_control": last["s1_control"],
                "S1_driven": last["s1_driven"],
            }
            for name, values in curves.items():
                np.savetxt(
                    out / f"time_{name}.csv",
                    np.column_stack([last["time"], values]),
                    delimiter=",",
                    header=f"time_s,{name}",
                    comments="",
                )
            messagebox.showinfo("CSV export complete", f"Three separate CSV files saved in:\n{folder}", parent=window)

        def export_time_png():
            if not last:
                messagebox.showwarning("No data", "Calculate the curves first.", parent=window)
                return
            folder = filedialog.askdirectory(parent=window, title="Choose folder for transient PNG files")
            if not folder:
                return
            out = Path(folder)
            curves = [
                ("deltaPL_over_PL", 100.0 * last["contrast"], "ΔPL/PL (%)"),
                ("S1_control", last["s1_control"], "S1 population"),
                ("S1_driven", last["s1_driven"], "S1 population"),
            ]
            for name, values, ylabel in curves:
                export_fig, export_ax = plt.subplots(figsize=(9, 6))
                export_ax.plot(last["time"], values, label=name.replace("_", " "))
                export_ax.set_xscale("log" if self.log_time_var.get() and np.all(last["time"] > 0) else "linear")
                export_ax.set_xlabel("Time (s)", fontsize=float(axis_size_var.get()))
                export_ax.set_ylabel(ylabel, fontsize=float(axis_size_var.get()))
                graph_title = title_var.get().strip()
                if graph_title:
                    export_ax.set_title(f"{graph_title} — {name.replace('_', ' ')}", fontsize=float(title_size_var.get()))
                export_ax.grid(False)
                export_ax.legend()
                export_fig.tight_layout()
                export_fig.savefig(out / f"time_{name}.png", dpi=300, bbox_inches="tight")
                plt.close(export_fig)
            messagebox.showinfo("PNG export complete", f"Three separate PNG files saved in:\n{folder}", parent=window)

        ttk.Label(top, text="PNG dpi").grid(row=1, column=8, padx=4, pady=3)
        ttk.Entry(top, textvariable=dpi_var, width=7).grid(row=1, column=9, padx=4, pady=3)
        ttk.Button(top, text="Recalculate from main parameters", command=calculate).grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=5)
        ttk.Button(top, text="Export ΔPL/PL graph", command=export_delta_time_graph).grid(row=1, column=2, columnspan=2, sticky="ew", padx=4, pady=5)
        ttk.Button(top, text="Export PNG curves", command=export_time_png).grid(row=1, column=4, columnspan=2, sticky="ew", padx=4, pady=5)
        ttk.Button(top, text="Export CSV curves", command=export_time_csv).grid(row=1, column=6, columnspan=2, sticky="ew", padx=4, pady=5)
        calculate()

    def export_csv(self) -> None:
        if self.last_data is None:
            messagebox.showwarning("No data", "Run a simulation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export populations",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        t, y = self.last_data
        data = np.column_stack([t, y.T, y[2] + y[3] + y[4]])
        header = "time_s,S0,S1,Tx,Ty,Tz,T1_total"
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        messagebox.showinfo("Export complete", f"Saved:\n{path}")


if __name__ == "__main__":
    app = PopulationApp()
    app.mainloop()