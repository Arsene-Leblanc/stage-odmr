from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp


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

    return np.array(
        [
            [-k01,          k10,                  kx,                  ky,                  kz],
            [ k01, -(k10 + kisc),                 0.0,                 0.0,                 0.0],
            [ 0.0,     kisc * px, -(kx + wxy + wxz),                 wyx,                 wzx],
            [ 0.0,     kisc * py,                 wxy, -(ky + wyx + wyz),                 wzy],
            [ 0.0,     kisc * pz,                 wxz,                 wyz, -(kz + wzx + wzy)],
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
        self.method_var = tk.StringVar(value="Radau")
        ttk.Combobox(
            sim,
            textvariable=self.method_var,
            values=["Radau", "BDF", "LSODA", "RK45"],
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
        ttk.Button(button_frame, text="Export CSV", command=self.export_csv).pack(side="left", padx=3)

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

    def run_simulation(self) -> None:
        try:
            p, y0, t_start, t_end, n_points = self._collect_parameters()

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
            sol = solve_ivp(
                fun=lambda t, y: A @ y,
                t_span=(t_start, t_end),
                y0=y0,
                method=method,
                t_eval=t_eval,
                rtol=1e-9,
                atol=1e-12,
            )
            if not sol.success:
                raise RuntimeError(sol.message)

            y = sol.y
            # Remove negligible numerical drift only.
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

            self.last_data = (sol.t.copy(), y.copy())
            self.k01_label.config(text=f"{p['k01']:.4e} s⁻¹")

            self.ax.clear()
            labels = ["S0", "S1", "Tx", "Ty", "Tz"]
            for values, label in zip(y, labels):
                self.ax.plot(sol.t, values, label=label)

            T1 = y[2] + y[3] + y[4]
            self.ax.plot(sol.t, T1, "--", linewidth=2, label="T1 = Tx + Ty + Tz")

            if self.log_time_var.get() and t_start > 0:
                self.ax.set_xscale("log")

            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Population")
            self.ax.set_ylim(-0.02, 1.02)
            self.ax.legend(ncol=2)
            self.ax.grid(False)
            self.ax.set_title("Population dynamics from Eq. (2)")
            self.fig.tight_layout()
            self.canvas.draw_idle()

            summary_lines = [
                f"k01 = σIλ/(hc) = {p['k01']:.6e} s⁻¹",
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
            self.status_var.set(
                "Simulation updated. Px, Py, Pz and the initial populations "
                "were normalized automatically."
            )

        except Exception as exc:
            self.status_var.set(f"Simulation not updated: {exc}")
            messagebox.showerror("Simulation error", str(exc))

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