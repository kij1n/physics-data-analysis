import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass


class Constants:
    GRAVITY = 9.81
    COLORS = ["blue", "green", "red", "orange"]


@dataclass
class PendulumsData:
    cols: list[str]
    labels: list[str]
    lengths: np.ndarray
    data: pd.DataFrame  

    def drop_cols(self, cols_to_drop: list[str]):
        self.data = self.data.drop(columns=cols_to_drop)
        for col in cols_to_drop:
            index = self.cols.index(col)
            self.cols.pop(index)
            self.labels.pop(index)
            self.lengths = np.delete(self.lengths, index)

class EnvelopeData:
    def __init__(self, cols: list[str], labels: list[str], data: pd.DataFrame):
        self.cols = cols
        self.labels = labels
        self.data = data
        self.results = {
            col: self._calc_envelope_max_mean(data["t"], data[col].to_numpy())
            for col in cols
        }

    def __repr__(self) -> str:
        lines = [
            f"{label}: peaks={len(self.results[col][1])}; mean={self.results[col][0]:.3f}"
            for col, label in zip(self.cols, self.labels)
        ]
        return "\n".join(lines)

    def _calc_envelope_max_mean(
        self,
        t: np.ndarray,
        signal: np.ndarray,
        env_distance_s: float = 25.0,
        min_height_frac: float = 0.3,
        prominence_frac: float = 0.15,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        t = np.asarray(t)
        signal = np.asarray(signal)
        abs_max = float(np.max(np.abs(signal)))
        height_threshold = min_height_frac * abs_max
        prominence_threshold = prominence_frac * abs_max

        def _envelope_indices(s: np.ndarray) -> np.ndarray:
            swing_idx, _ = find_peaks(s)
            if len(swing_idx) < 2:
                return np.array([], dtype=int)
            swing_dt = np.median(np.diff(t[swing_idx]))
            distance_samples = max(1, int(round(env_distance_s / swing_dt)))
            env_idx_in_swing, _ = find_peaks(
                s[swing_idx],
                distance=distance_samples,
                height=height_threshold,
                prominence=prominence_threshold,
            )
            return swing_idx[env_idx_in_swing]

        top_idx = _envelope_indices(signal)
        bot_idx = _envelope_indices(-signal)

        all_idx = np.concatenate([top_idx, bot_idx])
        if len(all_idx) == 0:
            return float("nan"), np.array([]), np.array([])

        order = np.argsort(t[all_idx])
        all_idx = all_idx[order]
        peak_t = t[all_idx]
        peak_h_signed = signal[all_idx]
        return float(np.abs(peak_h_signed).mean()), peak_t, peak_h_signed

    def get_amplitude_means(self) -> np.ndarray:
        return np.array([self.results[col][0] for col in self.cols])


class ResonanceFitData:
    def __init__(self, L_d: float, p_data: PendulumsData):
        self.L_d = L_d
        self.L = p_data.lengths

        self.envelope_data = EnvelopeData(p_data.cols, p_data.labels, p_data.data)
        self.A = self.envelope_data.get_amplitude_means()
        self.A_norm = self.A / self.A.max()

        self.omega_d = np.sqrt(Constants.GRAVITY / L_d)
        self.omega_0 = np.sqrt(Constants.GRAVITY / self.L)
        self.omega_0_max = self.omega_0[self.A.argmax()]

        self.gamma = None
        self.gamma_err = None

    def fit(self) -> None:
        def resonance_norm(omega_0_i, gamma):
            numerator = (self.omega_0_max**2 - self.omega_d**2) ** 2 + (
                gamma * self.omega_d
            ) ** 2
            denominator = (omega_0_i**2 - self.omega_d**2) ** 2 + (
                gamma * self.omega_d
            ) ** 2
            return np.sqrt(numerator / denominator)
        
        print(self.A)

        popt, pcov = curve_fit(
            resonance_norm, self.omega_0, self.A_norm, p0=[0.5], bounds=(0, np.inf)
        )
        gamma_fit = popt[0]
        gamma_err = np.sqrt(pcov[0, 0])

        self.gamma = gamma_fit
        self.gamma_err = gamma_err

    def plot_detected_peaks(self):
        col_names = ["A", "B", "C", "D"]
        fix, axd = plt.subplot_mosaic(
            [col_names[0:2], col_names[2:4]], sharex=True, sharey=True
        )

        for key, color, label, col in zip(
            col_names,
            Constants.COLORS,
            self.envelope_data.labels,
            self.envelope_data.cols,
        ):
            _, peak_t, peak_h_signed = self.envelope_data.results[col]
            axd[key].plot(
                self.envelope_data.data["t"],
                self.envelope_data.data[col],
                color=color,
                alpha=0.4,
            )
            axd[key].plot(peak_t, peak_h_signed, "rx", markersize=9, mew=2)
            axd[key].set_title(f"{label} ({len(peak_t)} env peaks)")

        axd["A"].set_ylabel("Amplitude")
        axd["C"].set_ylabel("Amplitude")
        axd["C"].set_xlabel("Time (s)")
        axd["D"].set_xlabel("Time (s)")

        plt.suptitle("Detected beat-envelope maxima")
        plt.tight_layout()

    def plot_resonance_curve(self, data_point_clr: str, fit_clr: str):
        omega_range = np.linspace(
            self.omega_0.min() - 0.05, self.omega_0.max() + 0.05, 400
        )
        fit_vals = resonance_norm(
            self.omega_d, omega_range, self.omega_0[self.A.argmax()], self.gamma
        )

        plt.plot(
            omega_range,
            fit_vals,
            "--",
            label=f"Fit (γ = {self.gamma:.3f} ± {self.gamma_err:.3f})",
            color=fit_clr,
        )
        plt.plot(self.omega_0, self.A_norm, "o", label="Data", color=data_point_clr)

        plt.axvline(
            self.omega_d,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Driver ω_d = {self.omega_d:.3f}",
        )


def resonance_norm(omega_d, omega_i, omega_i_max_A, gamma):
    numerator = (omega_i_max_A**2 - omega_d**2) ** 2 + (gamma * omega_d) ** 2
    denominator = (omega_i**2 - omega_d**2) ** 2 + (gamma * omega_d) ** 2
    return np.sqrt(numerator / denominator)


def plot_all(p_data: PendulumsData, cols: list[str] = None, color_override: list[str] = None):
    colors = color_override if color_override is not None else Constants.COLORS

    for col, color in zip(cols, colors):
        plt.plot(
            p_data.data["t"],
            p_data.data[col],
            color=color,
            label=p_data.labels[p_data.cols.index(col)],
            alpha=0.9,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs Time")
    plt.legend()


def plot_each(p_data: PendulumsData):
    fig, axd = plt.subplot_mosaic([["A", "B"], ["C", "D"]], sharex=True, sharey=True)

    opacity = 1

    axd["A"].plot(
        p_data.data["t"],
        p_data.data["mass A"],
        color="blue",
        label="97cm",
        alpha=opacity,
    )
    axd["A"].set_title("97cm")
    axd["B"].plot(
        p_data.data["t"],
        p_data.data["mass B"],
        color="orange",
        label="99cm",
        alpha=opacity,
    )
    axd["B"].set_title("99cm")
    axd["C"].plot(
        p_data.data["t"],
        p_data.data["mass C"],
        color="green",
        label="99.5cm",
        alpha=opacity,
    )
    axd["C"].set_title("99.5cm")
    axd["D"].plot(
        p_data.data["t"],
        p_data.data["mass D"],
        color="black",
        label="100cm",
        alpha=opacity,
    )
    axd["D"].set_title("100cm")

    axd["A"].set_ylabel("Amplitude")
    axd["C"].set_ylabel("Amplitude")
    axd["C"].set_xlabel("Time (s)")
    axd["D"].set_xlabel("Time (s)")

    plt.suptitle("Amplitude vs Time for Each Mass")
    plt.tight_layout()

    plt.suptitle("Amplitude vs Time for Each Mass")


def add_resonance_plot_labels():
    plt.xlabel("Natural Frequency ω₀ (rad/s)")
    plt.ylabel("Normalized Amplitude A/A_max")
    plt.title("Resonance Curve")
    plt.legend()
