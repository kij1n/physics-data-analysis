import pandas as pd
import numpy as np
from scipy.signal import find_peaks, detrend
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
from pathlib import Path


class Constants:
    """
    Physical and plotting constants for the pendulum resonance analysis.
    """

    GRAVITY = 9.81
    COLORS = ["blue", "green", "red", "orange"]


@dataclass
class PendulumsData:
    """
    Data structure for holding pendulum time series data and associated metadata.
    """

    cols: list[str]
    labels: list[str]
    lengths: np.ndarray
    data: pd.DataFrame

    def drop_cols(self, cols_to_drop: list[str]):
        """
        Drop specified columns from the data and update metadata accordingly.
        Args:
            cols_to_drop: List of column names to be dropped from the data.
        """
        self.data = self.data.drop(columns=cols_to_drop)
        for col in cols_to_drop:
            index = self.cols.index(col)
            self.cols.pop(index)
            self.labels.pop(index)
            self.lengths = np.delete(self.lengths, index)

    def drop_time_range(self, t_start: float, t_end: float):
        """
        Drop rows from the data where the time 't' is within the specified range.
        Args:
            t_start: Start of the time range to be dropped (inclusive).
            t_end: End of the time range to be dropped (inclusive).
        """
        mask = (self.data["t"] < t_start) | (self.data["t"] > t_end)
        self.data = self.data[mask].reset_index(drop=True)

    def preprocess_offsets(self):
        """
        Preprocess the data by centering each column around its median value.
        This helps to remove any DC offset and focus on the oscillatory behavior.
        """
        for col in self.cols:
            self.data[col] -= self.data[col].median()

    def detrend(self):
        for col in self.cols:
            self.data[col] = detrend(self.data[col], type="linear")


class EnvelopeData:
    """
    Data structure for storing envelope peak information for each pendulum time series.
    Attributes:
        cols: List of column names corresponding to the pendulum data.
        labels: List of labels for each pendulum, used for plotting and identification.
        data: The original DataFrame containing the time series data.
        results: A dictionary mapping each column name to a tuple containing:
            - The mean amplitude of the detected envelope peaks.
            - The time values of the detected peaks.
            - The signed amplitude values of the detected peaks.
    """

    def __init__(self, cols: list[str], labels: list[str], data: pd.DataFrame):
        self.cols: list[str] = cols
        self.labels: list[str] = labels
        self.data: pd.DataFrame = data
        self.results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {
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
        """
        Calculate the mean amplitude of the envelope peaks for a given signal.
        Args:
            t: Time array corresponding to the signal.
            signal: The signal array for which to calculate the envelope peaks.
            env_distance_s: Minimum time distance between detected peaks in seconds.
            min_height_frac: Minimum height of peaks as a fraction of the maximum absolute signal value.
            prominence_frac: Minimum prominence of peaks as a fraction of the maximum absolute signal value.
        Returns:
            A tuple containing:
            - The mean amplitude of the detected envelope peaks.
            - An array of time values corresponding to the detected peaks.
            - An array of signed amplitude values of the detected peaks.
        """
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
        """
        Get an array of mean amplitudes for each column in the data.
        Returns:
            An array of mean amplitudes corresponding to each column in the data.
        """
        return np.array([self.results[col][0] for col in self.cols])


class ResonanceFitData:
    """
    Data structure for performing resonance curve fitting and analysis.
    Attributes:
        L_d: Length of the driver pendulum.
        L: Array of lengths for the other pendulums.
        envelope_data: An instance of EnvelopeData containing the detected envelope peaks and their means.
        A: Array of mean amplitudes for each pendulum.
        A_norm: Array of mean amplitudes normalized by the maximum amplitude.
        omega_d: Natural frequency of the driver pendulum.
        omega_0: Array of natural frequencies for the other pendulums.
        omega_0_max: Natural frequency of the pendulum with the maximum amplitude.
        gamma: Fitted damping coefficient from the resonance curve fitting.
        gamma_err: Estimated error of the fitted damping coefficient.
    """

    def __init__(self, L_d: float, p_data: PendulumsData):
        self.L_d: float = L_d
        self.L: np.ndarray = p_data.lengths

        self.envelope_data: EnvelopeData = EnvelopeData(
            p_data.cols, p_data.labels, p_data.data
        )
        self.A: np.ndarray = self.envelope_data.get_amplitude_means()
        self.A_norm: np.ndarray = self.A / self.A.max()

        self.omega_d: np.ndarray = np.sqrt(Constants.GRAVITY / L_d)
        self.omega_0: np.ndarray = np.sqrt(Constants.GRAVITY / self.L)
        self.omega_0_max: float = self.omega_0[self.A.argmax()]

        self.gamma: float | None = None
        self.gamma_err: float | None = None

    def fit(self) -> None:
        """
        Fit the resonance curve to the normalized amplitude data to extract the damping coefficient.
        The values are stored in the instance variables `gamma` and `gamma_err`.
        """

        def resonance_norm(omega_0_i, gamma):
            numerator = (self.omega_0_max**2 - self.omega_d**2) ** 2 + (
                gamma * self.omega_d
            ) ** 2
            denominator = (omega_0_i**2 - self.omega_d**2) ** 2 + (
                gamma * self.omega_d
            ) ** 2
            return np.sqrt(numerator / denominator)

        popt, pcov = curve_fit(
            resonance_norm, self.omega_0, self.A_norm, p0=[0.5], bounds=(0, np.inf)
        )
        gamma_fit = popt[0]
        gamma_err = np.sqrt(pcov[0, 0])

        self.gamma = gamma_fit
        self.gamma_err = gamma_err

    def plot_detected_peaks(self, title: str = None):
        """
        Plot the original time series data for each pendulum along with the detected envelope peaks.
        Args:
            title: Optional title for the plot. If not provided, a default title will be used.
        """
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

        plt.suptitle(title if title is not None else "Detected beat-envelope maxima")
        plt.tight_layout()

    def plot_resonance_curve(self, data_point_clr: str, fit_clr: str):
        """
        Plot the resonance curve of normalized amplitude vs natural frequency, along with the fitted curve.
        Args:
            data_point_clr: Color for the data points in the plot.
            fit_clr: Color for the fitted resonance curve in the plot.
        """
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


def resonance_norm(omega_d, omega_i, omega_i_max_A, gamma: float) -> np.ndarray:
    """
    Calculate the normalized amplitude of a driven damped oscillator at a given natural frequency.
    Args:
        omega_d: Natural frequency of the driver pendulum.
        omega_i: Natural frequency of the pendulum being evaluated.
        omega_i_max_A: Natural frequency of the pendulum with the maximum amplitude.
        gamma: Damping coefficient of the system.
    Returns:
        The normalized amplitude A/A_max for the given natural frequency omega_i.
    """
    numerator = (omega_i_max_A**2 - omega_d**2) ** 2 + (gamma * omega_d) ** 2
    denominator = (omega_i**2 - omega_d**2) ** 2 + (gamma * omega_d) ** 2
    return np.sqrt(numerator / denominator)


def plot_all(
    p_data: PendulumsData,
    cols: list[str] = None,
    color_override: list[str] = None,
    title: str = None,
) -> None:
    """
    Plot the time series data for all pendulums on a single graph, with options to specify which columns to plot and override colors.
    Args:
        p_data: An instance of PendulumsData containing the time series data and metadata.
        cols: Optional list of column names to plot. If None, all columns in p_data will be plotted.
        color_override: Optional list of colors to use for the plots. If None, default colors from Constants.COLORS will be used.
        title: Optional title for the plot.
    """
    colors = color_override if color_override is not None else Constants.COLORS
    cols = cols if cols is not None else p_data.cols

    for col, color in zip(cols, colors):
        plt.plot(
            p_data.data["t"],
            p_data.data[col],
            color=color,
            label=p_data.labels[p_data.cols.index(col)],
            alpha=0.9,
        )

    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title if title is not None else "Amplitude vs Time")
    plt.legend()


def plot_each(p_data: PendulumsData, title: str = None) -> None:
    """
    Plot the time series data for each pendulum in separate subplots, sharing the same x and y axes.
    Args:
        p_data: An instance of PendulumsData containing the time series data and metadata.
        title: Optional title for the plot. If None, a default title will be used.
    """
    fig, axd = plt.subplot_mosaic([["A", "B"], ["C", "D"]], sharex=True, sharey=True)

    opacity = 1

    for col, color, label in zip(p_data.cols, Constants.COLORS, p_data.labels):
        key = col[-1]
        axd[key].plot(
            p_data.data["t"],
            p_data.data[col],
            color=color,
            label=label,
            alpha=opacity,
        )
        axd[key].set_title(label)
        axd[key].axhline(y=0, color="black", linestyle="--", linewidth=1)

    axd["A"].set_ylabel("Amplitude")
    axd["C"].set_ylabel("Amplitude")
    axd["C"].set_xlabel("Time (s)")
    axd["D"].set_xlabel("Time (s)")

    plt.suptitle(title if title is not None else "Amplitude vs Time for Each Mass")
    plt.tight_layout()


def add_resonance_plot_labels(title: str = None) -> None:
    """
    Add labels and title to the resonance curve plot.
    Args:
        title: Optional title for the plot. If None, a default title will be used.
    """
    plt.xlabel("Natural Frequency ω₀ (rad/s)")
    plt.ylabel("Normalized Amplitude A/A_max")
    plt.title(title if title is not None else "Resonance Curve")
    plt.legend()

def save_plot(filename: str, dpi: int = 300) -> None:
    """
    Save the current plot to a file with the specified filename and resolution.
    Args:
        filename: The name of the file to save the plot to (including extension, e.g., 'plot.png').
        dpi: The resolution in dots per inch for the saved plot. Default is 300.
    """
    path = Path(filename)
    figure_dir = Path("../figures")
    dir = path.parent 

    if not os.path.exists(figure_dir / dir):
        os.makedirs(figure_dir / dir)
    
    filename = os.path.join(figure_dir, filename)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")