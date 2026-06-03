import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from .pendulums_data import PendulumsData
from .envelope_data import EnvelopeData
from .constants import Constants


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

        self.single_pendulum_gamma: dict[str, tuple[float, float]] = {}

    def get_single_gamma_avg(self) -> float:
        """
        Get the average gamma value across all individual pendulum fits.
        Returns:
            The average gamma value calculated from the individual pendulum fits.
        """
        if not self.single_pendulum_gamma:
            raise ValueError(
                "No single pendulum gamma values found. Please run fit_single_pendulums() first."
            )

        gammas = [gamma for gamma, _ in self.single_pendulum_gamma.values()]
        return float(np.mean(gammas))

    def get_data_to_plot(self) -> tuple[list[str], list[float], list[float]]:
        data_to_plot = ([], [], [])

        for col in self.envelope_data.cols:
            gamma, gamma_err = self.single_pendulum_gamma[col]
            data_to_plot[0].append(col)
            data_to_plot[1].append(gamma)
            data_to_plot[2].append(gamma_err)

        return data_to_plot

def fit_single_pendulums(self) -> None: 
        """
        Fit the decay of the envelope peaks for each individual pendulum to extract the damping coefficient.
        The values are stored in the instance variable `single_pendulum_gamma` as a dictionary mapping column names to tuples of (gamma, gamma_err).
        """
        for col in self.envelope_data.cols:
            t_peaks = self.envelope_data.results[col][1]
            h_peaks_signed = self.envelope_data.results[col][2]
            abs_peaks = np.abs(h_peaks_signed)

            def decay(t, A_ss, A_nat, gamma):
                return A_ss + A_nat * np.exp(-gamma * t / 2)

            p0 = [
                np.mean(abs_peaks[-4:]),
                np.mean(abs_peaks[:4]) - np.mean(abs_peaks[-4:]),
                0.05,
            ]

            popt, pcov = curve_fit(
                decay,
                t_peaks,
                abs_peaks,
                p0=p0,
                bounds=([0, 0, 0], [1, 1, 1]),
                maxfev=10000,
            )
            A_ss, A_nat, gamma = popt
            gamma_err = np.sqrt(pcov[2, 2])

            self.single_pendulum_gamma[col] = (gamma, gamma_err)

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
        fig, axd = plt.subplot_mosaic(
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
