import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


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
        prominence_frac: float = 0.05,
        detrend_window_s: float = 75.0,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate the mean amplitude of the envelope peaks for a given signal.
        Args:
            t: Time array corresponding to the signal.
            signal: The signal array for which to calculate the envelope peaks.
            env_distance_s: Minimum time distance between detected peaks in seconds.
            min_height_frac: Minimum height of peaks as a fraction of the maximum absolute signal value.
            prominence_frac: Minimum prominence of peaks (on the detrended envelope) as a fraction of the maximum absolute signal value.
            detrend_window_s: Width, in seconds, of the moving-average baseline removed from the swing-amplitude envelope before peak detection. Should span roughly one beat period or more.
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
            swing_env = s[swing_idx]
            swing_dt = np.median(np.diff(t[swing_idx]))
            distance_samples = max(1, int(round(env_distance_s / swing_dt)))

            # The swing-amplitude envelope can ride on a slow trend (e.g. the
            # exponential decay of a freely-swinging pendulum). scipy measures
            # prominence against that trend, so on a decaying envelope every
            # beat except the global maximum collapses to a near-zero
            # prominence and gets discarded. Subtracting a slow moving-average
            # baseline leaves the beat modulation on a flat zero line, so
            # prominence reflects the beat depth itself.
            window = max(3, int(round(detrend_window_s / swing_dt)))
            baseline = uniform_filter1d(swing_env, size=window, mode="nearest")
            detrended = swing_env - baseline

            env_idx_in_swing, _ = find_peaks(
                detrended,
                distance=distance_samples,
                prominence=prominence_threshold,
            )
            # Keep the absolute-amplitude gate on the original envelope so that
            # low-amplitude beats are still rejected after detrending.
            env_idx_in_swing = env_idx_in_swing[
                swing_env[env_idx_in_swing] >= height_threshold
            ]
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
