import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
