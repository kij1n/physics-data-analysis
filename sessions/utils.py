import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def envelope_max_mean(
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


def get_gamma(
    L_d: float, A: np.ndarray, L: np.ndarray, g: float
) -> tuple[float, float]:
    omega_d = np.sqrt(g / L_d)
    omega_0 = np.sqrt(g / L)

    A_norm = A / A.max()
    omega_0_max = omega_0[A.argmax()]

    def resonance_norm(omega_0_i, gamma):
        numerator = (omega_0_max**2 - omega_d**2) ** 2 + (gamma * omega_d) ** 2
        denominator = (omega_0_i**2 - omega_d**2) ** 2 + (gamma * omega_d) ** 2
        return np.sqrt(numerator / denominator)

    popt, pcov = curve_fit(
        resonance_norm, omega_0, A_norm, p0=[0.5], bounds=(0, np.inf)
    )
    gamma_fit = popt[0]
    gamma_err = np.sqrt(pcov[0, 0])

    # DEBUG
    print(f"L_d: {L_d}, A: {A}, L: {L}, g: {g}")
    print(f"Omega_d: {omega_d}, Omega_0: {omega_0}, A_norm: {A_norm}")
    print(f"Fit result - Gamma: {gamma_fit}, Error: {gamma_err}")

    return gamma_fit, gamma_err
