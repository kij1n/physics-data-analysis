import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def get_gamma(L_d: float, A: np.ndarray, L: np.ndarray, g: float) -> tuple[float, float]:
    omega_d = np.sqrt(g / L_d)
    omega_0 = np.sqrt(g / L)

    A_norm = A / A.max()
    omega_0_max = omega_0[A.argmax()]

    def resonance_norm(omega_0_i, gamma):
        numerator = (omega_0_max**2 - omega_d**2)**2 + (gamma * omega_d)**2
        denominator = (omega_0_i**2 - omega_d**2)**2 + (gamma * omega_d)**2
        return np.sqrt(numerator / denominator)

    popt, pcov = curve_fit(resonance_norm, omega_0, A_norm, p0=[0.5])
    gamma_fit = popt[0]
    gamma_err = np.sqrt(pcov[0, 0])

    # DEBUG
    print(f"L_d: {L_d}, A: {A}, L: {L}, g: {g}")
    print(f"Omega_d: {omega_d}, Omega_0: {omega_0}, A_norm: {A_norm}")
    print(f"Fit result - Gamma: {gamma_fit}, Error: {gamma_err}")

    return gamma_fit, gamma_err