import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from .pendulums_data import PendulumsData
from .constants import Constants


def plot_all(
    p_data: PendulumsData,
    cols: list[str] = None,
    color_override: list[str] = None,
    title: str = None,
    t_range: tuple[int | float, int | float] | None = None,
) -> None:
    """
    Plot the time series data for all pendulums on a single graph, with options to specify which columns to plot and override colors.
    Args:
        p_data: An instance of PendulumsData containing the time series data and metadata.
        cols: Optional list of column names to plot. If None, all columns in p_data will be plotted.
        color_override: Optional list of colors to use for the plots. If None, default colors from Constants.COLORS will be used.
        title: Optional title for the plot.
        t_range: Optional (t_start, t_end) tuple to restrict the plotted time range (inclusive).
            Use float('inf') or np.inf as t_end to plot from t_start to the end of the data.
    """
    colors = color_override if color_override is not None else Constants.COLORS
    cols = cols if cols is not None else p_data.cols

    if t_range is not None:
        mask = (p_data.data["t"] >= t_range[0]) & (p_data.data["t"] <= t_range[1])
        df = p_data.data[mask]
    else:
        df = p_data.data

    for col, color in zip(cols, colors):
        plt.plot(
            df["t"],
            df[col],
            color=color,
            label=p_data.labels[p_data.cols.index(col)],
            alpha=0.9,
        )

    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title if title is not None else "Amplitude vs Time")
    plt.legend()


def plot_each(
    p_data: PendulumsData,
    title: str = None,
    t_range: tuple[int | float, int | float] | None = None,
) -> None:
    """
    Plot the time series data for each pendulum in separate subplots, sharing the same x and y axes.
    Args:
        p_data: An instance of PendulumsData containing the time series data and metadata.
        title: Optional title for the plot. If None, a default title will be used.
        t_range: Optional (t_start, t_end) tuple to restrict the plotted time range (inclusive).
            Use float('inf') or np.inf as t_end to plot from t_start to the end of the data.
    """
    fig, axd = plt.subplot_mosaic([["A", "B"], ["C", "D"]], sharex=True, sharey=True)

    opacity = 1

    if t_range is not None:
        mask = (p_data.data["t"] >= t_range[0]) & (p_data.data["t"] <= t_range[1])
        df = p_data.data[mask]
    else:
        df = p_data.data

    for col, color, label in zip(p_data.cols, Constants.COLORS, p_data.labels):
        key = col[-1]
        axd[key].plot(
            df["t"],
            df[col],
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
    figures_dir = Path(__file__).resolve().parents[2] / "figures"
    target_path = figures_dir / path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_path, dpi=dpi, bbox_inches="tight")


def plot_transient_decay(
    gammas: float | list[float],
    plot_range: tuple[float, float],
    measurement_range: tuple[float, float],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    threshold_pct: float = 5.0,
    title: str | None = None,
    round_digits: int = 2,
) -> None:
    """
    Plot the decay of the natural transient amplitude for one or more damping
    coefficients, expressed as a percentage of its initial value.

    The underlying model is A = A_ss + A_nat * exp(-gamma * t / 2), but only the
    natural transient term A_nat is of interest here, so each curve is normalized
    to start at 100% at t = 0 and follows 100 * exp(-gamma * t / 2). The gamma
    values are per-second (as produced by the envelope/resonance fits), while the
    x-axis is in minutes, so time is converted internally.

    Args:
        gammas: A single damping coefficient or a list of them (per second) to plot.
        plot_range: (t_start, t_end) range of the x-axis, in minutes. Both bounds
            must be finite.
        measurement_range: (t_start, t_end) of the grayed measurement window, in minutes.
        labels: Optional labels for each gamma curve. If None, labels are derived
            from the gamma values.
        colors: Optional list of colors for the curves. If None, Constants.COLORS is used.
        threshold_pct: Percentage at which the transient is considered to be in
            steady state, drawn as a horizontal dashed line. Default is 5.0.
        title: Optional title for the plot. If None, a default title is used.
    """

    def decay(t_min: np.ndarray, gamma: float) -> np.ndarray:
        """Natural transient amplitude as a % of its initial value (t in minutes)."""
        t_seconds = t_min * 60.0
        return 100.0 * np.exp(-gamma * t_seconds / 2.0)

    if np.isscalar(gammas):
        gammas = [gammas]

    colors = colors if colors is not None else Constants.COLORS
    t = np.linspace(plot_range[0], plot_range[1], 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axvspan(
        measurement_range[0],
        measurement_range[1],
        color="gray",
        alpha=0.2,
        label=f"measurement window ({measurement_range[0]}–{measurement_range[1]} min)",
    )

    for i, gamma in enumerate(gammas):
        color = colors[i % len(colors)]
        label = (
            labels[i]
            if labels is not None
            else f"γ = {np.round(gamma, round_digits)} rad/s"
        )
        ax.plot(t, decay(t, gamma), color=color, label=label, linewidth=2)

    ax.axhline(
        threshold_pct,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"steady-state threshold (transient = {threshold_pct:.0f}%)",
    )

    ax.set_xlim(plot_range)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("time (min)")
    ax.set_ylabel("transient amplitude (% of initial value)")
    ax.set_title(title if title is not None else "Transient amplitude decay")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()


def bar_chart(
    data: tuple[list[str], list[float], list[float]], title: str = None
) -> None:
    """
    Create a bar chart with error bars from the provided data.
    Args:
        data: A list of tuples, where each tuple contains (label, value, error).
        title: Optional title for the plot. If None, a default title will be used.
    """
    labels, values, errors = data

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(
        labels,
        values,
        yerr=errors,
        capsize=6,
        color="#4C72B0",
        edgecolor="black",
        alpha=0.8,
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        error = errors[i]

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + error + 0.0005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Gamma Value")
    ax.set_title("Baseline Gamma Values by Mass" if title is None else title)

    ax.set_ylim(0, max(values) + max(errors) + 0.005)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()


def bar_charts(
    datasets: list[tuple[list[str], list[float], list[float]]],
    titles: list[str] | None = None,
) -> None:
    """
    Plot multiple bar charts with error bars in a mosaic layout.
    Args:
        datasets: List of (labels, values, errors) tuples, one per chart.
        titles: Optional list of titles, one per chart. If None, no titles are set.
    """
    n = len(datasets)
    cols = 2
    rows = (n + cols - 1) // cols
    keys = [chr(ord("A") + i) for i in range(n)]

    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            row.append(keys[idx] if idx < n else keys[n - 1])
        grid.append(row)

    _, axd = plt.subplot_mosaic(grid, figsize=(8 * cols, 6 * rows), sharey=True)

    for i, (data, key) in enumerate(zip(datasets, keys)):
        labels, values, errors = data
        ax = axd[key]

        bars = ax.bar(
            labels,
            values,
            yerr=errors,
            capsize=6,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.8,
        )

        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + errors[j] + 0.0005,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        if i % cols == 0:
            ax.set_ylabel("Gamma Value")
        else:
            ax.tick_params(labelleft=False)

        ax.set_ylim(0, max(values) + max(errors) + 0.005)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        if titles is not None:
            ax.set_title(titles[i])

    plt.tight_layout()


def plot_fft_results_all(
    fft_results: dict[
        str, tuple[np.ndarray[float], np.ndarray[complex], np.ndarray[float]]
    ],
    freq_range: tuple[float, float] = (0.43, 0.55),
    title: str | None = None,
    labels: list[str] | None = None,
) -> None:
    """
    Plot the FFT results for each pendulum in separate subplots.
    Args:
        fft_results: A dictionary mapping column names to tuples of (frequencies, FFT values, magnitudes).
    """

    for i, (xf, _, mag) in enumerate(fft_results.values()):
        plt.plot(
            xf,
            mag,
            color=Constants.COLORS[i % len(Constants.COLORS)],
            label=labels[i] if labels is not None else f"Pendulum {i + 1}",
            alpha=0.8,
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title if title is not None else "FFT Magnitude vs Frequency")
    plt.legend()
    plt.xlim(*freq_range)  # Limit x-axis to focus on relevant frequencies
    plt.grid()
    plt.tight_layout()


def plot_fft_results_each(
    fft_results: dict[
        str, tuple[np.ndarray[float], np.ndarray[complex], np.ndarray[float]]
    ],
    freq_range: tuple[float, float] = (0.43, 0.55),
    title: str | None = None,
    labels: list[str] | None = None,
) -> None:
    """
    Plot the FFT results for each pendulum in separate subplots.
    Args:
        fft_results: A dictionary mapping column names to tuples of (frequencies, FFT values, magnitudes).
    """
    fig, axd = plt.subplot_mosaic([["A", "B"], ["C", "D"]], sharex=True, sharey=True)

    for i, (col, (xf, _, mag)) in enumerate(fft_results.items()):
        key = col[-1]
        axd[key].plot(
            xf,
            mag,
            color=Constants.COLORS[i % len(Constants.COLORS)],
            label=labels[i] if labels is not None else f"Pendulum {key}",
            alpha=0.8,
        )
        axd[key].set_title(labels[i] if labels is not None else f"Pendulum {key}")
        axd[key].set_xlim(*freq_range)  # Limit x-axis to focus on relevant frequencies
        axd[key].grid()

    axd["C"].set_xlabel("Frequency (Hz)")
    axd["D"].set_xlabel("Frequency (Hz)")
    axd["A"].set_ylabel("Magnitude")
    axd["C"].set_ylabel("Magnitude")

    plt.suptitle(
        title if title is not None else "FFT Magnitude vs Frequency for Each Pendulum"
    )
    plt.tight_layout()
