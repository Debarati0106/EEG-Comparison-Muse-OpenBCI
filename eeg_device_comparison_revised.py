"""
Muse vs OpenBCI EEG comparison pipeline
--------------------------------------
Input files:
  OpenBCI: FocusGym_Focus-Relax_20261016_130629.csv
  Muse:    recorded_Session_102759_20260910_102804.csv

Outputs:
  comparison_output/device_summary.csv
  comparison_output/channel_metrics.csv
  comparison_output/band_power.csv
  comparison_output/common_channel_comparison.csv
  comparison_output/plots/*.png

Install:
  pip install pandas numpy scipy matplotlib

Important:
- This script compares signal-quality characteristics, not clinical accuracy.
- These files appear to be from different dates, so sample-by-sample correlation
  is NOT meaningful unless you have matching event markers and aligned segments.
- Amplitude is assumed to already be in microvolts (µV); verify this against the export settings before publication. No numerical unit conversion is applied.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# --------------------------- CONFIGURATION ---------------------------

OPENBCI_FILE = Path("FocusGym_Focus-Relax_20261016_130629.csv")
MUSE_FILE = Path("recorded_Session_102759_20260910_102804.csv")

TIME_COL = "timestamp_unix"
OUTPUT_DIR = Path("comparison_output")
PLOT_DIR = OUTPUT_DIR / "plots"

# Both exports use 250 Hz according to timestamp intervals. The script estimates
# the rate from timestamps, then uses the estimated rate for spectral analysis.
# Do not force a nominal rate unless timestamps are unavailable.

# Common channels present in both exports; these are the only direct channel-name
# comparisons supported by the files. Verify electrode locations/montage before
# interpreting O1/O2 as equivalent across devices.
COMMON_CHANNELS = ["O1", "O2"]

# Standard frequency bands (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma_low": (30.0, 40.0),
}

# Filter used for conventional EEG quality metrics. A separate raw PSD is used
# for 50 Hz line-noise estimation so the filter does not remove the line noise.
EEG_BAND = (0.5, 40.0)
LINE_BAND = (49.0, 51.0)  # narrow 50 Hz estimate; inspect PSD plot too

# --------------------------- LOADING / TIME ---------------------------

def load_recording(path: Path, device: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.resolve()}\n"
            "Put this script in the same folder as both CSV files, or edit "
            "OPENBCI_FILE and MUSE_FILE at the top."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if TIME_COL not in df.columns:
        raise ValueError(
            f"{path.name}: expected timestamp column {TIME_COL!r}; "
            f"found {df.columns.tolist()}"
        )

    # Convert timestamp to numeric Unix seconds. The supplied files use Unix
    # timestamps; conversion to float also makes the intended units explicit.
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    valid_t = np.isfinite(t)
    df = df.loc[valid_t].reset_index(drop=True)
    t = t[valid_t]

    # Numeric signal columns, excluding timestamp and non-EEG metadata.
    metadata_names = {TIME_COL.lower(), "segment_label", "event", "marker",
                      "trigger", "timestamp", "time"}
    channels = []
    for col in df.columns:
        if col.lower() in metadata_names:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.95:
            df[col] = numeric
            channels.append(col)

    if len(t) < 3:
        raise ValueError(f"{path.name}: not enough valid timestamps.")

    # Keep original timestamp vector; sample interval diagnostics will detect
    # gaps and irregularity. Don't silently resample before reporting them.
    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(positive_dt):
        raise ValueError(f"{path.name}: timestamps do not increase.")

    fs = 1.0 / np.median(positive_dt)
    duration_s = float(t[-1] - t[0])
    nominal_dt = 1.0 / fs
    # A gap is an interval > 1.5 times median interval.
    gap_count = int(np.sum(dt > 1.5 * nominal_dt))
    duplicate_or_reverse_count = int(np.sum(dt <= 0))
    interval_jitter_ms = float(np.std(positive_dt) * 1000)

    info = {
        "device": device,
        "file": path.name,
        "rows": len(df),
        "channels_detected": ", ".join(channels),
        "duration_seconds": duration_s,
        "duration_minutes": duration_s / 60,
        "estimated_fs_hz": fs,
        "median_interval_ms": float(np.median(positive_dt) * 1000),
        "interval_jitter_ms": interval_jitter_ms,
        "gaps_over_1.5x_median": gap_count,
        "duplicate_or_reverse_intervals": duplicate_or_reverse_count,
        "timestamp_start_unix": float(t[0]),
        "timestamp_end_unix": float(t[-1]),
    }
    return df, t, channels, info


def get_clean_signal(df, channel):
    """Return numeric 1-D signal with NaN/Inf removed, preserving order."""
    x = pd.to_numeric(df[channel], errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def bandpass_if_possible(x, fs, low=1.0, high=40.0):
    """Zero-phase Butterworth filter; returns None if rate/data are unsuitable."""
    x = np.asarray(x, dtype=float)
    nyquist = fs / 2
    high = min(high, nyquist * 0.95)
    if len(x) < max(30, int(fs * 2)) or low >= high:
        return None
    sos = signal.butter(4, [low, high], btype="bandpass",
                        fs=fs, output="sos")
    # filtfilt avoids phase delay, but needs a sufficiently long series.
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        return None


def robust_outlier_pct(x, threshold_mad=5.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0.0
    robust_sigma = 1.4826 * mad
    return float(100 * np.mean(np.abs(x - med) > threshold_mad * robust_sigma))


def flatline_step_pct(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    finite_pairs = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    if not finite_pairs.any():
        return np.nan
    return float(100 * np.mean(np.diff(x)[finite_pairs] == 0))


def spectral_metrics(x, fs):
    """
    Welch PSD on raw, demeaned signal.
    Returns absolute band powers in assumed µV² and relative powers in %.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    result = {}
    if len(x) < max(32, int(fs * 2)):
        return result

    x = signal.detrend(x, type="constant")
    nperseg = min(len(x), max(32, int(round(fs * 4))))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg,
                          noverlap=nperseg // 2, detrend="constant")

    def integrate_band(lo, hi):
        hi = min(hi, fs / 2)
        mask = (f >= lo) & (f <= hi)
        if mask.sum() < 2:
            return np.nan
        return float(np.trapezoid(pxx[mask], f[mask]))

    total_1_40 = integrate_band(1.0, 40.0)
    result["power_1_40"] = total_1_40

    for name, (lo, hi) in BANDS.items():
        bp = integrate_band(lo, hi)
        result[f"{name}_power"] = bp
        result[f"{name}_relative_pct"] = (
            100 * bp / total_1_40
            if np.isfinite(bp) and np.isfinite(total_1_40) and total_1_40 > 0
            else np.nan
        )

    line_power = integrate_band(*LINE_BAND)
    result["power_50hz_49_51"] = line_power
    result["line_noise_ratio_50hz_to_1_40_pct"] = (
        100 * line_power / total_1_40
        if np.isfinite(line_power) and np.isfinite(total_1_40) and total_1_40 > 0
        else np.nan
    )

    # PSD peak location is descriptive, not a standalone quality score.
    search = (f >= 1.0) & (f <= min(40.0, fs / 2))
    result["dominant_frequency_1_40_hz"] = (
        float(f[search][np.argmax(pxx[search])]) if search.any() else np.nan
    )
    return result


def compute_channel_metrics(df, channels, device, fs):
    rows = []
    for ch in channels:
        # For metric calculation, keep finite samples. Missing values are
        # reported separately; don't interpolate them silently.
        original = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(original)
        x = original[finite]
        if len(x) < 3:
            continue

        filt = bandpass_if_possible(x, fs, *EEG_BAND)
        signal_for_artifacts = filt if filt is not None else x
        sd = float(np.std(x, ddof=1))
        row = {
            "device": device,
            "channel": ch,
            "valid_samples": int(len(x)),
            "missing_or_nonfinite_pct": float(100 * (1 - finite.mean())),
            "mean_uV_assuming_csv_is_uV": float(np.mean(x)),
            "std_raw_uV_assuming_csv_is_uV": sd,
            "rms_raw_uV_assuming_csv_is_uV": float(np.sqrt(np.mean(x ** 2))),
            "peak_to_peak_raw_uV_assuming_csv_is_uV": float(np.ptp(x)),
            "flatline_steps_pct": flatline_step_pct(original),
            "robust_outliers_raw_pct": robust_outlier_pct(x),
            "std_0p5_40hz_uV_assuming_csv_is_uV": (
                float(np.std(filt, ddof=1)) if filt is not None else np.nan
            ),
            "robust_outliers_1_40hz_pct": robust_outlier_pct(signal_for_artifacts),
            "all_zero_channel": bool(np.all(x == 0)),
        }
        row.update(spectral_metrics(x, fs))
        rows.append(row)
    return rows


def plot_recording(df, t, channels, device, fs, plot_dir):
    # Plot first 30 seconds for readability; full-duration data are used in metrics.
    n = min(len(df), int(round(30 * fs)))
    if n < 2:
        return
    rel_t = t[:n] - t[0]
    fig, axes = plt.subplots(len(channels), 1,
                             figsize=(13, max(3, 2.3 * len(channels))),
                             squeeze=False)
    for ax, ch in zip(axes[:, 0], channels):
        x = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)[:n]
        ax.plot(rel_t, x, linewidth=0.55)
        ax.set_ylabel("Amplitude (µV; assuming CSV values are µV)")
        ax.set_title(f"{device} — {ch}")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Seconds from start")
    fig.suptitle(f"{device}: first 30 seconds (raw)", y=1.002)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{device.lower()}_raw_first30s.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def plot_psd(df, channels, device, fs, plot_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plotted = 0
    for ch in channels:
        x = get_clean_signal(df, ch)
        if len(x) < max(32, int(fs * 2)):
            continue
        x = signal.detrend(x, type="constant")
        nperseg = min(len(x), max(32, int(round(fs * 4))))
        f, pxx = signal.welch(x, fs=fs, nperseg=nperseg,
                              noverlap=nperseg // 2)
        max_f = min(60, fs / 2)
        mask = (f >= 0.5) & (f <= max_f)
        ax.semilogy(f[mask], pxx[mask], label=ch, linewidth=1)
        plotted += 1
    ax.axvline(50, linestyle="--", linewidth=1, label="50 Hz mains")
    ax.set_title(f"{device}: raw power spectral density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz; assuming CSV values are µV)")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{device.lower()}_raw_psd.png", dpi=160)
    plt.close(fig)


def compare_common_channels(metrics_df, common_channels):
    """
    Compare only matching channel labels. Because recordings are from different
    sessions, this compares summary metrics, not sample-by-sample waveforms.
    """
    rows = []
    for ch in common_channels:
        subset = metrics_df[metrics_df["channel"] == ch]
        by_device = {r["device"]: r for _, r in subset.iterrows()}
        if "OpenBCI" not in by_device or "Muse" not in by_device:
            rows.append({
                "channel": ch,
                "status": "Not present in both files; not compared"
            })
            continue

        a = by_device["OpenBCI"]
        b = by_device["Muse"]
        row = {"channel": ch, "status": "summary metrics only; separate sessions"}
        metric_cols = [
            "std_raw_uV_assuming_csv_is_uV", "rms_raw_uV_assuming_csv_is_uV",
            "peak_to_peak_raw_uV_assuming_csv_is_uV", "std_0p5_40hz_uV_assuming_csv_is_uV",
            "robust_outliers_1_40hz_pct",
            "line_noise_ratio_50hz_to_1_40_pct",
            "alpha_relative_pct", "theta_relative_pct", "beta_relative_pct",
            "flatline_steps_pct", "all_zero_channel"
        ]
        for col in metric_cols:
            row[f"OpenBCI_{col}"] = a.get(col, np.nan)
            row[f"Muse_{col}"] = b.get(col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    openbci_df, openbci_t, openbci_channels, openbci_info = load_recording(
        OPENBCI_FILE, "OpenBCI"
    )
    muse_df, muse_t, muse_channels, muse_info = load_recording(
        MUSE_FILE, "Muse"
    )

    print("\nDetected channels")
    print("OpenBCI:", openbci_channels)
    print("Muse:   ", muse_channels)

    summary = pd.DataFrame([openbci_info, muse_info])
    summary.to_csv(OUTPUT_DIR / "device_summary.csv", index=False)

    all_rows = []
    all_rows.extend(compute_channel_metrics(
        openbci_df, openbci_channels, "OpenBCI", openbci_info["estimated_fs_hz"]
    ))
    all_rows.extend(compute_channel_metrics(
        muse_df, muse_channels, "Muse", muse_info["estimated_fs_hz"]
    ))
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUTPUT_DIR / "channel_metrics.csv", index=False)

    # Band power table, easy to use in a report or spreadsheet.
    power_cols = [
        "device", "channel", "power_1_40",
        "delta_power", "delta_relative_pct",
        "theta_power", "theta_relative_pct",
        "alpha_power", "alpha_relative_pct",
        "beta_power", "beta_relative_pct",
        "gamma_low_power", "gamma_low_relative_pct",
        "power_50hz_49_51", "line_noise_ratio_50hz_to_1_40_pct",
        "dominant_frequency_1_40_hz"
    ]
    existing_power_cols = [c for c in power_cols if c in metrics_df.columns]
    metrics_df[existing_power_cols].to_csv(
        OUTPUT_DIR / "band_power.csv", index=False
    )

    common = compare_common_channels(metrics_df, COMMON_CHANNELS)
    common.to_csv(OUTPUT_DIR / "common_channel_comparison.csv", index=False)

    plot_recording(openbci_df, openbci_t, openbci_channels, "OpenBCI",
                   openbci_info["estimated_fs_hz"], PLOT_DIR)
    plot_recording(muse_df, muse_t, muse_channels, "Muse",
                   muse_info["estimated_fs_hz"], PLOT_DIR)
    plot_psd(openbci_df, openbci_channels, "OpenBCI",
             openbci_info["estimated_fs_hz"], PLOT_DIR)
    plot_psd(muse_df, muse_channels, "Muse",
             muse_info["estimated_fs_hz"], PLOT_DIR)

    print("\n================ DEVICE SUMMARY ================")
    print(summary[[
        "device", "rows", "duration_minutes", "estimated_fs_hz",
        "median_interval_ms", "interval_jitter_ms",
        "gaps_over_1.5x_median", "duplicate_or_reverse_intervals"
    ]].round(4).to_string(index=False))

    print("\n================ CHANNEL METRICS ===============")
    display_cols = [
        "device", "channel", "all_zero_channel", "std_raw_uV_assuming_csv_is_uV",
        "std_0p5_40hz_uV_assuming_csv_is_uV", "robust_outliers_1_40hz_pct",
        "line_noise_ratio_50hz_to_1_40_pct",
        "alpha_relative_pct", "theta_relative_pct", "beta_relative_pct"
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols].round(4).to_string(index=False))

    print("\n================ COMMON CHANNELS ===============")
    print(common.round(4).to_string(index=False))

    print(f"\nDone. Results saved in: {OUTPUT_DIR.resolve()}")
    print("Read device_summary.csv first, then channel_metrics.csv and band_power.csv.")
    print("Plots are in comparison_output/plots/")

    print("\nINTERPRETATION NOTES:")
    print("- A higher amplitude/standard deviation does not necessarily mean more noise.")
    print("- A lower outlier rate is not a complete accuracy score.")
    print("- 50 Hz line-noise ratio is estimated from the CSV signal PSD before this script applies its EEG band-pass. If the acquisition/export already applied a 50 Hz notch, this is residual post-notch noise, not original pre-notch mains noise.")
    print("- Flat/all-zero channels should be investigated as possible acquisition/export issues.")
    print("- Amplitude and PSD plots are labelled in µV/µV² per Hz only under the assumption that the CSV values are already µV; verify export units before publication. No scaling conversion is applied.")
    print("- Separate sessions cannot support direct sample-by-sample device agreement.")
    print("- Check electrode montage, reference, impedance/contact, filtering, and event timing.")


if __name__ == "__main__":
    main()
