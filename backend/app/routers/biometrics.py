"""BioMetrics module — HRV analysis, EDA/GSR processing, accelerometer
analysis, and multi-signal session summaries."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from typing import Optional

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class HRVRequest(BaseModel):
    rr_intervals: list[float]


class EDARequest(BaseModel):
    eda_signal: list[float]
    sampling_rate: float = 4.0


class AccelerometerRequest(BaseModel):
    x: list[float]
    y: list[float]
    z: list[float]
    sampling_rate: float = 50.0


class SessionSummaryRequest(BaseModel):
    heart_rate: Optional[list[float]] = None
    eda: Optional[list[float]] = None
    temperature: Optional[list[float]] = None
    duration_seconds: float


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/hrv")
def hrv_analysis(req: HRVRequest):
    """Heart Rate Variability analysis from R-R intervals (in milliseconds)."""
    rr = np.array(req.rr_intervals, dtype=float)
    if len(rr) < 3:
        raise HTTPException(400, "At least 3 R-R intervals required for HRV analysis.")
    if np.any(rr <= 0):
        raise HTTPException(400, "All R-R intervals must be positive.")
    if np.any(rr < 200) or np.any(rr > 2000):
        raise HTTPException(400, "R-R intervals should be between 200 and 2000 ms.")

    n = len(rr)

    # ── Time domain ──
    mean_rr = float(np.mean(rr))
    mean_hr = 60000.0 / mean_rr  # bpm
    sdnn = float(np.std(rr, ddof=1))

    # Successive differences
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    nn50 = int(np.sum(np.abs(diffs) > 50))
    pnn50 = float(nn50 / len(diffs) * 100) if len(diffs) > 0 else 0.0

    time_domain = {
        "mean_rr_ms": round(mean_rr, 2),
        "mean_hr_bpm": round(mean_hr, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "nn50": nn50,
        "pnn50": round(pnn50, 2),
    }

    # ── Frequency domain (Lomb-Scargle on unevenly sampled RR) ──
    # Build time axis from cumulative RR intervals (in seconds)
    t = np.cumsum(rr) / 1000.0
    t = t - t[0]

    # Frequency bands
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # Generate frequency grid
    freq = np.linspace(0.003, 0.5, 512)
    angular_freq = 2 * np.pi * freq

    # Lomb-Scargle periodogram
    pgram = sp_signal.lombscargle(t, rr - np.mean(rr), angular_freq, normalize=False)

    # Compute power in each band via trapezoidal integration
    df = freq[1] - freq[0]
    vlf_mask = (freq >= vlf_band[0]) & (freq < vlf_band[1])
    lf_mask = (freq >= lf_band[0]) & (freq < lf_band[1])
    hf_mask = (freq >= hf_band[0]) & (freq <= hf_band[1])

    vlf_power = float(np.trapz(pgram[vlf_mask], freq[vlf_mask])) if vlf_mask.any() else 0.0
    lf_power = float(np.trapz(pgram[lf_mask], freq[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapz(pgram[hf_mask], freq[hf_mask])) if hf_mask.any() else 0.0

    total_lf_hf = lf_power + hf_power
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
    lf_nu = (lf_power / total_lf_hf * 100) if total_lf_hf > 0 else 0.0
    hf_nu = (hf_power / total_lf_hf * 100) if total_lf_hf > 0 else 0.0

    frequency_domain = {
        "vlf_power": round(vlf_power, 4),
        "lf_power": round(lf_power, 4),
        "hf_power": round(hf_power, 4),
        "lf_hf_ratio": round(lf_hf_ratio, 4),
        "lf_nu": round(lf_nu, 2),
        "hf_nu": round(hf_nu, 2),
    }

    # ── Stress Index (Baevsky) ──
    # SI = AMo / (2 * Mo * MxDMn)
    # where Mo = mode of RR, AMo = % of mode, MxDMn = max - min
    bin_width = 50  # ms
    bins = np.arange(rr.min() - bin_width / 2, rr.max() + bin_width, bin_width)
    hist, bin_edges = np.histogram(rr, bins=bins)
    mode_idx = np.argmax(hist)
    mo = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    amo = hist[mode_idx] / n * 100  # percentage
    mxdmn = float(rr.max() - rr.min())
    stress_index = (amo / (2 * mo / 1000 * mxdmn / 1000)) if (mo > 0 and mxdmn > 0) else 0.0

    return {
        "time_domain": time_domain,
        "frequency_domain": frequency_domain,
        "stress_index": round(float(stress_index), 2),
    }


@router.post("/eda-analysis")
def eda_analysis(req: EDARequest):
    """Electrodermal Activity (EDA/GSR) signal analysis."""
    eda = np.array(req.eda_signal, dtype=float)
    fs = req.sampling_rate

    if len(eda) < 4:
        raise HTTPException(400, "At least 4 EDA samples required.")
    if fs <= 0:
        raise HTTPException(400, "Sampling rate must be positive.")

    n_samples = len(eda)
    duration = n_samples / fs

    # Basic stats
    mean_scl = float(np.mean(eda))
    min_scl = float(np.min(eda))
    max_scl = float(np.max(eda))

    # ── Tonic/Phasic decomposition (simple low-pass filter approach) ──
    # Tonic: low-pass filtered signal (cutoff ~0.05 Hz)
    nyq = fs / 2.0
    cutoff = min(0.05, nyq * 0.8)  # ensure cutoff is below Nyquist
    if nyq > 0.05 and len(eda) > 12:
        b, a = sp_signal.butter(2, cutoff / nyq, btype="low")
        tonic = sp_signal.filtfilt(b, a, eda)
    else:
        # Fallback: use moving average
        win = max(int(fs * 4), 3)
        if win % 2 == 0:
            win += 1
        win = min(win, len(eda))
        tonic = np.convolve(eda, np.ones(win) / win, mode="same")

    tonic_mean = float(np.mean(tonic))

    # Phasic component
    phasic = eda - tonic

    # ── SCR peak detection ──
    # Find peaks in phasic component
    min_amplitude = 0.01  # microsiemens threshold
    min_distance = max(int(fs * 1.0), 1)  # at least 1 second apart

    if len(phasic) > 2:
        peaks, properties = sp_signal.find_peaks(
            phasic,
            height=min_amplitude,
            distance=min_distance,
        )
    else:
        peaks = np.array([], dtype=int)
        properties = {"peak_heights": np.array([])}

    n_scr_peaks = len(peaks)
    scr_rate = (n_scr_peaks / duration * 60) if duration > 0 else 0.0

    if n_scr_peaks > 0:
        amplitudes = properties.get("peak_heights", phasic[peaks])
        mean_scr_amplitude = float(np.mean(amplitudes))
        max_scr_amplitude = float(np.max(amplitudes))
    else:
        mean_scr_amplitude = 0.0
        max_scr_amplitude = 0.0

    return {
        "mean_scl": round(mean_scl, 4),
        "min_scl": round(min_scl, 4),
        "max_scl": round(max_scl, 4),
        "tonic_mean": round(tonic_mean, 4),
        "n_scr_peaks": n_scr_peaks,
        "scr_rate_per_min": round(scr_rate, 2),
        "mean_scr_amplitude": round(mean_scr_amplitude, 4),
        "max_scr_amplitude": round(max_scr_amplitude, 4),
        "duration_seconds": round(duration, 2),
        "n_samples": n_samples,
    }


@router.post("/accelerometer")
def accelerometer_analysis(req: AccelerometerRequest):
    """Tri-axial accelerometer analysis for activity/movement quantification."""
    x = np.array(req.x, dtype=float)
    y = np.array(req.y, dtype=float)
    z = np.array(req.z, dtype=float)
    fs = req.sampling_rate

    if not (len(x) == len(y) == len(z)):
        raise HTTPException(400, "x, y, z arrays must have the same length.")
    n_samples = len(x)
    if n_samples < 4:
        raise HTTPException(400, "At least 4 samples required.")
    if fs <= 0:
        raise HTTPException(400, "Sampling rate must be positive.")

    duration = n_samples / fs

    # ── Signal magnitude ──
    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    mean_magnitude = float(np.mean(magnitude))

    # Signal Magnitude Area (SMA) — normalized by duration
    sma = float(np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / n_samples)

    # ── Jerk (derivative of acceleration) ──
    dt = 1.0 / fs
    jerk_x = np.diff(x) / dt
    jerk_y = np.diff(y) / dt
    jerk_z = np.diff(z) / dt
    jerk_mag = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)
    mean_jerk = float(np.mean(jerk_mag))

    # ── Dominant frequency (FFT on magnitude) ──
    # Remove DC component
    mag_centered = magnitude - np.mean(magnitude)

    if n_samples >= 8:
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(mag_centered))
        # Ignore DC (index 0) and very low frequencies
        min_idx = max(1, int(0.5 * n_samples / fs))  # above ~0.5 Hz
        if min_idx < len(fft_vals):
            dominant_idx = min_idx + np.argmax(fft_vals[min_idx:])
            dominant_freq = float(freqs[dominant_idx])
        else:
            dominant_freq = 0.0
    else:
        dominant_freq = 0.0

    # ── Step detection ──
    # Band-pass filter magnitude for step frequencies (0.5 - 5 Hz)
    nyq = fs / 2.0
    low_cut = min(0.5, nyq * 0.8)
    high_cut = min(5.0, nyq * 0.8)

    if high_cut > low_cut and n_samples > 12:
        b, a = sp_signal.butter(2, [low_cut / nyq, high_cut / nyq], btype="band")
        filtered = sp_signal.filtfilt(b, a, mag_centered)

        # Peak detection for steps
        min_step_distance = max(int(fs * 0.3), 1)  # min 0.3s between steps
        step_threshold = np.std(filtered) * 0.5
        peaks, _ = sp_signal.find_peaks(
            filtered,
            height=step_threshold,
            distance=min_step_distance,
        )
        estimated_steps = len(peaks)
    else:
        estimated_steps = 0

    step_rate = (estimated_steps / duration * 60) if duration > 0 else 0.0

    # ── Activity level classification ──
    if sma < 0.5:
        activity_level = "sedentary"
    elif sma < 2.0:
        activity_level = "light"
    elif sma < 5.0:
        activity_level = "moderate"
    else:
        activity_level = "vigorous"

    return {
        "n_samples": n_samples,
        "duration_seconds": round(duration, 2),
        "mean_magnitude": round(mean_magnitude, 4),
        "signal_magnitude_area": round(sma, 4),
        "mean_jerk": round(mean_jerk, 4),
        "dominant_frequency_hz": round(dominant_freq, 4),
        "estimated_steps": estimated_steps,
        "step_rate_per_min": round(step_rate, 2),
        "activity_level": activity_level,
    }


@router.post("/session-summary")
def session_summary(req: SessionSummaryRequest):
    """Multi-signal biometric session summary."""
    if req.duration_seconds <= 0:
        raise HTTPException(400, "duration_seconds must be positive.")

    result: dict = {
        "duration_seconds": req.duration_seconds,
    }

    # ── Heart rate summary ──
    if req.heart_rate is not None:
        hr = np.array(req.heart_rate, dtype=float)
        if len(hr) < 1:
            raise HTTPException(400, "heart_rate array must not be empty.")
        hr_clean = hr[(hr > 30) & (hr < 220)]  # basic artifact rejection
        if len(hr_clean) == 0:
            raise HTTPException(400, "No valid heart rate values (expected 30-220 bpm).")

        # Resting estimate: 10th percentile of sorted HR
        resting = float(np.percentile(hr_clean, 10))

        result["heart_rate"] = {
            "mean": round(float(np.mean(hr_clean)), 2),
            "min": round(float(np.min(hr_clean)), 2),
            "max": round(float(np.max(hr_clean)), 2),
            "resting_estimate": round(resting, 2),
        }

    # ── EDA summary ──
    if req.eda is not None:
        eda = np.array(req.eda, dtype=float)
        if len(eda) < 1:
            raise HTTPException(400, "eda array must not be empty.")

        result["eda"] = {
            "mean_scl": round(float(np.mean(eda)), 4),
            "range": round(float(np.max(eda) - np.min(eda)), 4),
        }

    # ── Temperature summary ──
    if req.temperature is not None:
        temp = np.array(req.temperature, dtype=float)
        if len(temp) < 1:
            raise HTTPException(400, "temperature array must not be empty.")

        result["temperature"] = {
            "mean": round(float(np.mean(temp)), 2),
            "min": round(float(np.min(temp)), 2),
            "max": round(float(np.max(temp)), 2),
        }

    return result
