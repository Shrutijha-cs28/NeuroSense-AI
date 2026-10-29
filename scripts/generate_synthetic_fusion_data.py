"""Generate synthetic matched multi-modal datasets for pipeline plumbing verification.

WARNING:
This data is purely SYNTHETIC. It is for plumbing tests only, NOT scientific or clinical results.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "data" / "synthetic"


def generate_synthetic_alzheimer(n_participants: int = 60, seed: int = 42) -> Path:
    np.random.seed(seed)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(n_participants):
        p_id = f"SYNTH_ALZ_{i:03d}"
        label = int(np.random.rand() > 0.5)

        # 1 to 2 visits per participant
        n_visits = np.random.choice([1, 2], p=[0.7, 0.3])
        for v in range(n_visits):
            visit_date = f"2025-0{v+1}-15"

            # Synthetic MRI features (correlated with label)
            hippocampus = np.random.normal(3200 if label == 1 else 4100, 300)
            cortical = np.random.normal(2.1 if label == 1 else 2.6, 0.15)
            ventricle = np.random.normal(45000 if label == 1 else 32000, 4000)

            # Synthetic EEG features (correlated with label)
            theta_power = np.random.normal(0.42 if label == 1 else 0.22, 0.08)
            alpha_peak = np.random.normal(8.2 if label == 1 else 10.1, 0.7)
            entropy = np.random.normal(0.78 if label == 1 else 0.65, 0.06)

            # Introduce intentional missing modality in ~10% of rows
            has_mri = np.random.rand() > 0.10
            has_eeg = np.random.rand() > 0.10

            rows.append({
                "participant_id": p_id,
                "label": label,
                "visit_date": visit_date,
                "mri__hippocampus_volume": round(hippocampus, 2) if has_mri else np.nan,
                "mri__cortical_thickness": round(cortical, 3) if has_mri else np.nan,
                "mri__ventricle_volume": round(ventricle, 2) if has_mri else np.nan,
                "eeg__theta_power": round(theta_power, 4) if has_eeg else np.nan,
                "eeg__alpha_peak_freq": round(alpha_peak, 2) if has_eeg else np.nan,
                "eeg__spectral_entropy": round(entropy, 4) if has_eeg else np.nan,
                "data_type": "SYNTHETIC_PLUMBING_TEST_NOT_A_RESULT",
            })

    out_csv = SYNTH_DIR / "alzheimer_matched_synthetic.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Generated synthetic Alzheimer matched dataset: {out_csv} ({len(df)} rows)")
    return out_csv


def generate_synthetic_parkinsons(n_participants: int = 60, seed: int = 42) -> Path:
    np.random.seed(seed)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(n_participants):
        p_id = f"SYNTH_PD_{i:03d}"
        label = int(np.random.rand() > 0.5)

        n_visits = np.random.choice([1, 2], p=[0.7, 0.3])
        for v in range(n_visits):
            visit_date = f"2025-0{v+1}-10"

            jitter = np.random.normal(0.015 if label == 1 else 0.005, 0.003)
            shimmer = np.random.normal(0.060 if label == 1 else 0.025, 0.010)
            hnr = np.random.normal(16.0 if label == 1 else 24.0, 2.5)

            stride_time = np.random.normal(1.25 if label == 1 else 1.05, 0.10)
            cadence = np.random.normal(95.0 if label == 1 else 115.0, 8.0)
            swing_time = np.random.normal(0.38 if label == 1 else 0.45, 0.04)

            tremor_amp = np.random.normal(1.8 if label == 1 else 0.3, 0.4)

            has_speech = np.random.rand() > 0.08
            has_gait = np.random.rand() > 0.08
            has_tremor = np.random.rand() > 0.08

            rows.append({
                "participant_id": p_id,
                "label": label,
                "visit_date": visit_date,
                "speech__jitter": round(max(0.0001, jitter), 5) if has_speech else np.nan,
                "speech__shimmer": round(max(0.0001, shimmer), 5) if has_speech else np.nan,
                "speech__hnr": round(max(1.0, hnr), 2) if has_speech else np.nan,
                "gait__stride_time": round(stride_time, 3) if has_gait else np.nan,
                "gait__cadence": round(cadence, 1) if has_gait else np.nan,
                "gait__swing_time": round(swing_time, 3) if has_gait else np.nan,
                "tremor__amplitude": round(max(0.0, tremor_amp), 3) if has_tremor else np.nan,
                "data_type": "SYNTHETIC_PLUMBING_TEST_NOT_A_RESULT",
            })

    out_csv = SYNTH_DIR / "parkinsons_matched_synthetic.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Generated synthetic Parkinson matched dataset: {out_csv} ({len(df)} rows)")
    return out_csv


if __name__ == "__main__":
    generate_synthetic_alzheimer()
    generate_synthetic_parkinsons()
