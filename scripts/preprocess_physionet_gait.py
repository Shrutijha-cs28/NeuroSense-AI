"""Create a compact one-row-per-walk feature table from PhysioNet GaitPDB."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FILENAME = re.compile(r"(?P<study>Ga|Ju|Si)(?P<group>Co|Pt)(?P<subject>\d+)_(?P<walk>\d+)\.txt$")
CHANNELS = [*(f"L{i}" for i in range(1, 9)), *(f"R{i}" for i in range(1, 9)), "L_total", "R_total"]
STATISTICS = ("mean", "std", "median", "p05", "p95", "min", "max", "range", "rms", "zero_fraction")


def features_for_file(path: Path) -> dict[str, object]:
    match = FILENAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected recording filename: {path.name}")
    metadata = match.groupdict()
    values = np.loadtxt(path, delimiter="\t", ndmin=2)
    if values.ndim != 2 or values.shape[1] != 19 or values.shape[0] < 100:
        raise ValueError(f"Expected at least 100 rows and 19 columns in {path}; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {path}")

    row: dict[str, object] = {
        "participant_id": f"{metadata['study']}{metadata['group']}{metadata['subject']}",
        "recording_id": path.stem,
        "study": metadata["study"],
        "walk_condition": "dual_task" if metadata["study"] == "Ga" and metadata["walk"] == "10" else "usual" if metadata["walk"] == "01" else "other",
        "label": 1 if metadata["group"] == "Pt" else 0,
        "sampling_hz": 100,
        "n_samples": int(values.shape[0]),
        "duration_seconds": float((values[-1, 0] - values[0, 0]) if values.shape[0] > 1 else 0),
    }
    signal = values[:, 1:]
    for index, channel in enumerate(CHANNELS):
        x = signal[:, index]
        stats = {
            "mean": np.mean(x),
            "std": np.std(x, ddof=1),
            "median": np.median(x),
            "p05": np.quantile(x, 0.05),
            "p95": np.quantile(x, 0.95),
            "min": np.min(x),
            "max": np.max(x),
            "range": np.ptp(x),
            "rms": np.sqrt(np.mean(np.square(x))),
            "zero_fraction": np.mean(x == 0),
        }
        row.update({f"gait__{channel}__{name}": float(value) for name, value in stats.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/physionet_gaitpdb/1.0.0"))
    parser.add_argument("--output", type=Path, default=Path("data/external/physionet_gaitpdb/gait_features.csv"))
    args = parser.parse_args()
    paths = sorted(path for path in args.raw_dir.glob("*.txt") if FILENAME.fullmatch(path.name))
    if not paths:
        raise FileNotFoundError(f"No PhysioNet gait recordings found in {args.raw_dir}. Run scripts/download_physionet_gait.py first.")
    frame = pd.DataFrame(features_for_file(path) for path in paths)
    if frame["recording_id"].duplicated().any():
        raise ValueError("Duplicate recording IDs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.8g")
    print(f"Saved {len(frame)} walks from {frame['participant_id'].nunique()} participants to {args.output}")
    print(frame.groupby("label")["participant_id"].nunique().rename(index={0: "control", 1: "parkinsons"}).to_string())
    print(f"Feature columns: {sum(column.startswith('gait__') for column in frame.columns)}")


if __name__ == "__main__":
    main()
