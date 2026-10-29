"""Research-only per-session score history using an existing fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="Repeated-visit CSV containing participant_id, visit_date, and model features")
    parser.add_argument("--output", type=Path, default=Path("reports") / "longitudinal_scores.csv")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_names" not in bundle:
        raise SystemExit("Expected a NeuroSenseAI model bundle with pipeline and feature_names.")
    frame = pd.read_csv(args.data)
    required = {"participant_id", "visit_date", *bundle["feature_names"]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Longitudinal CSV is missing columns: {missing[:20]}")
    if frame[["participant_id", "visit_date"]].isna().any().any():
        raise SystemExit("participant_id and visit_date cannot be blank.")
    frame["visit_date"] = pd.to_datetime(frame["visit_date"], errors="coerce")
    if frame["visit_date"].isna().any():
        raise SystemExit("Some visit_date values could not be parsed.")
    features = bundle["feature_names"]
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().all(axis=1).any():
        raise SystemExit("At least one visit has no usable model features.")
    scores = bundle["pipeline"].predict_proba(X)[:, 1]
    output = frame[["participant_id", "visit_date"]].copy()
    output["model_score_class_1"] = scores
    output = output.sort_values(["participant_id", "visit_date"])
    output["change_from_first_visit"] = output["model_score_class_1"] - output.groupby("participant_id")["model_score_class_1"].transform("first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved research score history: {args.output}")
    print("Scores are model outputs, not calibrated clinical risks or evidence of disease progression.")


if __name__ == "__main__":
    main()
