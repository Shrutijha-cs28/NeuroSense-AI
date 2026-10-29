"""Train disease-specific multi-modal fusion benchmarks from matched participant records.

Strict constraint:
  Never join unrelated studies by row order or guessed ID.
  Join only on verified matching participant/session IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from .fusion import EarlyFusionClassifier, LateFusionClassifier, evaluate_fusion_cv
from .registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disease", choices=["alzheimer", "parkinsons"], required=True)
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Approved, matched CSV with participant_id, label, visit_date, and modality__feature columns",
    )
    p.add_argument("--model-out", type=Path)
    p.add_argument("--report-out", type=Path)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()


def group_modality_columns(columns: list[str]) -> dict[str, list[str]]:
    modality_features: dict[str, list[str]] = {}
    for col in columns:
        if "__" in col:
            prefix = col.split("__")[0]
            modality_features.setdefault(prefix, []).append(col)
    return modality_features


def main() -> None:
    args = parse_args()
    data_path = args.data
    if not data_path.exists():
        raise SystemExit(f"Matched feature CSV not found: {data_path}")

    frame = pd.read_csv(data_path)
    required = {"participant_id", "label"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"CSV missing required columns: {required}")

    modality_features = group_modality_columns(list(frame.columns))
    if len(modality_features) < 2:
        raise SystemExit(
            f"Fusion requires at least 2 distinct modality feature prefixes (e.g., 'mri__*', 'eeg__*'). Found: {list(modality_features.keys())}"
        )

    print(f"Loaded matched dataset for {args.disease}: {len(frame)} rows, {frame['participant_id'].nunique()} participants.")
    print(f"Detected modalities: {list(modality_features.keys())}")

    # Run participant-grouped CV comparing early fusion, late fusion, and single branches
    cv_results = evaluate_fusion_cv(frame, modality_features, n_splits=args.folds, random_state=42)

    # Pick best strategy by roc_auc
    best_strategy = max(cv_results.keys(), key=lambda k: cv_results[k]["roc_auc"]["mean"])
    print(f"Top performing strategy: {best_strategy} (AUC: {cv_results[best_strategy]['roc_auc']['mean']:.4f})")

    # Fit final model
    all_numeric_features = [f for feats in modality_features.values() for f in feats]
    if "Early" in best_strategy:
        clf_type = "rf" if "RandomForest" in best_strategy else "logistic"
        final_model = EarlyFusionClassifier(classifier_type=clf_type)
        final_model.fit(frame[all_numeric_features], frame["label"].astype(int))
    else:
        strat = "stacking" if "Stacking" in best_strategy else "average"
        final_model = LateFusionClassifier(modality_features, strategy=strat)
        final_model.fit(frame[all_numeric_features], frame["label"].astype(int))

    model_out = args.model_out or (ROOT / "models" / f"{args.disease}_fusion.joblib")
    report_out = args.report_out or (ROOT / "reports" / f"{args.disease}_fusion_metrics.json")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "disease": args.disease,
            "best_strategy": best_strategy,
            "modality_features": modality_features,
            "cv_metrics": cv_results[best_strategy],
            "dataset_path": str(data_path),
        },
        model_out,
    )

    report = {
        "disease": args.disease,
        "input_dataset": str(data_path),
        "participants": int(frame["participant_id"].nunique()),
        "total_records": len(frame),
        "modality_feature_counts": {k: len(v) for k, v in modality_features.items()},
        "strategies_evaluated": cv_results,
        "selected_best_strategy": best_strategy,
        "best_strategy_metrics": cv_results[best_strategy],
        "warning": (
            "Research prototype only. Results are valid ONLY if inputs originate from verified "
            "within-participant matched data. Never merge disparate cohorts by index."
        ),
    }

    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved fusion model to: {model_out}")
    print(f"Saved fusion report to: {report_out}")


if __name__ == "__main__":
    main()
