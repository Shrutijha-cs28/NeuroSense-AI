"""Create global/local SHAP explanations for a saved tabular model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="CSV with the exact numeric model feature columns")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "explanations")
    parser.add_argument("--rows", type=int, default=100,
                        help="Maximum rows explained to keep runtime manageable")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install optional dependencies with: pip install -r requirements-explainability.txt") from exc

    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_names" not in bundle:
        raise SystemExit("Expected a NeuroSenseAI model bundle with pipeline and feature_names.")
    features = bundle["feature_names"]
    frame = pd.read_csv(args.data)
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise SystemExit(f"CSV is missing model feature columns: {missing[:20]}")
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="all").head(args.rows)
    if X.empty:
        raise SystemExit("No rows with numeric model features were found.")
    pipeline = bundle["pipeline"]
    background = X.sample(n=min(30, len(X)), random_state=42)
    explainer = shap.Explainer(pipeline.predict_proba, background)
    explanation = explainer(X, max_evals=max(2 * len(features) + 1, 100))
    values = explanation.values
    # SHAP versions return either (rows, features, classes) or class-specific arrays.
    disease_values = values[:, :, 1] if getattr(values, "ndim", 0) == 3 else values
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({"feature": features,
                            "mean_absolute_shap": abs(disease_values).mean(axis=0)})
    summary = summary.sort_values("mean_absolute_shap", ascending=False)
    summary.to_csv(args.output_dir / "shap_feature_summary.csv", index=False)
    plt.figure()
    shap.summary_plot(disease_values, X, show=False, max_display=min(20, len(features)))
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    result = {
        "model": str(args.model), "rows_explained": int(len(X)),
        "class_explained": "class index 1 (disease cohort)",
        "top_features_by_mean_absolute_shap": summary.head(20).to_dict(orient="records"),
        "warning": "SHAP describes model behavior on these inputs; it is not causal evidence, a diagnosis, or proof of clinical validity.",
    }
    (args.output_dir / "shap_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved SHAP summary to {args.output_dir}")


if __name__ == "__main__":
    main()
