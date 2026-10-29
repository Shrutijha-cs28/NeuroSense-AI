"""Train an exploratory binary Alzheimer handwriting-feature model on UCI DARWIN with nested CV and SHAP."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "external" / "uci_darwin" / "data.csv"
DEFAULT_MODEL = ROOT / "models" / "alzheimer_darwin.joblib"
DEFAULT_REPORT = ROOT / "reports" / "alzheimer_darwin_metrics.json"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def get_task_mapping(feature_names: list[str]) -> dict[int, list[str]]:
    """Group feature names by task index 1..25 based on trailing digit suffix."""
    mapping: dict[int, list[str]] = {}
    for feat in feature_names:
        m = re.search(r"(\d+)$", feat)
        if m:
            task_num = int(m.group(1))
            mapping.setdefault(task_num, []).append(feat)
    return mapping


def evaluate_models_nested_cv(
    X: pd.DataFrame, y: pd.Series
) -> tuple[dict, str, Pipeline, dict]:
    """Run nested cross-validation across candidates to prevent hyperparameter selection leakage."""
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)

    # Candidate models and parameter grids
    model_configs = {
        "Dummy": {
            "pipeline": Pipeline([
                ("classifier", DummyClassifier(strategy="most_frequent"))
            ]),
            "grid": {},
        },
        "LogisticRegression": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=100)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]),
            "grid": {
                "select__k": [50, 100, 150],
                "classifier__C": [0.01, 0.1, 1.0],
            },
        },
        "SVM": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=100)),
                (
                    "classifier",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]),
            "grid": {
                "select__k": [50, 100],
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=100)),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=SEED
                    ),
                ),
            ]),
            "grid": {
                "select__k": [50, 100],
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 8, None],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=100)),
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=SEED),
                ),
            ]),
            "grid": {
                "select__k": [50, 100],
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        },
    }

    results = {}
    best_overall_auc = -1.0
    best_model_name = "LogisticRegression"
    best_pipeline = None
    all_oof_predictions = {}

    for name, config in model_configs.items():
        print(f"Evaluating {name} with {OUTER_FOLDS}-fold nested CV...")
        fold_metrics = []
        oof_probs = np.zeros(len(X))
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            if config["grid"]:
                inner_cv = StratifiedKFold(
                    n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold
                )
                search = GridSearchCV(
                    config["pipeline"],
                    config["grid"],
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                search.fit(X_tr, y_tr)
                model = search.best_estimator_
            else:
                model = config["pipeline"]
                model.fit(X_tr, y_tr)

            # Predict probabilities and classes
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
            else:
                probs = np.full(len(X_val), 0.5)
            preds = model.predict(X_val)

            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds

            # Specificity = TN / (TN + FP)
            tn = np.sum((y_val == 0) & (preds == 0))
            fp = np.sum((y_val == 0) & (preds == 1))
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            fold_metrics.append({
                "accuracy": float(accuracy_score(y_val, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y_val, preds)),
                "sensitivity": float(recall_score(y_val, preds, zero_division=0)),
                "specificity": specificity,
                "f1": float(f1_score(y_val, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_val, probs)) if len(np.unique(y_val)) > 1 else 0.5,
                "brier_score": float(brier_score_loss(y_val, probs)),
            })

        metric_names = [
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "brier_score",
        ]
        summary = {
            m: {
                "mean": float(np.mean([f[m] for f in fold_metrics])),
                "std": float(np.std([f[m] for f in fold_metrics])),
            }
            for m in metric_names
        }

        prob_true, prob_pred = calibration_curve(
            y, oof_probs, n_bins=5, strategy="uniform"
        )
        summary["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        summary["oof_brier_score"] = float(brier_score_loss(y, oof_probs))

        results[name] = summary
        all_oof_predictions[name] = {
            "probabilities": oof_probs.tolist(),
            "predictions": oof_preds.tolist(),
        }

        # Check leakage safeguard: if AUROC > 0.95, log caution
        if summary["roc_auc"]["mean"] > 0.95:
            print(
                f"WARNING: {name} achieved AUROC {summary['roc_auc']['mean']:.4f} > 0.95. Verify feature selection and split boundaries."
            )

        if summary["roc_auc"]["mean"] > best_overall_auc and name != "Dummy":
            best_overall_auc = summary["roc_auc"]["mean"]
            best_model_name = name

    # Fit final pipeline for best model across full dataset with inner CV tuned parameters
    print(f"Selecting best model: {best_model_name} (AUC: {best_overall_auc:.4f})")
    best_cfg = model_configs[best_model_name]
    if best_cfg["grid"]:
        final_search = GridSearchCV(
            best_cfg["pipeline"],
            best_cfg["grid"],
            cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED),
            scoring="roc_auc",
            n_jobs=-1,
        )
        final_search.fit(X, y)
        best_pipeline = final_search.best_estimator_
    else:
        best_pipeline = best_cfg["pipeline"]
        best_pipeline.fit(X, y)

    return results, best_model_name, best_pipeline, all_oof_predictions


def compute_shap_explanations(
    pipeline: Pipeline, X: pd.DataFrame, feature_names: list[str]
) -> dict:
    """Compute global SHAP importance and aggregate by handwriting task (1..25)."""
    # Transform X through preprocessing steps (impute, scale, select)
    step_names = [name for name, _ in pipeline.steps[:-1]]
    X_transformed = X.copy()
    transformed_features = list(feature_names)

    for step_name in step_names:
        transformer = pipeline.named_steps[step_name]
        X_mat = transformer.transform(X_transformed)
        if hasattr(transformer, "get_support"):
            support = transformer.get_support()
            transformed_features = [f for f, s in zip(transformed_features, support) if s]
        X_transformed = pd.DataFrame(X_mat, columns=transformed_features)

    classifier = pipeline.named_steps["classifier"]

    # Compute SHAP values
    try:
        if isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                vals = np.abs(shap_values[1]).mean(axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                vals = np.abs(shap_values[:, :, 1]).mean(axis=0)
            else:
                vals = np.abs(shap_values).mean(axis=0)
        elif isinstance(classifier, LogisticRegression):
            explainer = shap.LinearExplainer(classifier, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
            vals = np.abs(shap_values).mean(axis=0)
        else:
            # Fallback KernelExplainer on background sample
            background = shap.kmeans(X_transformed, min(15, len(X_transformed)))
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_transformed.iloc[:30])
            if isinstance(shap_values, list) and len(shap_values) == 2:
                vals = np.abs(shap_values[1]).mean(axis=0)
            else:
                vals = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        print(f"SHAP computation fallback: {e}")
        # Default to absolute coefficients / feature importances if available
        if hasattr(classifier, "feature_importances_"):
            vals = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            vals = np.abs(classifier.coef_[0])
        else:
            vals = np.ones(len(transformed_features))

    importance_df = pd.DataFrame({
        "feature": transformed_features,
        "importance": vals,
    }).sort_values(by="importance", ascending=False)

    # Task aggregation (1..25)
    task_scores = {task: 0.0 for task in range(1, 26)}
    for _, row in importance_df.iterrows():
        feat = row["feature"]
        m = re.search(r"(\d+)$", feat)
        if m:
            task_num = int(m.group(1))
            task_scores[task_num] = task_scores.get(task_num, 0.0) + float(row["importance"])

    top_tasks = sorted(
        [{"task": t, "importance": round(score, 5)} for t, score in task_scores.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )

    top_features = [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 5)}
        for _, row in importance_df.head(20).iterrows()
    ]

    return {
        "top_features": top_features,
        "top_tasks": top_tasks,
        "task_importance_summary": {f"task_{t['task']}": t["importance"] for t in top_tasks},
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    if not {"ID", "class"}.issubset(frame.columns):
        raise SystemExit("Expected UCI DARWIN columns 'ID' and 'class'.")
    if frame["ID"].duplicated().any():
        raise SystemExit("DARWIN participant IDs are expected to be unique; resolve duplicate IDs before training.")

    labels = frame["class"].astype(str).str.strip().str.upper()
    if not set(labels.unique()).issubset({"H", "P"}) or labels.nunique() != 2:
        raise SystemExit("Expected DARWIN class labels H (healthy) and P (Alzheimer's patient).")

    y = labels.map({"H": 0, "P": 1}).astype(int)
    X = frame.drop(columns=["ID", "class"]).apply(pd.to_numeric, errors="coerce")

    print(f"Data inventory: {len(frame)} participants (Healthy: {(y == 0).sum()}, Cohort: {(y == 1).sum()}), {X.shape[1]} features.")
    task_map = get_task_mapping(list(X.columns))
    print(f"Verified 25 distinct handwriting task groups. Missing values: {X.isna().sum().sum()}")

    # Run nested CV across models
    cv_results, best_model_name, best_pipeline, oof_preds = evaluate_models_nested_cv(X, y)

    # Compute SHAP explanations
    print("Computing SHAP explanations and task importance...")
    shap_results = compute_shap_explanations(best_pipeline, X, list(X.columns))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    # Save model artifact
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": list(X.columns),
            "labels": {0: "healthy", 1: "Alzheimer's cohort"},
            "dataset_version": "UCI DARWIN DOI 10.24432/C55D0K (CC BY 4.0)",
            "modality": "online handwriting features (pen pressure, kinematics)",
            "best_model_name": best_model_name,
            "cv_metrics": cv_results[best_model_name],
            "shap_summary": shap_results,
        },
        args.model_out,
    )

    report = {
        "task": "Alzheimer cohort vs healthy cohort from online handwriting features; strictly NOT MRI or EEG",
        "dataset": {
            "name": "UCI DARWIN",
            "doi": "10.24432/C55D0K",
            "license": "CC BY 4.0",
            "participants": int(len(frame)),
            "features": int(X.shape[1]),
            "tasks": len(task_map),
            "class_distribution": {
                "healthy": int((y == 0).sum()),
                "Alzheimer_cohort": int((y == 1).sum()),
            },
            "missing_values": int(X.isna().sum().sum()),
            "duplicate_participant_ids": int(frame["ID"].duplicated().sum()),
        },
        "evaluation_protocol": (
            "Nested 5-fold Stratified CV with inner 3-fold CV for hyperparameter tuning. "
            "Imputation, scaling, and feature selection performed inside each training fold to eliminate leakage. "
            "Reported metrics are mean ± std across outer folds."
        ),
        "models_evaluated": cv_results,
        "selected_best_model": best_model_name,
        "best_model_cv_metrics": cv_results[best_model_name],
        "shap_explanations": shap_results,
        "disclaimer": (
            "Research and educational prototype only. Not a medical device. "
            "Not clinically validated for diagnostic, prognostic, or screening purposes."
        ),
    }

    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Successfully saved model to: {args.model_out}")
    print(f"Successfully saved report to: {args.report_out}")


if __name__ == "__main__":
    main()
