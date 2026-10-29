"""Train exploratory Parkinson's voice models on UCI replicated acoustic features.

Uses GroupKFold on participant ID to prevent recording leakage across folds.
Compares:
  1) Pre-aggregation: Average recordings per participant prior to modeling (80 rows).
  2) Post-aggregation: Train at recording level (240 rows) with participant-grouped CV,
     then aggregate out-of-fold recording probabilities to participant level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "external" / "uci_parkinsons" / "data.csv"
DEFAULT_MODEL = ROOT / "models" / "parkinsons_voice.joblib"
DEFAULT_REPORT = ROOT / "reports" / "parkinsons_voice_metrics.json"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def get_model_definitions() -> dict:
    return {
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
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "SVM": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
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
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__kernel": ["rbf", "linear"],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=SEED
                    ),
                ),
            ]),
            "grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 8, None],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=SEED),
                ),
            ]),
            "grid": {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 4],
            },
        },
    }


def evaluate_participant_averaged(X_subj: pd.DataFrame, y_subj: pd.Series, groups: pd.Series) -> dict:
    """Strategy 1: Average recordings per participant first (80 rows), then 5-fold CV."""
    gkf = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    models = get_model_definitions()
    results = {}

    for name, cfg in models.items():
        fold_metrics = []
        oof_probs = np.zeros(len(X_subj))
        oof_preds = np.zeros(len(X_subj))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_subj, y_subj, groups)):
            X_tr, y_tr = X_subj.iloc[train_idx], y_subj.iloc[train_idx]
            X_val, y_val = X_subj.iloc[val_idx], y_subj.iloc[val_idx]
            gr_tr = groups.iloc[train_idx]

            if cfg["grid"]:
                inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold)
                search = GridSearchCV(cfg["pipeline"], cfg["grid"], cv=inner_cv, scoring="roc_auc", n_jobs=-1)
                search.fit(X_tr, y_tr, groups=gr_tr)
                estimator = search.best_estimator_
            else:
                estimator = cfg["pipeline"]
                estimator.fit(X_tr, y_tr)

            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X_val)[:, 1]
            else:
                probs = np.full(len(X_val), 0.5)
            preds = estimator.predict(X_val)

            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds

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

        prob_true, prob_pred = calibration_curve(y_subj, oof_probs, n_bins=5, strategy="uniform")
        results[name] = {
            m: {
                "mean": float(np.mean([f[m] for f in fold_metrics])),
                "std": float(np.std([f[m] for f in fold_metrics])),
            }
            for m in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "brier_score"]
        }
        results[name]["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        results[name]["oof_brier_score"] = float(brier_score_loss(y_subj, oof_probs))

    return results


def evaluate_recording_level_with_aggregation(
    X_rec: pd.DataFrame, y_rec: pd.Series, groups_rec: pd.Series
) -> tuple[dict, str, Pipeline]:
    """Strategy 2: Train on recordings (240 rows) grouped by participant, aggregate out-of-fold probs to participant."""
    gkf = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    models = get_model_definitions()
    results = {}
    best_auc = -1.0
    best_name = "LogisticRegression"
    best_pipeline = None

    unique_participants = groups_rec.unique()
    participant_labels = y_rec.groupby(groups_rec).first()

    for name, cfg in models.items():
        rec_oof_probs = np.zeros(len(X_rec))
        fold_participant_metrics = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_rec, y_rec, groups_rec)):
            X_tr, y_tr = X_rec.iloc[train_idx], y_rec.iloc[train_idx]
            X_val, y_val = X_rec.iloc[val_idx], y_rec.iloc[val_idx]
            gr_tr = groups_rec.iloc[train_idx]

            # Verify no participant leakage in fold
            assert set(gr_tr).isdisjoint(set(groups_rec.iloc[val_idx])), "Participant leakage detected in fold"

            if cfg["grid"]:
                inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold)
                search = GridSearchCV(cfg["pipeline"], cfg["grid"], cv=inner_cv, scoring="roc_auc", n_jobs=-1)
                search.fit(X_tr, y_tr, groups=gr_tr)
                estimator = search.best_estimator_
            else:
                estimator = cfg["pipeline"]
                estimator.fit(X_tr, y_tr)

            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X_val)[:, 1]
            else:
                probs = np.full(len(X_val), 0.5)

            rec_oof_probs[val_idx] = probs

            # Fold-level participant aggregation
            val_df = pd.DataFrame({
                "participant_id": groups_rec.iloc[val_idx],
                "y_true": y_val,
                "prob": probs,
            })
            subj_val = val_df.groupby("participant_id").agg({"y_true": "first", "prob": "mean"})
            subj_preds = (subj_val["prob"].to_numpy() >= 0.5).astype(int)
            subj_y = subj_val["y_true"].to_numpy()

            tn = np.sum((subj_y == 0) & (subj_preds == 0))
            fp = np.sum((subj_y == 0) & (subj_preds == 1))
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            fold_participant_metrics.append({
                "accuracy": float(accuracy_score(subj_y, subj_preds)),
                "balanced_accuracy": float(balanced_accuracy_score(subj_y, subj_preds)),
                "sensitivity": float(recall_score(subj_y, subj_preds, zero_division=0)),
                "specificity": specificity,
                "f1": float(f1_score(subj_y, subj_preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(subj_y, subj_val["prob"])) if len(np.unique(subj_y)) > 1 else 0.5,
                "brier_score": float(brier_score_loss(subj_y, subj_val["prob"])),
            })

        # Global participant out-of-fold aggregation
        all_rec_df = pd.DataFrame({
            "participant_id": groups_rec,
            "y_true": y_rec,
            "prob": rec_oof_probs,
        })
        global_subj = all_rec_df.groupby("participant_id").agg({"y_true": "first", "prob": "mean"})
        prob_true, prob_pred = calibration_curve(global_subj["y_true"], global_subj["prob"], n_bins=5, strategy="uniform")

        results[name] = {
            m: {
                "mean": float(np.mean([f[m] for f in fold_participant_metrics])),
                "std": float(np.std([f[m] for f in fold_participant_metrics])),
            }
            for m in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "brier_score"]
        }
        results[name]["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        results[name]["oof_brier_score"] = float(brier_score_loss(global_subj["y_true"], global_subj["prob"]))

        if results[name]["roc_auc"]["mean"] > best_auc and name != "Dummy":
            best_auc = results[name]["roc_auc"]["mean"]
            best_name = name

    # Fit final pipeline on participant-averaged data for cleanest single-input inference
    best_cfg = models[best_name]
    if best_cfg["grid"]:
        final_search = GridSearchCV(
            best_cfg["pipeline"],
            best_cfg["grid"],
            cv=StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED),
            scoring="roc_auc",
            n_jobs=-1,
        )
        final_search.fit(X_rec, y_rec, groups=groups_rec)
        best_pipeline = final_search.best_estimator_
    else:
        best_pipeline = best_cfg["pipeline"]
        best_pipeline.fit(X_rec, y_rec)

    return results, best_name, best_pipeline


def compute_voice_shap(pipeline: Pipeline, X: pd.DataFrame, feature_names: list[str]) -> dict:
    step_names = [name for name, _ in pipeline.steps[:-1]]
    X_trans = X.copy()
    for s in step_names:
        X_trans = pd.DataFrame(pipeline.named_steps[s].transform(X_trans), columns=feature_names)

    classifier = pipeline.named_steps["classifier"]
    try:
        if isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(X_trans)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                vals = np.abs(shap_vals[1]).mean(axis=0)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                vals = np.abs(shap_vals[:, :, 1]).mean(axis=0)
            else:
                vals = np.abs(shap_vals).mean(axis=0)
        elif isinstance(classifier, LogisticRegression):
            explainer = shap.LinearExplainer(classifier, X_trans)
            shap_vals = explainer.shap_values(X_trans)
            vals = np.abs(shap_vals).mean(axis=0)
        else:
            bg = shap.kmeans(X_trans, min(15, len(X_trans)))
            explainer = shap.KernelExplainer(classifier.predict_proba, bg)
            shap_vals = explainer.shap_values(X_trans.iloc[:25])
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                vals = np.abs(shap_vals[1]).mean(axis=0)
            else:
                vals = np.abs(shap_vals).mean(axis=0)
    except Exception as e:
        print(f"SHAP calculation fallback: {e}")
        if hasattr(classifier, "coef_"):
            vals = np.abs(classifier.coef_[0])
        elif hasattr(classifier, "feature_importances_"):
            vals = classifier.feature_importances_
        else:
            vals = np.ones(len(feature_names))

    df_imp = pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values(
        by="importance", ascending=False
    )
    return {
        "top_features": [
            {"feature": r["feature"], "importance": round(float(r["importance"]), 5)}
            for _, r in df_imp.head(20).iterrows()
        ]
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    required = {"ID", "Recording", "Status"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"Expected columns: {sorted(required)}")

    frame["Status"] = pd.to_numeric(frame["Status"], errors="raise").astype(int)
    if frame.groupby("ID")["Status"].nunique().max() != 1:
        raise SystemExit("A participant has inconsistent status labels.")

    metadata_cols = ["ID", "Recording", "Status"]
    feature_cols = [c for c in frame.columns if c not in metadata_cols]
    X_rec = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_rec = frame["Status"]
    groups_rec = frame["ID"]

    # Pre-aggregated version (averaging recordings per participant)
    grouped = frame.groupby("ID", as_index=False).agg(
        {c: "mean" for c in feature_cols} | {"Status": "first"}
    )
    X_subj = grouped[feature_cols]
    y_subj = grouped["Status"]
    groups_subj = grouped["ID"]

    print(f"Evaluating Parkinson's voice: {len(groups_subj)} participants, {len(frame)} total recordings.")

    print("\n--- Strategy 1: Pre-aggregation (Participant-level averaging) ---")
    results_pre = evaluate_participant_averaged(X_subj, y_subj, groups_subj)

    print("\n--- Strategy 2: Post-aggregation (Recording-level training with GroupKFold) ---")
    results_post, best_model_name, best_pipeline = evaluate_recording_level_with_aggregation(
        X_rec, y_rec, groups_rec
    )

    print(f"\nBest model selected: {best_model_name}")
    print("Pre-agg Logistic AUC:", results_pre.get("LogisticRegression", {}).get("roc_auc"))
    print("Post-agg Logistic AUC:", results_post.get("LogisticRegression", {}).get("roc_auc"))

    shap_summary = compute_voice_shap(best_pipeline, X_rec, feature_cols)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": feature_cols,
            "labels": {0: "control", 1: "Parkinson's cohort"},
            "dataset_version": "UCI Parkinsons Replicated Acoustic Features DOI 10.24432/C5701F (CC BY 4.0)",
            "modality": "acoustic voice features (jitter, shimmer, MFCCs, deltas)",
            "best_model_name": best_model_name,
            "participant_count": int(len(groups_subj)),
            "recording_count": int(len(frame)),
            "cv_metrics": results_post[best_model_name],
            "shap_summary": shap_summary,
        },
        args.model_out,
    )

    report = {
        "task": "Parkinson's cohort vs control from acoustic voice features",
        "dataset": {
            "name": "UCI Parkinson Dataset with Replicated Acoustic Features",
            "doi": "10.24432/C5701F",
            "license": "CC BY 4.0",
            "participants": int(len(groups_subj)),
            "recordings": int(len(frame)),
            "features": len(feature_cols),
            "class_distribution_participants": {
                "control": int((y_subj == 0).sum()),
                "parkinsons": int((y_subj == 1).sum()),
            },
        },
        "evaluation_protocol": (
            "Participant-grouped 5-fold CV (GroupKFold) with nested 3-fold inner CV. "
            "Recordings from the same participant are strictly quarantined to a single fold. "
            "Comparison between pre-aggregation (averaging features per participant) and post-aggregation (recording-level fit with participant-level voting)."
        ),
        "comparison_strategies": {
            "pre_aggregation_participant_level": results_pre,
            "post_aggregation_recording_level": results_post,
        },
        "selected_best_model": best_model_name,
        "best_model_metrics": results_post[best_model_name],
        "shap_explanations": shap_summary,
        "disclaimer": (
            "Research and educational prototype only. Not a medical device. "
            "Not clinically validated for diagnostic, prognostic, or screening purposes."
        ),
    }

    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Parkinson's voice model: {args.model_out}")
    print(f"Saved Parkinson's voice report: {args.report_out}")


if __name__ == "__main__":
    main()
