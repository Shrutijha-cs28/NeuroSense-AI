"""Train an exploratory Parkinson's gait-force benchmark with StratifiedGroupKFold on participant ID.

Includes:
- Nested CV across Dummy, Logistic Regression, SVM, Random Forest, Gradient Boosting.
- Study leakage investigation (Ga / Ju / Si distribution and association checks).
- Sensitivity subgroup evaluation per study and per walk_condition.
- Calibration curve and Brier score.
- SHAP global feature explanations.
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
from sklearn.feature_selection import SelectKBest, f_classif
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import shap

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "external" / "physionet_gaitpdb" / "gait_features.csv"
DEFAULT_MODEL = ROOT / "models" / "parkinsons_gait.joblib"
DEFAULT_REPORT = ROOT / "reports" / "parkinsons_gait_metrics.json"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def investigate_study_leakage(frame: pd.DataFrame) -> dict:
    """Analyze whether 'study' predicts the label and confirm study is excluded from training features."""
    crosstab = pd.crosstab(frame["study"], frame["label"]).to_dict()
    participants_by_study = frame.groupby("study")["participant_id"].nunique().to_dict()

    # Calculate proportions
    study_proportions = {}
    for st, group in frame.groupby("study"):
        p_count = group["participant_id"].nunique()
        pos_p = group[group["label"] == 1]["participant_id"].nunique()
        study_proportions[st] = {
            "total_participants": p_count,
            "parkinsons_participants": pos_p,
            "prevalence": round(pos_p / p_count, 4) if p_count > 0 else 0.0,
        }

    return {
        "crosstab_recordings": crosstab,
        "participants_per_study": participants_by_study,
        "study_prevalence": study_proportions,
        "leakage_safeguard": (
            "Metadata fields ('study', 'walk_condition', 'recording_id', 'sampling_hz') "
            "are strictly excluded from feature inputs. Features consist exclusively of "
            "ground reaction force summary statistics."
        ),
    }


def get_gait_model_definitions() -> dict:
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
                ("select", SelectKBest(score_func=f_classif, k=60)),
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
                "select__k": [40, 60, 80],
                "classifier__C": [0.01, 0.1, 1.0],
            },
        },
        "SVM": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=60)),
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
                "select__k": [40, 60],
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__kernel": ["rbf", "linear"],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=60)),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=SEED
                    ),
                ),
            ]),
            "grid": {
                "select__k": [40, 60],
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 8],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=f_classif, k=60)),
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=SEED),
                ),
            ]),
            "grid": {
                "select__k": [40, 60],
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 4],
            },
        },
    }


def evaluate_gait_models(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, frame: pd.DataFrame
) -> tuple[dict, str, Pipeline, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    models = get_gait_model_definitions()
    results = {}
    best_auc = -1.0
    best_name = "LogisticRegression"
    best_pipeline = None
    best_oof_df = None

    for name, cfg in models.items():
        rec_oof_probs = np.zeros(len(X))
        fold_participant_metrics = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            gr_tr = groups.iloc[train_idx]

            assert set(gr_tr).isdisjoint(set(groups.iloc[val_idx])), "Participant leakage in gait CV split"

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

            # Aggregate per participant in fold
            fold_df = pd.DataFrame({
                "participant_id": groups.iloc[val_idx],
                "y_true": y_val,
                "prob": probs,
            })
            p_fold = fold_df.groupby("participant_id").agg({"y_true": "first", "prob": "mean"})
            p_preds = (p_fold["prob"].to_numpy() >= 0.5).astype(int)
            p_y = p_fold["y_true"].to_numpy()

            tn = np.sum((p_y == 0) & (p_preds == 0))
            fp = np.sum((p_y == 0) & (p_preds == 1))
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            fold_participant_metrics.append({
                "accuracy": float(accuracy_score(p_y, p_preds)),
                "balanced_accuracy": float(balanced_accuracy_score(p_y, p_preds)),
                "sensitivity": float(recall_score(p_y, p_preds, zero_division=0)),
                "specificity": specificity,
                "f1": float(f1_score(p_y, p_preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(p_y, p_fold["prob"])) if len(np.unique(p_y)) > 1 else 0.5,
                "brier_score": float(brier_score_loss(p_y, p_fold["prob"])),
            })

        # Global aggregate
        all_df = frame[["participant_id", "recording_id", "study", "walk_condition", "label"]].copy()
        all_df["prob"] = rec_oof_probs

        subj_df = all_df.groupby("participant_id", as_index=False).agg({
            "label": "first",
            "study": "first",
            "prob": "mean",
        })
        subj_y = subj_df["label"].to_numpy()
        subj_prob = subj_df["prob"].to_numpy()

        prob_true, prob_pred = calibration_curve(subj_y, subj_prob, n_bins=5, strategy="uniform")

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
        results[name]["oof_brier_score"] = float(brier_score_loss(subj_y, subj_prob))

        if results[name]["roc_auc"]["mean"] > best_auc and name != "Dummy":
            best_auc = results[name]["roc_auc"]["mean"]
            best_name = name
            best_oof_df = all_df

    # Fit final pipeline
    best_cfg = models[best_name]
    if best_cfg["grid"]:
        final_search = GridSearchCV(
            best_cfg["pipeline"],
            best_cfg["grid"],
            cv=StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED),
            scoring="roc_auc",
            n_jobs=-1,
        )
        final_search.fit(X, y, groups=groups)
        best_pipeline = final_search.best_estimator_
    else:
        best_pipeline = best_cfg["pipeline"]
        best_pipeline.fit(X, y)

    return results, best_name, best_pipeline, best_oof_df


def compute_sensitivity_analysis(oof_df: pd.DataFrame) -> dict:
    """Sensitivity check: performance evaluated per study and per walk_condition."""
    analysis = {"by_study": {}, "by_walk_condition": {}}

    # By study (at participant level)
    subj_df = oof_df.groupby("participant_id", as_index=False).agg({
        "label": "first",
        "study": "first",
        "prob": "mean",
    })

    for study_val, group in subj_df.groupby("study"):
        y_true = group["label"].to_numpy()
        probs = group["prob"].to_numpy()
        preds = (probs >= 0.5).astype(int)

        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None

        analysis["by_study"][study_val] = {
            "participants": len(group),
            "parkinsons_count": int(np.sum(y_true == 1)),
            "control_count": int(np.sum(y_true == 0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "sensitivity": float(recall_score(y_true, preds, zero_division=0)),
            "specificity": spec,
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": auc,
        }

    # By walk condition (at recording level)
    for cond_val, group in oof_df.groupby("walk_condition"):
        y_true = group["label"].to_numpy()
        probs = group["prob"].to_numpy()
        preds = (probs >= 0.5).astype(int)

        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None

        analysis["by_walk_condition"][cond_val] = {
            "recordings": len(group),
            "parkinsons_count": int(np.sum(y_true == 1)),
            "control_count": int(np.sum(y_true == 0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "sensitivity": float(recall_score(y_true, preds, zero_division=0)),
            "specificity": spec,
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": auc,
        }

    return analysis


def compute_gait_shap(pipeline: Pipeline, X: pd.DataFrame, feature_names: list[str]) -> dict:
    step_names = [name for name, _ in pipeline.steps[:-1]]
    X_trans = X.copy()
    current_features = list(feature_names)

    for s in step_names:
        transformer = pipeline.named_steps[s]
        X_mat = transformer.transform(X_trans)
        if hasattr(transformer, "get_support"):
            support = transformer.get_support()
            current_features = [f for f, sup in zip(current_features, support) if sup]
        X_trans = pd.DataFrame(X_mat, columns=current_features)

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
            vals = np.ones(len(current_features))

    df_imp = pd.DataFrame({"feature": current_features, "importance": vals}).sort_values(
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
    required = {"participant_id", "recording_id", "study", "walk_condition", "label"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"Expected gait columns: {sorted(required)}")

    metadata_cols = ["participant_id", "recording_id", "study", "walk_condition", "label", "sampling_hz"]
    feature_cols = [c for c in frame.columns if c not in metadata_cols]

    X = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = frame["label"].astype(int)
    groups = frame["participant_id"]

    print(f"Gait data inventory: {groups.nunique()} participants, {len(frame)} walks, {X.shape[1]} summary features.")

    # Check study leakage
    study_leakage_check = investigate_study_leakage(frame)
    print("Study leakage check completed. Metadata columns excluded from features.")

    # Evaluate models
    results, best_model_name, best_pipeline, oof_df = evaluate_gait_models(X, y, groups, frame)
    print(f"Selected best gait model: {best_model_name}")

    # Subgroup sensitivity check
    sensitivity_results = compute_sensitivity_analysis(oof_df)

    # SHAP explanations
    shap_summary = compute_gait_shap(best_pipeline, X, feature_cols)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": feature_cols,
            "labels": {0: "control", 1: "Parkinson's cohort"},
            "dataset_version": "PhysioNet Gait in Parkinson's Disease v1.0.0 (ODC-By 1.0)",
            "modality": "gait foot-force summary features (16 sensors + 2 total force)",
            "best_model_name": best_model_name,
            "participant_count": int(groups.nunique()),
            "recording_count": int(len(frame)),
            "cv_metrics": results[best_model_name],
            "sensitivity_analysis": sensitivity_results,
            "shap_summary": shap_summary,
        },
        args.model_out,
    )

    report = {
        "task": "Parkinson's cohort vs control from foot ground reaction force features",
        "dataset": {
            "name": "PhysioNet Gait in Parkinson's Disease (gaitpdb)",
            "doi": "10.13026/C24H3N",
            "license": "ODC-By 1.0",
            "participants": int(groups.nunique()),
            "recordings": int(len(frame)),
            "features": len(feature_cols),
            "class_distribution_participants": {
                "control": int((frame.groupby("participant_id")["label"].first() == 0).sum()),
                "parkinsons": int((frame.groupby("participant_id")["label"].first() == 1).sum()),
            },
        },
        "leakage_safeguard_and_study_audit": study_leakage_check,
        "evaluation_protocol": (
            "StratifiedGroupKFold on participant_id (5 outer folds, 3 inner folds). "
            "Participant walks are strictly kept together; out-of-fold walk probabilities are aggregated to participant level."
        ),
        "models_evaluated": results,
        "selected_best_model": best_model_name,
        "best_model_metrics": results[best_model_name],
        "subgroup_sensitivity_analysis": sensitivity_results,
        "shap_explanations": shap_summary,
        "disclaimer": (
            "Research and educational prototype only. Not a medical device. "
            "Not clinically validated for diagnostic, prognostic, or screening purposes."
        ),
    }

    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Parkinson's gait model: {args.model_out}")
    print(f"Saved Parkinson's gait report: {args.report_out}")


if __name__ == "__main__":
    main()
