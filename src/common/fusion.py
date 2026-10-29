"""Multi-modal fusion algorithms: Early Fusion, Late Fusion, and Stacking with missing modality support."""

from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class EarlyFusionClassifier:
    """Concatenates all numeric features across modalities with missing-modality indicators and calibration."""

    def __init__(self, classifier_type: Literal["logistic", "rf"] = "logistic", random_state: int = 42):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.classes_ = np.array([0, 1])

        if classifier_type == "logistic":
            base_clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
                random_state=random_state,
            )
            self.pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("classifier", base_clf),
            ])
        else:
            base_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=random_state,
            )
            self.pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("classifier", base_clf),
            ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EarlyFusionClassifier:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class LateFusionClassifier:
    """Combines modality branches by weighted averaging or stacking probabilities, gracefully handling missing modalities."""

    def __init__(
        self,
        modality_features: dict[str, list[str]],
        strategy: Literal["average", "stacking"] = "average",
        random_state: int = 42,
    ):
        self.modality_features = modality_features
        self.strategy = strategy
        self.random_state = random_state
        self.branch_pipelines: dict[str, Pipeline] = {}
        self.meta_classifier: LogisticRegression | None = None
        self.classes_ = np.array([0, 1])

    def _build_branch_pipeline(self) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=0.1,
                    random_state=self.random_state,
                ),
            ),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LateFusionClassifier:
        meta_features = []

        for modality, feats in self.modality_features.items():
            pipe = self._build_branch_pipeline()
            X_mod = X[feats]
            # Valid mask: rows that are not entirely NaN for this modality
            valid_mask = ~X_mod.isna().all(axis=1)

            if valid_mask.sum() > 0 and y[valid_mask].nunique() > 1:
                pipe.fit(X_mod[valid_mask], y[valid_mask])
                self.branch_pipelines[modality] = pipe

                # Generate probability for meta features
                mod_probs = np.full(len(X), 0.5)
                mod_probs[valid_mask] = pipe.predict_proba(X_mod[valid_mask])[:, 1]
            else:
                mod_probs = np.full(len(X), 0.5)
            meta_features.append(mod_probs)

        if self.strategy == "stacking":
            meta_X = np.column_stack(meta_features)
            self.meta_classifier = LogisticRegression(random_state=self.random_state)
            self.meta_classifier.fit(meta_X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        branch_probs = []
        modality_available_masks = []

        for modality, feats in self.modality_features.items():
            if modality in self.branch_pipelines:
                pipe = self.branch_pipelines[modality]
                X_mod = X[feats]
                # Rows with at least some non-null values
                valid_mask = ~X_mod.isna().all(axis=1)
                mod_prob = np.full(len(X), np.nan)
                if valid_mask.any():
                    mod_prob[valid_mask] = pipe.predict_proba(X_mod[valid_mask])[:, 1]
                branch_probs.append(mod_prob)
                modality_available_masks.append(valid_mask.to_numpy())

        if not branch_probs:
            uniform = np.full((len(X), 2), 0.5)
            return uniform

        probs_matrix = np.column_stack(branch_probs)  # shape (n_samples, n_modalities)

        if self.strategy == "stacking" and self.meta_classifier is not None:
            # Replace NaNs with 0.5 for stacking
            imputed_matrix = np.where(np.isnan(probs_matrix), 0.5, probs_matrix)
            return self.meta_classifier.predict_proba(imputed_matrix)

        # Average over modalities that are actually present for each row
        with np.errstate(all="ignore"):
            mean_probs = np.nanmean(probs_matrix, axis=1)
        mean_probs = np.nan_to_num(mean_probs, nan=0.5)
        mean_probs = np.clip(mean_probs, 0.0, 1.0)
        return np.column_stack((1.0 - mean_probs, mean_probs))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def evaluate_fusion_cv(
    df: pd.DataFrame,
    modality_features: dict[str, list[str]],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run participant-grouped CV comparing single branches, early fusion, and late fusion strategies."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = df["participant_id"]
    y = df["label"].astype(int)
    all_numeric_features = [f for feats in modality_features.values() for f in feats]

    strategies = {
        "Early_Fusion_Logistic": lambda: EarlyFusionClassifier("logistic", random_state=random_state),
        "Early_Fusion_RandomForest": lambda: EarlyFusionClassifier("rf", random_state=random_state),
        "Late_Fusion_Average": lambda: LateFusionClassifier(modality_features, strategy="average", random_state=random_state),
        "Late_Fusion_Stacking": lambda: LateFusionClassifier(modality_features, strategy="stacking", random_state=random_state),
    }

    # Add single-branch baselines
    for modality, feats in modality_features.items():
        strategies[f"Single_Branch_{modality}"] = lambda m_feats=feats: Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, random_state=random_state)),
        ])

    results = {}

    for strat_name, model_fn in strategies.items():
        fold_metrics = []
        oof_probs = np.zeros(len(df))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups)):
            df_tr, df_val = df.iloc[train_idx], df.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            gr_tr, gr_val = groups.iloc[train_idx], groups.iloc[val_idx]

            assert set(gr_tr).isdisjoint(set(gr_val)), f"Leakage detected in fold {fold}"

            model = model_fn()
            if strat_name.startswith("Single_Branch_"):
                mod_name = strat_name.replace("Single_Branch_", "")
                cols = modality_features[mod_name]
                model.fit(df_tr[cols], y_tr)
                probs = model.predict_proba(df_val[cols])[:, 1]
            else:
                model.fit(df_tr[all_numeric_features], y_tr)
                probs = model.predict_proba(df_val[all_numeric_features])[:, 1]

            oof_probs[val_idx] = probs
            preds = (probs >= 0.5).astype(int)

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

        prob_true, prob_pred = calibration_curve(y, oof_probs, n_bins=5, strategy="uniform")
        metric_keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "brier_score"]
        results[strat_name] = {
            m: {
                "mean": float(np.mean([f[m] for f in fold_metrics])),
                "std": float(np.std([f[m] for f in fold_metrics])),
            }
            for m in metric_keys
        }
        results[strat_name]["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        results[strat_name]["oof_brier_score"] = float(brier_score_loss(y, oof_probs))

    return results
