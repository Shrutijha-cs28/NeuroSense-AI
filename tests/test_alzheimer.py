from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DARWIN_CSV = ROOT / "data" / "external" / "uci_darwin" / "data.csv"
MODEL_PATH = ROOT / "models" / "alzheimer_darwin.joblib"
REPORT_PATH = ROOT / "reports" / "alzheimer_darwin_metrics.json"


def test_darwin_schema():
    assert DARWIN_CSV.exists(), "DARWIN dataset not found"
    df = pd.read_csv(DARWIN_CSV)
    assert len(df) == 174, f"Expected 174 rows, got {len(df)}"
    assert "ID" in df.columns, "Missing 'ID' column"
    assert "class" in df.columns, "Missing 'class' column"
    assert df["ID"].duplicated().sum() == 0, "Duplicate participant IDs found"
    assert set(df["class"].unique()) == {"H", "P"}, "Unexpected class labels"
    assert df.drop(columns=["ID", "class"]).isna().sum().sum() == 0, "Found missing values in feature matrix"
    assert df.shape[1] == 452, f"Expected 452 columns (ID, class, 450 features), got {df.shape[1]}"


def test_no_leakage_in_folds():
    df = pd.read_csv(DARWIN_CSV)
    y = df["class"].map({"H": 0, "P": 1}).to_numpy()
    ids = df["ID"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(df, y):
        train_ids = set(ids[train_idx])
        test_ids = set(ids[test_idx])
        # Assert strictly disjoint participant sets
        assert train_ids.isdisjoint(test_ids), "Participant leakage detected between train and test folds"


def test_model_artifact_and_predict_smoke():
    assert MODEL_PATH.exists(), "Trained model artifact missing"
    artifact = joblib.load(MODEL_PATH)
    assert "pipeline" in artifact, "Missing pipeline in joblib bundle"
    assert "feature_names" in artifact, "Missing feature_names in joblib bundle"
    assert "cv_metrics" in artifact, "Missing cv_metrics in joblib bundle"
    assert "shap_summary" in artifact, "Missing shap_summary in joblib bundle"

    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(DARWIN_CSV)
    sample_X = df[feature_names].iloc[:3]
    probs = pipeline.predict_proba(sample_X)
    assert probs.shape == (3, 2), f"Expected shape (3, 2), got {probs.shape}"
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)


def test_report_schema_and_metrics():
    assert REPORT_PATH.exists(), "Report JSON missing"
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert "dataset" in report
    assert report["dataset"]["participants"] == 174
    assert "models_evaluated" in report
    assert "RandomForest" in report["models_evaluated"]
    assert "Dummy" in report["models_evaluated"]
    # Verify metrics come from CV
    rf_auc = report["models_evaluated"]["RandomForest"]["roc_auc"]["mean"]
    assert 0.80 <= rf_auc <= 0.98
    assert "disclaimer" in report
