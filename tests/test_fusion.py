from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src.common.fusion import EarlyFusionClassifier, LateFusionClassifier, evaluate_fusion_cv

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYNTH_ALZ_CSV = ROOT / "data" / "synthetic" / "alzheimer_matched_synthetic.csv"
SYNTH_PD_CSV = ROOT / "data" / "synthetic" / "parkinsons_matched_synthetic.csv"


def test_synthetic_datasets_explicitly_labeled():
    assert SYNTH_ALZ_CSV.exists()
    assert SYNTH_PD_CSV.exists()

    df_alz = pd.read_csv(SYNTH_ALZ_CSV)
    df_pd = pd.read_csv(SYNTH_PD_CSV)

    assert "data_type" in df_alz.columns
    assert (df_alz["data_type"] == "SYNTHETIC_PLUMBING_TEST_NOT_A_RESULT").all()
    assert (df_pd["data_type"] == "SYNTHETIC_PLUMBING_TEST_NOT_A_RESULT").all()


def test_fusion_no_participant_leakage():
    df = pd.read_csv(SYNTH_ALZ_CSV)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = df["participant_id"].to_numpy()
    y = df["label"].to_numpy()

    for train_idx, test_idx in sgkf.split(df, y, groups):
        train_p = set(groups[train_idx])
        test_p = set(groups[test_idx])
        assert train_p.isdisjoint(test_p), f"Participant leakage in fusion split: {train_p.intersection(test_p)}"


def test_early_vs_late_fusion_fit_predict():
    df = pd.read_csv(SYNTH_ALZ_CSV)
    modality_features = {
        "mri": [c for c in df.columns if c.startswith("mri__")],
        "eeg": [c for c in df.columns if c.startswith("eeg__")],
    }
    all_features = modality_features["mri"] + modality_features["eeg"]
    y = df["label"].astype(int)

    # Early Fusion
    early = EarlyFusionClassifier("logistic", random_state=42)
    early.fit(df[all_features], y)
    probs_early = early.predict_proba(df[all_features])
    assert probs_early.shape == (len(df), 2)
    np.testing.assert_allclose(probs_early.sum(axis=1), np.ones(len(df)), atol=1e-5)

    # Late Fusion
    late = LateFusionClassifier(modality_features, strategy="average", random_state=42)
    late.fit(df[all_features], y)
    probs_late = late.predict_proba(df[all_features])
    assert probs_late.shape == (len(df), 2)
    np.testing.assert_allclose(probs_late.sum(axis=1), np.ones(len(df)), atol=1e-5)


def test_missing_modality_handling():
    """Verify that late fusion and early fusion gracefully handle missing modalities."""
    df = pd.read_csv(SYNTH_ALZ_CSV).copy()
    modality_features = {
        "mri": [c for c in df.columns if c.startswith("mri__")],
        "eeg": [c for c in df.columns if c.startswith("eeg__")],
    }
    all_features = modality_features["mri"] + modality_features["eeg"]
    y = df["label"].astype(int)

    late = LateFusionClassifier(modality_features, strategy="average", random_state=42)
    late.fit(df[all_features], y)

    # Test sample with entirely missing MRI features
    test_sample = df[all_features].iloc[:2].copy()
    test_sample[modality_features["mri"]] = np.nan

    probs = late.predict_proba(test_sample)
    assert not np.isnan(probs).any(), "Found NaNs in late fusion predictions with missing modality"
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-5)

    # Early fusion with missing features
    early = EarlyFusionClassifier("logistic", random_state=42)
    early.fit(df[all_features], y)
    probs_early = early.predict_proba(test_sample)
    assert not np.isnan(probs_early).any()
    assert probs_early.shape == (2, 2)
    np.testing.assert_allclose(probs_early.sum(axis=1), np.ones(2), atol=1e-5)
