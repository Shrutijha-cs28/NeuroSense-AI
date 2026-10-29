from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VOICE_CSV = ROOT / "data" / "external" / "uci_parkinsons" / "data.csv"
GAIT_CSV = ROOT / "data" / "external" / "physionet_gaitpdb" / "gait_features.csv"
VOICE_MODEL = ROOT / "models" / "parkinsons_voice.joblib"
GAIT_MODEL = ROOT / "models" / "parkinsons_gait.joblib"
VOICE_REPORT = ROOT / "reports" / "parkinsons_voice_metrics.json"
GAIT_REPORT = ROOT / "reports" / "parkinsons_gait_metrics.json"


def test_voice_dataset_schema():
    assert VOICE_CSV.exists()
    df = pd.read_csv(VOICE_CSV)
    assert set(["ID", "Recording", "Status"]).issubset(df.columns)
    assert df["ID"].nunique() == 80
    assert len(df) == 240
    # Every participant must have exactly 3 recordings
    assert (df.groupby("ID")["Recording"].count() == 3).all()
    # Labels must be consistent per participant
    assert (df.groupby("ID")["Status"].nunique() == 1).all()


def test_voice_no_participant_leakage():
    df = pd.read_csv(VOICE_CSV)
    gkf = GroupKFold(n_splits=5)
    groups = df["ID"].to_numpy()
    X = df.drop(columns=["ID", "Recording", "Status"])
    y = df["Status"].to_numpy()

    for train_idx, test_idx in gkf.split(X, y, groups):
        train_participants = set(groups[train_idx])
        test_participants = set(groups[test_idx])
        assert train_participants.isdisjoint(test_participants), (
            f"Leakage detected: {train_participants.intersection(test_participants)}"
        )


def test_gait_dataset_schema():
    assert GAIT_CSV.exists()
    df = pd.read_csv(GAIT_CSV)
    assert set(["participant_id", "recording_id", "study", "walk_condition", "label"]).issubset(df.columns)
    assert df["participant_id"].nunique() == 165
    assert len(df) == 306
    # Labels must be consistent per participant
    assert (df.groupby("participant_id")["label"].nunique() == 1).all()


def test_gait_no_participant_leakage():
    df = pd.read_csv(GAIT_CSV)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = df["participant_id"].to_numpy()
    y = df["label"].to_numpy()
    metadata_cols = ["participant_id", "recording_id", "study", "walk_condition", "label", "sampling_hz"]
    X = df.drop(columns=[c for c in metadata_cols if c in df.columns])

    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_participants = set(groups[train_idx])
        test_participants = set(groups[test_idx])
        assert train_participants.isdisjoint(test_participants), (
            f"Gait participant leakage detected: {train_participants.intersection(test_participants)}"
        )


def test_gait_metadata_excluded_from_model():
    artifact = joblib.load(GAIT_MODEL)
    feature_names = artifact["feature_names"]
    forbidden = ["participant_id", "recording_id", "study", "walk_condition", "label", "sampling_hz"]
    for f in forbidden:
        assert f not in feature_names, f"Metadata column '{f}' found in gait model features!"


def test_voice_model_predict_smoke():
    assert VOICE_MODEL.exists()
    artifact = joblib.load(VOICE_MODEL)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(VOICE_CSV)
    sample = df[feature_names].iloc[:3]
    probs = pipeline.predict_proba(sample)
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)


def test_gait_model_predict_smoke():
    assert GAIT_MODEL.exists()
    artifact = joblib.load(GAIT_MODEL)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(GAIT_CSV)
    sample = df[feature_names].iloc[:3]
    probs = pipeline.predict_proba(sample)
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)
