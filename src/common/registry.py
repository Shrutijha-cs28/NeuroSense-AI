"""Registry for NeuroSenseAI modality branches categorized by disease."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DiseaseType = Literal["alzheimer", "parkinsons"]


@dataclass(frozen=True)
class ModalitySlot:
    id: str
    name: str
    disease: DiseaseType
    status: Literal["active", "pending"]
    source_description: str
    cohort_independent: bool
    notes: str


REGISTRY: dict[DiseaseType, dict[str, ModalitySlot]] = {
    "alzheimer": {
        "handwriting": ModalitySlot(
            id="handwriting",
            name="Online Handwriting Kinematics",
            disease="alzheimer",
            status="active",
            source_description="UCI DARWIN (DOI 10.24432/C55D0K, 174 participants)",
            cohort_independent=True,
            notes="Active tabular benchmark based on pen pressure and velocity features; NOT MRI or EEG.",
        ),
        "mri": ModalitySlot(
            id="mri",
            name="Structural Brain MRI",
            disease="alzheimer",
            status="pending",
            source_description="OASIS / ADNI (requires DUA / application; not checked into repository)",
            cohort_independent=True,
            notes="Pending approved matched participant records. Do not merge with DARWIN.",
        ),
        "eeg": ModalitySlot(
            id="eeg",
            name="Resting-State EEG",
            disease="alzheimer",
            status="pending",
            source_description="NEMAR on004504 (88 participants, CC0)",
            cohort_independent=True,
            notes="Pending preprocessed matched derivative data.",
        ),
    },
    "parkinsons": {
        "voice": ModalitySlot(
            id="voice",
            name="Acoustic Voice Features",
            disease="parkinsons",
            status="active",
            source_description="UCI Parkinson Replicated Acoustic Features (DOI 10.24432/C5701F, 80 participants)",
            cohort_independent=True,
            notes="Active benchmark with 3 recordings per participant; GroupKFold cross-validation.",
        ),
        "gait": ModalitySlot(
            id="gait",
            name="Foot Ground Reaction Force Gait Dynamics",
            disease="parkinsons",
            status="active",
            source_description="PhysioNet Gait in Parkinson's Disease v1.0.0 (ODC-By 1.0, 165 participants)",
            cohort_independent=True,
            notes="Active benchmark with foot vertical force sensors; StratifiedGroupKFold on participant ID.",
        ),
        "tremor": ModalitySlot(
            id="tremor",
            name="Rest and Postural Tremor Accelerometry",
            disease="parkinsons",
            status="pending",
            source_description="PhysioNet PADS / PPMI (smartwatch/wearable)",
            cohort_independent=True,
            notes="Pending intra-cohort matched sensor dataset.",
        ),
        "movement": ModalitySlot(
            id="movement",
            name="Rapid Alternating Movement / Finger Tapping",
            disease="parkinsons",
            status="pending",
            source_description="PhysioNet PADS (smartwatch motor tasks)",
            cohort_independent=True,
            notes="Pending intra-cohort matched smartwatch dataset.",
        ),
    },
}


def get_available_modalities(disease: DiseaseType) -> list[ModalitySlot]:
    """Return all active modalities for a given disease."""
    return [slot for slot in REGISTRY.get(disease, {}).values() if slot.status == "active"]


def get_pending_modalities(disease: DiseaseType) -> list[ModalitySlot]:
    """Return all pending slots for a given disease."""
    return [slot for slot in REGISTRY.get(disease, {}).values() if slot.status == "pending"]


def get_all_modalities(disease: DiseaseType) -> list[ModalitySlot]:
    """Return all registered modality slots for a disease."""
    return list(REGISTRY.get(disease, {}).values())
