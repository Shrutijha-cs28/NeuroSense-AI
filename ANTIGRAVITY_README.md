# NeuroSenseAI — Antigravity handoff package

This ZIP is a compact project starter for Antigravity. It contains the current
application and model source plus the small, attributed public feature tables
already present in the repository. It deliberately does not contain raw MRI,
EEG, video, or other controlled/large datasets.

## Included data

- `data/external/uci_darwin/data.csv` — handwriting features for the Alzheimer-related DARWIN benchmark; it is **not MRI**.
- `data/external/uci_parkinsons/data.csv` — participant-level replicated voice features.
- `data/external/physionet_gaitpdb/gait_features.csv` — derived gait features.
- `data/external/README.md` — source URLs, attribution, and limitations.

These are separate cohorts and modalities. Do not join their rows or claim
cross-modal fusion across participants. Read the provenance and license notes
before redistribution. Use the original sources to retrieve any additional
data; do not commit OASIS, NEMAR or other access-controlled/raw human data to
GitHub or upload it to a third-party workspace unless its terms explicitly
permit that storage and sharing.

## Additional candidate datasets to acquire separately

- Alzheimer EEG: NEMAR `on004504` (88 participants; CC0). Start with a small,
  participant-filtered download and keep participant IDs together in splits.
- Alzheimer MRI: verify the exact Kaggle collection, license, duplicates, and
  participant-level split. The Kaggle search page is not one dataset. OASIS is
  a stronger research source but requires following its data-use terms.
- Parkinson movement: PhysioNet PADS (469 participants, wearable signals;
  CC BY-NC-SA 4.0). Download only after reading its license and cite the source.
- Parkinson pose: MMU visual dataset (167 subjects; CC BY-NC 4.0). Use only for
  non-commercial work under its terms, and verify the supplied split/IDs.

## Run locally

1. Extract this ZIP to a project folder.
2. Create/activate a Python virtual environment.
3. Install `requirements.txt`.
4. Launch the prototype with `streamlit run app.py`.
5. For model experiments, use the training modules under `src/` and preserve
   participant-level splits. Check the source dataset README before training.

This is an educational research prototype, not a medical diagnostic device.
Model outputs must not be used for clinical decisions.
