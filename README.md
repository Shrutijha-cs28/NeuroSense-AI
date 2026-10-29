# NeuroSenseAI

**A student research prototype exploring machine-learning workflows for Alzheimer’s- and Parkinson’s-related digital biomarkers.**

---

### 📌 GitHub Repository Details
- **Description:** *Student research prototype exploring machine-learning workflows for Alzheimer’s- and Parkinson’s-related data with separate handwriting, voice, and gait ML benchmarks.*
- **Suggested Topics:** `neuroinformatics`, `machine-learning`, `alzheimers-research`, `parkinsons-research`, `explainable-ai`, `python`, `data-science`, `cross-validation`, `shap`

---

> [!WARNING]
> **RESEARCH USE ONLY — NOT A MEDICAL DEVICE**  
> NeuroSenseAI is an academic research prototype. It has **not** undergone clinical validation, prospective evaluation, or regulatory clearance.  
> Model outputs, class probabilities, and feature explanations must **never** be used for clinical screening, diagnosis, prognosis, treatment planning, or care decisions.

---

## 📖 About NeuroSenseAI

**NeuroSenseAI** investigates machine learning pipelines applied to non-invasive digital biomarkers associated with neurodegenerative conditions (Alzheimer's and Parkinson's diseases). The platform prioritizes **methodological rigor, transparency, and statistical honesty** over inflated metrics.

### Key Architectural Pillars
1. **Strict Cohort Disjunction:** Publicly available benchmark cohorts (UCI DARWIN handwriting, UCI replicated voice, PhysioNet gaitpdb) originate from **completely independent groups of individuals**. They are evaluated as isolated modality-specific experiments and are **never merged across rows or paired by guessed IDs**.
2. **Participant-Quarantined Cross-Validation:** Because voice and gait datasets include repeated trials and recordings from the same individual, all evaluations use **`GroupKFold` and `StratifiedGroupKFold`** on participant identifiers. No participant ever appears in both training and test folds.
3. **Internal Preprocessing (Zero Data Leakage):** Median imputation, z-score standardization, and univariate feature selection (`SelectKBest`) are executed strictly within training folds.
4. **Nested Cross-Validation for Hyperparameter Tuning:** Outer 5-fold CV evaluates generalizability, while inner 3-fold CV selects hyperparameters, eliminating test-set optimization leakage.
5. **Modality Reality Checks:** The Alzheimer's benchmark is explicitly characterized as **online handwriting kinematics** (pen pressure and velocity across 25 drawing tasks), **strictly NOT brain MRI or EEG analysis**.

---

## 📊 Modality Registry & Project Status

| Area | Modality | Dataset & Provenance | Model & Validation Protocol | Status |
| --- | --- | --- | --- | --- |
| **Alzheimer's** | Online Handwriting Kinematics | [UCI DARWIN (DOI 10.24432/C55D0K)](https://doi.org/10.24432/C55D0K), 174 participants (85 H, 89 P), CC BY 4.0. **Strictly NOT MRI.** | Random Forest, Nested 5-fold Stratified CV with inner hyperparameter tuning. Imputation/scaling in folds. | 🟢 Active Benchmark (85.0% Bal Acc, 94.9% AUROC) |
| **Alzheimer's** | Structural Brain MRI | OASIS / ADNI | Slot pending matched participant records. DUA required; not checked into repository. | 🟡 Pending Matched Data |
| **Alzheimer's** | Resting-State EEG | [NEMAR on004504](https://www.nemar.org/dataset/on004504), 88 participants, CC0 | Slot pending preprocessed spectral derivatives. | 🟡 Pending Matched Data |
| **Parkinson's** | Acoustic Voice Features | [UCI Parkinson Replicated Acoustic Features (DOI 10.24432/C5701F)](https://doi.org/10.24432/C5701F), 80 participants (240 recordings), CC BY 4.0. | Logistic Regression, Participant GroupKFold (5-fold) nested CV. Comparison of pre vs post-aggregation. | 🟢 Active Benchmark (81.3% Bal Acc, 89.2% AUROC) |
| **Parkinson's** | Foot-Force Gait Dynamics | [PhysioNet Gait in Parkinson's Disease (gaitpdb)](https://doi.org/10.13026/C24H3N), 165 participants (306 walks), ODC-By 1.0. | Logistic Regression, StratifiedGroupKFold on participant ID. Sub-study and walk-condition sensitivity audit. | 🟢 Active Benchmark (76.2% Bal Acc, 86.0% AUROC) |
| **Parkinson's** | Tremor / Motor Tasks | PhysioNet PADS (smartwatch, 469 participants) | Slot pending intra-cohort smartwatch sensor feature extraction. | 🟡 Pending Matched Data |
| **Multimodal Fusion** | Matched Modality Fusion Engine | Synthetic matched dataset (`SYNTH_ALZ`, `SYNTH_PD`) | Early feature concatenation vs Late probability voting vs Stacking with missing-modality handling. | 🧪 Proven Plumbing Engine (Real data pending) |

---

## 📈 Benchmark Performance Summary

All metrics are calculated strictly out-of-fold across participant-grouped cross-validation iterations (never from the final training fit):

| Benchmark Branch | Modality | Best Model | Balanced Accuracy | Sensitivity | Specificity | Macro F1 | AUROC | Brier Score | Baseline Acc |
|---|---|---|---|---|---|---|---|---|---|
| **Alzheimer's** | Handwriting Kinematics (UCI DARWIN) | Random Forest | **85.0% ± 6.2%** | 86.5% ± 4.4% | 83.5% ± 11.4% | 85.7% ± 5.4% | **94.9% ± 3.6%** | 0.108 | 51.1% (Dummy) |
| **Parkinson's** | Acoustic Voice (UCI Replicated) | Logistic Regression | **81.3% ± 6.8%** | 82.5% ± 6.1% | 80.0% ± 11.2% | 81.1% ± 6.9% | **89.2% ± 7.2%** | 0.144 | 50.0% (Dummy) |
| **Parkinson's** | Gait Dynamics (PhysioNet Gait) | Logistic Regression | **76.2% ± 6.5%** | 74.3% ± 5.2% | 78.1% ± 12.2% | 77.8% ± 5.0% | **86.0% ± 3.0%** | 0.151 | 56.4% (Dummy) |
| **Multimodal (Plumbing)** | Synthetic Matched (MRI + EEG) | Early Fusion Logistic | 98.8% ± 1.5% | 97.6% ± 3.2% | 100.0% ± 0.0% | 98.8% ± 1.5% | 99.7% ± 0.4% | 0.015 | 50.0% (Dummy) |

*(Note: Synthetic fusion results are generated strictly to prove algorithmic pipeline plumbing and missing-modality handling, not as scientific findings.)*

---

## 🔍 Model Explainability (SHAP) & Audits

- **Alzheimer Handwriting Tasks:** Aggregating SHAP feature importances across writing tasks reveals that **Tasks 17, 15, 6, 23, 9, and 7** provide the highest discriminative signal between patients and controls.
- **Parkinson Voice Acoustics:** SHAP highlights jitter variants, shimmer dB, harmonic-to-noise ratio (HNR), and MFCC delta coefficients as key indicators.
- **Parkinson Gait Sensors:** Vertical ground reaction force summary statistics across foot sensors (specifically `L6`, `L7`, `R8`, and `R4`) show the highest diagnostic attribution.
- **Gait Sub-Study Audit:** Evaluated across three acquisition centers (Ga: AUC 0.852; Ju: AUC 0.865; Si: AUC 0.887) and walking conditions (Usual walk: AUC 0.857; Dual-task: AUC 0.826) to verify stability. Acquisition labels (`study`, `walk_condition`) are strictly excluded from model feature inputs.

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.11+ / 3.13
- Windows PowerShell or Linux/macOS Make

### PowerShell (`run.ps1`)
```powershell
# 1. Set up virtual environment and install dependencies
.\run.ps1 setup

# 2. Run all training pipelines and generate reports
.\run.ps1 train-all

# 3. Execute the full test suite (15 tests: schema, leakage, smoke, fusion)
.\run.ps1 test

# 4. Launch the interactive Streamlit research dashboard
.\run.ps1 app
```

### Linux / macOS (`Makefile`)
```bash
make setup
make train-all
make test
make app
```

---

## 🧪 Automated Testing Suite (`pytest`)

The test suite validates data schemas, guarantees zero participant leakage, and tests prediction pipelines:
- `tests/test_alzheimer.py`:
  - `test_darwin_schema`: Asserts 174 rows, 452 columns, valid class labels (`H`, `P`), zero duplicate IDs, zero missing values.
  - `test_no_leakage_in_folds`: Confirms train and test folds have disjoint participant IDs.
  - `test_model_artifact_and_predict_smoke`: Verifies joblib model bundle and probability predictions.
  - `test_report_schema_and_metrics`: Asserts cross-validation report metrics.
- `tests/test_parkinsons.py`:
  - `test_voice_dataset_schema`: Checks 80 participants, 240 recordings (3 per person).
  - `test_voice_no_participant_leakage`: Confirms `GroupKFold` strictly isolates participant recordings.
  - `test_gait_dataset_schema`: Checks 165 participants, 306 walks.
  - `test_gait_no_participant_leakage`: Confirms `StratifiedGroupKFold` prevents participant leakage.
  - `test_gait_metadata_excluded_from_model`: Asserts `study` and metadata columns are excluded from features.
  - `test_voice_model_predict_smoke` & `test_gait_model_predict_smoke`: Model inference tests.
- `tests/test_fusion.py`:
  - `test_synthetic_datasets_explicitly_labeled`: Checks `data_type` column for synthetic plumbing flag.
  - `test_fusion_no_participant_leakage`: Asserts participant-grouped fold isolation.
  - `test_early_vs_late_fusion_fit_predict`: Tests early concatenation and late probability voting.
  - `test_missing_modality_handling`: Verifies that samples with entirely missing modalities are handled gracefully.

---

## 📁 Repository Structure

```text
NeuroSenseAI/
├── data/
│   ├── external/       # Attributed public benchmark datasets (DARWIN, voice, gait)
│   ├── synthetic/      # Synthetic matched data for fusion plumbing verification
│   ├── raw/            # Downloaded source signals (ignored by Git)
│   └── approved/       # Local-only location for approved matched feature tables
├── docs/
│   ├── MULTIMODAL_DATA_PLAN.md     # Legitimate open data roadmap (NEMAR, PADS, OASIS/PPMI terms)
│   ├── ANTIGRAVITY_DATA_HANDOFF.md # Data handoff notes and source plans
│   └── DATA_AND_PRIVACY.md         # Privacy and data governance guidelines
├── models/             # Serialized Joblib model bundles with metadata & CV metrics
├── reports/            # JSON benchmark metrics, calibration curves, and SHAP summaries
├── scripts/            # Dataset acquisition and synthetic generation utilities
├── src/
│   ├── alzheimer/      # Alzheimer handwriting benchmark training with nested CV & SHAP
│   ├── parkinsons/     # Parkinson voice & gait training with GroupKFold & sensitivity audits
│   └── common/         # Common interface, registry, and multimodal fusion engine
├── tests/              # Pytest test suite covering schema, leakage, and fusion
├── app.py              # Unified Streamlit research dashboard
├── pyproject.toml      # Project configuration
├── requirements.txt    # Production dependencies
├── run.ps1             # PowerShell task automation
└── Makefile            # Make task automation
```

---

## 📜 Licenses & Attribution

- **UCI DARWIN Dataset:** Fontanella et al. (2022), DOI: [10.24432/C55D0K](https://doi.org/10.24432/C55D0K), Licensed under **CC BY 4.0**.
- **UCI Parkinson Replicated Acoustic Features:** Perez (2016), DOI: [10.24432/C5701F](https://doi.org/10.24432/C5701F), Licensed under **CC BY 4.0**.
- **PhysioNet Gait in Parkinson's Disease:** Hausdorff et al., DOI: [10.13026/C24H3N](https://doi.org/10.13026/C24H3N), Licensed under **ODC-By 1.0**.
