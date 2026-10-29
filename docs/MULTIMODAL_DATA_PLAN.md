# NeuroSenseAI Multimodal Data Readiness Plan

> [!WARNING]
> **Research and educational use only.** NeuroSenseAI is not a medical device and is not clinically validated.
> Model outputs must not be used for screening, diagnosis, treatment, or care decisions.

---

## 1. Core Rule for Multimodal Fusion

**Under no circumstances should unrelated public cohorts be merged into pseudo-multimodal records.**
Joining DARWIN (handwriting), UCI replicated (voice), and PhysioNet (gait) by row index, participant order, or guessed identifier is statistically fraudulent and scientifically invalid.

Multimodal fusion algorithms are only legitimate when applied to **verified within-participant matched sessions**, where every modality measurement is collected from the same individual under an approved clinical protocol.

---

## 2. Legitimate Open Candidate Sources for True Fusion

### A. Alzheimer's Disease: NEMAR on004504 (Resting-State EEG)
- **Source:** [NEMAR dataset on004504](https://www.nemar.org/dataset/on004504)
- **Modality:** 19-channel resting-state EEG (eyes-closed)
- **Cohort Size:** 88 participants (Alzheimer's cohort, Mild Cognitive Impairment, and Healthy Controls)
- **License / Terms:** **CC0 (Public Domain Dedication)**
- **ID Structure:** BIDS format (`sub-001`, `sub-002`, ..., with associated `participants.tsv`)
- **Integration Plan:**
  1. Download metadata and preprocessed frequency spectrum derivatives (delta, theta, alpha, beta power).
  2. Partition by participant ID; keep all epochs from a participant within the same cross-validation fold.
  3. Validated within-cohort EEG spectral analysis slot in `src/common/registry.py`.

### B. Parkinson's Disease: PhysioNet PADS (Smartwatch Multi-Task Movement)
- **Source:** [PhysioNet Parkinson's Disease Smartwatch (PADS) v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)
- **Modality:** Multi-task inertial sensor movement (accelerometer and gyroscope during rest tremor, postural tremor, finger tapping, and gait)
- **Cohort Size:** 469 participants
- **License / Terms:** **CC BY-NC-SA 4.0**
- **ID Structure:** Standardized `subject_id` with `task_id` (`rest_tremor`, `postural_tremor`, `finger_tap`, `walking`)
- **Legitimate Fusion Opportunity:**
  Because the **same 469 participants** performed all four sensor tasks, **intra-cohort multimodal fusion across tasks (tremor + tap + gait) is legitimate and scientifically sound**.
- **Integration Plan:**
  1. Download approved summary features.
  2. Map tasks to modality prefixes: `tremor__*`, `movement__*`, `gait__*`.
  3. Validate using `StratifiedGroupKFold` on `subject_id`.

---

## 3. Controlled-Access Clinical Registries (Do NOT Upload to Third-Party Tools)

The following datasets require strict institutional compliance, Data Use Agreements (DUAs), and individual credentialing:

| Registry | Disease Focus | Modalities | Access Conditions | Restriction |
|---|---|---|---|---|
| **OASIS** (Washington University) | Alzheimer's | Longitudinal 3D MRI, PET, clinical scores | Formal application & Data Use Agreement | **Do NOT commit or upload to GitHub, cloud IDEs, or Antigravity scratch.** Keep strictly in private local storage. |
| **ADNI** (LONI / USC) | Alzheimer's | 3T MRI, PET, CSF, cognitive batteries | Scientific review & signed DUA | Redistribution strictly prohibited; credentialed local access only. |
| **PPMI** (Michael J. Fox Foundation) | Parkinson's | DaTscan, SPECT, MRI, CSF, digital sensors | Formal registration, committee review, and strict publication citation rules | Redistribution prohibited; non-commercial research use only. |

---

## 4. Input Schema Specification for Prospective Matched Data

Any future approved dataset ingested by `src/common/train_fusion.py` must conform strictly to the following tabular schema:

```csv
participant_id,label,visit_date,modality1__featA,modality1__featB,modality2__featA,modality2__featB
SUBJ_001,0,2025-01-10,4120.5,2.65,0.22,10.2
SUBJ_002,1,2025-02-11,3180.2,2.12,0.44,8.1
```

- `participant_id` (string): Unique verified subject identifier.
- `label` (int): Binary cohort indicator (`0`: Control, `1`: Cohort).
- `visit_date` (string / ISO): Session or visit timestamp.
- `<modality>__<feature>` (float): Numeric features prefixed by modality and double underscore (`__`). Missing modalities for a session are indicated with `NaN` and handled by the pipeline's imputation and masking layers.
