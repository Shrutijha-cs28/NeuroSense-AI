# Antigravity data handoff

## What is in the upload ZIP

`NeuroSenseAI_Antigravity.zip` contains the current source code and three small,
attributed public feature tables:

- UCI DARWIN handwriting features (Alzheimer-related cohort; **not MRI**).
- UCI replicated Parkinson's voice features.
- PhysioNet Gait-in-Parkinson's force-platform features.

It intentionally excludes raw clinical files, downloaded archives, model
checkpoints, `.crdownload` files, and any dataset whose source/license is not
verified. It is a code-and-data starter, not the complete five-modality dataset
collection proposed for the project.

## Candidate source plan

| Branch | Source | Before modeling |
| --- | --- | --- |
| Alzheimer MRI | Exact Kaggle dataset must be selected; a Kaggle search URL is not one dataset. OASIS is another MRI option with data-use terms. | Record the dataset/version/license; inspect subject IDs, label definitions, duplicates and original train/test split. Never report slice-random splitting as subject-independent validation. Do not upload OASIS files to a third-party IDE or GitHub unless its current terms permit it. |
| Alzheimer EEG | [NEMAR on004504](https://www.nemar.org/dataset/on004504), 88 participants, 19-channel EEG; CC0. | Download a small subject subset first; keep all epochs from a participant in one split. Compare raw vs denoised derivatives only after checking their metadata. |
| Parkinson voice | [UCI Parkinsons 174](https://archive.ics.uci.edu/dataset/174/parkinsons), CC BY 4.0. | There are repeated recordings per participant. Use participant-grouped folds; do not randomly split recording rows. |
| Parkinson wearable movement | [PhysioNet PADS](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/), 469 participants; CC BY-NC-SA 4.0. | Download the approved files, retain subject/task IDs, review the official preprocessed files and recommended splits, and report class balance. |
| Parkinson pose/gait | [MMU Visual-Based Parkinson's Dataset](https://www.kaggle.com/dsv/12473275), 167 subjects; CC BY-NC 4.0. | It contains pose keypoints extracted from public walking videos. Verify participant IDs, labels, and split leakage; honor non-commercial terms. |

The MRI and EEG sources, and the different Parkinson datasets, are separate
cohorts. They may support separate modality-specific experiments. They must not
be joined into one multimodal patient record unless the source supplies verified
matching participant/session IDs.

## Antigravity workflow

1. Upload `NeuroSenseAI_Antigravity.zip` (currently under 1 MB).
2. Ask Antigravity to inspect the README, data provenance, current scripts, and
   evaluation code first. Do not ask it to retrain or overwrite models yet.
3. Acquire one candidate at a time from its cited source. Keep raw data local
   under `data/raw/` and ignored by Git. Never rename partial `.crdownload`
   files to pretend they are complete archives.
4. Run an inventory before preprocessing: file integrity, labels, subject IDs,
   class counts, missingness, duplicates, source version, and license.
5. Preprocess per modality and save reproducible scripts/configuration. Keep
   participant IDs for grouping; do imputation/scaling/feature selection inside
   training folds only.
6. Evaluate with participant-grouped folds and report balanced accuracy,
   sensitivity, specificity, F1, AUROC, participant counts, and fold variability.
   Compare against a simple baseline before increasing model complexity.
7. Save only small derived artifacts that the source terms allow. Keep raw
   records and restricted derivative data out of GitHub and third-party IDEs.

## Prompt to paste into Antigravity

> Inspect this NeuroSenseAI package and create a data-readiness plan before
> changing code or training models. Inventory the included datasets and scripts.
> Keep the Alzheimer and Parkinson branches separate. For every proposed new
> source, verify its exact version, license, labels, participant IDs, class
> counts, and allowed storage/sharing. Do not download or upload OASIS or other
> controlled data without confirmed permission. Do not use incomplete downloads.
> Propose reproducible modality-specific preprocessing and participant-grouped
> validation; explain how to prevent participant/window/slice leakage. First
> report gaps and proposed edits for my review; then implement one source at a
> time without claiming clinical diagnostic performance.

## Research-use limitation

This student prototype is not clinically validated and is not a medical device.
Model outputs must not be used for screening, diagnosis, treatment, or care
decisions.
