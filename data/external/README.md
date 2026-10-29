# External verified datasets and attribution

These CSVs were downloaded from the official [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Both pages label the files **CC BY 4.0**. Keep this attribution with any redistribution and check the linked dataset record for updated terms and citations.

| Local CSV | Official source | Contents | Attribution |
| --- | --- | --- | --- |
| `uci_darwin/data.csv` | [DARWIN, UCI dataset 732](https://archive.ics.uci.edu/dataset/732/darwin), [DOI 10.24432/C55D0K](https://doi.org/10.24432/C55D0K) | 174 participants; 451 online-handwriting features; 89 `P` and 85 `H` labels. This supports an Alzheimer cohort versus healthy cohort **handwriting-feature** experiment, not MRI classification. | Fontanella et al. (2022), UCI DARWIN; see the UCI page for full citation and paper references. |
| `uci_parkinsons/data.csv` | [Parkinson dataset with replicated acoustic features, UCI dataset 489](https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features), [DOI 10.24432/C5701F](https://doi.org/10.24432/C5701F) | 240 acoustic-feature rows from 80 participants (40 Parkinson's, 40 control), three recordings each. The model averages each participant's recordings before splitting. | Prez (2016), UCI Parkinson Dataset with Replicated Acoustic Features; see the UCI page for its cited study. |
| `physionet_gaitpdb/gait_features.csv` | [Gait in Parkinson's Disease, PhysioNet v1.0.0](https://physionet.org/content/gaitpdb/1.0.0/), [DOI 10.13026/C24H3N](https://doi.org/10.13026/C24H3N) | Derived from 306 force recordings; 165 represented participants (72 control, 93 Parkinson's). Each walk has 180 summary features from 16 foot sensors and two total-force channels. Raw samples are 100 Hz. | Hausdorff et al., Gait in Parkinson's Disease, PhysioNet. Licensed under Open Data Commons Attribution License v1.0 (ODC-By). See the source page for study citation details. |

The training reports include a participant-level cross-validation summary and a majority-class baseline. Gait validation holds out participants by fold and averages a participant's walk probabilities before calculating metrics. Results are exploratory internal validation, not independent clinical validation. These sources are small benchmark cohorts and must not be described as sufficient for health screening, diagnosis, or treatment decisions.

The gait feature CSV is a derived feature table produced by `scripts/preprocess_physionet_gait.py`. The download utility retrieves the source signal text files into `data/raw/physionet_gaitpdb/`, which is ignored by Git. The derived CSV is tracked with this source attribution and contains participant-group identifiers; share it only under the ODC-By license and project attribution requirements. Raw recording signals are not committed.

## Download again

Run `python scripts/download_verified_datasets.py` from the repository root to fetch the UCI files. Use `--force` to replace those CSV files. To reproduce the gait feature table, run `python scripts/download_physionet_gait.py` followed by `python scripts/preprocess_physionet_gait.py`.

## More clinically representative cohorts (application required)

- [OASIS at Washington University in St. Louis](https://sites.wustl.edu/oasisbrains/request-access/) provides Alzheimer-related MRI and clinical/cognitive measurements. Users must request access and follow OASIS's data-use terms. Do not commit OASIS files or derivatives unless the agreement explicitly permits that use and redistribution.
- [ADNI data access](https://adni.loni.usc.edu/data-samples/adni-data/) requires an application and acceptance of its Data Use Agreement.
- [PPMI Parkinson's access](https://www.ppmi-info.org/access-data-specimens/data) requires registration, a Data Use Agreement, application review, and compliance with its publication policy.

For approved OASIS/ADNI/PPMI data, keep raw data in a private, ignored local directory and use participant IDs for leakage-safe evaluation. This repository's UCI pipelines are not substitutes for those access-controlled cohorts or for prospective external validation.
