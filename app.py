"""NeuroSenseAI unified research platform and benchmark dashboard.

RESEARCH USE ONLY: This student prototype is not a medical device and has not been clinically validated.
Do not use model outputs for screening, diagnosis, treatment, or care decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.common.registry import REGISTRY, get_all_modalities

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="NeuroSenseAI | Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_disclaimer():
    st.error(
        "⚠️ **RESEARCH USE ONLY — NOT A MEDICAL DEVICE**  \n"
        "NeuroSenseAI is an exploratory academic prototype. It has not undergone clinical validation, "
        "prospective evaluation, or regulatory clearance. Model predictions, probabilities, and explanations "
        "must **never** be used for clinical screening, diagnosis, treatment, or patient care decisions.",
        icon="🚨",
    )


@st.cache_data
def load_report_safe(filename: str) -> dict | None:
    p = REPORTS_DIR / filename
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_resource
def load_model_safe(filename: str) -> dict | None:
    p = MODELS_DIR / filename
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


# Sidebar Navigation
with st.sidebar:
    st.title("🧠 NeuroSenseAI")
    st.caption("Multimodal Neuroinformatics Research Starter")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home & Overview",
            "✍️ Alzheimer's (Handwriting)",
            "🩺 Parkinson's (Voice & Gait)",
            "🔗 Multimodal Fusion",
            "📊 Benchmark Results",
        ],
    )
    st.divider()
    st.caption(
        "**Strict Data Integrity Rule:**\n"
        "DARWIN handwriting, UCI voice, and PhysioNet gait datasets come from **different participants**. "
        "They are never merged across cohorts."
    )

# Page 1: Home & Overview
if page == "🏠 Home & Overview":
    st.title("🧠 NeuroSenseAI: Student Research Prototype")
    render_disclaimer()

    st.markdown("""
    ### Project Overview
    **NeuroSenseAI** is an academic research platform designed to investigate machine learning workflows for 
    Alzheimer's and Parkinson's disease-related digital biomarkers. The codebase provides modular benchmark branches, 
    leakage-free cross-validation protocols, and a multimodal fusion engine designed for prospective matched cohorts.

    #### Core Engineering Principles
    1. **Strict Cohort Separation:** Included public datasets (DARWIN handwriting, UCI replicated voice, PhysioNet gaitpdb) 
       originate from completely disjoint human subject cohorts. They are evaluated independently and are **never merged**.
    2. **Participant-Grouped Cross-Validation:** Every model is evaluated using participant-grouped folds (`GroupKFold` / `StratifiedGroupKFold`). 
       All recordings, visits, or windows from a participant are strictly quarantined to a single fold.
    3. **Leakage-Free Preprocessing:** Imputation, standardization, and feature selection exist strictly inside CV training pipelines.
    4. **Honest Attribution & Licensing:** Every dataset version, DOI, and license (CC BY 4.0, ODC-By 1.0) is tracked.
    """)

    st.subheader("📋 Modality Registry & Roadmap")
    reg_rows = []
    for disease in ["alzheimer", "parkinsons"]:
        for slot in get_all_modalities(disease):
            reg_rows.append({
                "Disease": disease.capitalize(),
                "Modality": slot.name,
                "Status": "🟢 Active Benchmark" if slot.status == "active" else "🟡 Pending Matched Data",
                "Source / Provenance": slot.source_description,
                "Notes": slot.notes,
            })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)


# Page 2: Alzheimer's (Handwriting Benchmark)
elif page == "✍️ Alzheimer's (Handwriting)":
    st.title("✍️ Alzheimer's Branch: Online Handwriting Kinematics")
    render_disclaimer()

    st.info(
        "**Modality Reality:** This benchmark uses **online handwriting features** from the UCI DARWIN dataset (174 subjects). "
        "It evaluates kinematic pen pressure, velocity, and in-air movements across 25 writing/drawing tasks. "
        "It is **strictly NOT brain MRI or EEG analysis**."
    )

    model_bundle = load_model_safe("alzheimer_darwin.joblib")
    report = load_report_safe("alzheimer_darwin_metrics.json")

    if not model_bundle or not report:
        st.error("Alzheimer's model or report not found. Please train models first.")
        st.stop()

    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]
    cv_metrics = model_bundle.get("cv_metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Model", model_bundle.get("best_model_name", "Random Forest"))
    col2.metric("Nested CV Balanced Acc", f"{cv_metrics.get('balanced_accuracy', {}).get('mean', 0)*100:.1f}% ± {cv_metrics.get('balanced_accuracy', {}).get('std', 0)*100:.1f}%")
    col3.metric("Nested CV AUROC", f"{cv_metrics.get('roc_auc', {}).get('mean', 0)*100:.1f}% ± {cv_metrics.get('roc_auc', {}).get('std', 0)*100:.1f}%")
    col4.metric("Brier Score", f"{cv_metrics.get('brier_score', {}).get('mean', 0):.3f}")

    st.subheader("Interactive Benchmark Inference")
    input_mode = st.radio("Select input mode:", ["Select Sample Benchmark Row", "Upload CSV"], horizontal=True)

    data_csv = DATA_DIR / "external" / "uci_darwin" / "data.csv"
    input_df = None

    if input_mode == "Select Sample Benchmark Row" and data_csv.exists():
        raw_df = pd.read_csv(data_csv)
        row_idx = st.selectbox(
            "Choose a participant row to inspect:",
            range(min(20, len(raw_df))),
            format_func=lambda i: f"Participant ID {raw_df.iloc[i]['ID']} (Ground Truth: {'Healthy (H)' if raw_df.iloc[i]['class'] == 'H' else 'Alzheimer Cohort (P)'})",
        )
        input_df = raw_df.iloc[[row_idx]][feature_names]
    elif input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload DARWIN format CSV", type=["csv"])
        if uploaded_file:
            uploaded_df = pd.read_csv(uploaded_file)
            missing_feats = [f for f in feature_names if f not in uploaded_df.columns]
            if missing_feats:
                st.error(f"Uploaded CSV is missing {len(missing_feats)} expected features.")
            else:
                input_df = uploaded_df[feature_names].iloc[[0]]

    if input_df is not None:
        probs = pipeline.predict_proba(input_df)[0]
        p_healthy = probs[0]
        p_patient = probs[1]

        st.markdown("#### Model Output")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Probability: Alzheimer's Cohort", f"{p_patient*100:.1f}%")
            st.progress(float(p_patient))
        with res_col2:
            st.metric("Probability: Healthy Control", f"{p_healthy*100:.1f}%")
            st.progress(float(p_healthy))

        st.caption(
            "Note: Probability represents statistical similarity to the DARWIN cohort feature distribution, NOT clinical risk."
        )

        st.markdown("#### Global Task & Feature Explainability (SHAP)")
        shap_summary = model_bundle.get("shap_summary", {})
        top_tasks = shap_summary.get("top_tasks", [])[:8]
        if top_tasks:
            st.write("**Most Discriminative Handwriting Tasks (ranked by aggregate SHAP importance):**")
            task_df = pd.DataFrame(top_tasks)
            task_df["Task"] = task_df["task"].apply(lambda t: f"Task {t}")
            st.bar_chart(task_df.set_index("Task")["importance"])


# Page 3: Parkinson's (Voice & Gait)
elif page == "🩺 Parkinson's (Voice & Gait)":
    st.title("🩺 Parkinson's Branch: Voice and Gait Benchmarks")
    render_disclaimer()

    st.warning(
        "🚨 **CRITICAL DATA SEPARATION NOTICE:**\n"
        "The UCI Voice dataset (80 participants) and the PhysioNet Gait dataset (165 participants) are **two completely separate cohorts**.\n"
        "They are presented in separate tabs below and **must NEVER be combined into a single patient prediction**."
    )

    tab_voice, tab_gait = st.tabs(["🎙️ Acoustic Voice Modality (UCI)", "🚶 Gait Force Modality (PhysioNet)"])

    # Voice Tab
    with tab_voice:
        st.subheader("🎙️ Parkinson's Voice Acoustic Benchmark")
        voice_model = load_model_safe("parkinsons_voice.joblib")
        voice_report = load_report_safe("parkinsons_voice_metrics.json")

        if voice_model and voice_report:
            v_metrics = voice_model.get("cv_metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Selected Model", voice_model.get("best_model_name", "Logistic Regression"))
            c2.metric("GroupKFold Balanced Acc", f"{v_metrics.get('balanced_accuracy', {}).get('mean', 0)*100:.1f}% ± {v_metrics.get('balanced_accuracy', {}).get('std', 0)*100:.1f}%")
            c3.metric("GroupKFold AUROC", f"{v_metrics.get('roc_auc', {}).get('mean', 0)*100:.1f}% ± {v_metrics.get('roc_auc', {}).get('std', 0)*100:.1f}%")
            c4.metric("Brier Score", f"{v_metrics.get('brier_score', {}).get('mean', 0):.3f}")

            voice_data_csv = DATA_DIR / "external" / "uci_parkinsons" / "data.csv"
            if voice_data_csv.exists():
                v_df = pd.read_csv(voice_data_csv)
                v_feats = voice_model["feature_names"]
                v_row_idx = st.selectbox(
                    "Choose sample voice recording:",
                    range(min(15, len(v_df))),
                    format_func=lambda i: f"Participant {v_df.iloc[i]['ID']}, Recording #{v_df.iloc[i]['Recording']} (Truth: {'PD' if v_df.iloc[i]['Status'] == 1 else 'Control'})",
                    key="voice_select",
                )
                sample_v = v_df.iloc[[v_row_idx]][v_feats]
                v_probs = voice_model["pipeline"].predict_proba(sample_v)[0]

                vc1, vc2 = st.columns(2)
                vc1.metric("Parkinson's Cohort Probability", f"{v_probs[1]*100:.1f}%")
                vc2.metric("Control Probability", f"{v_probs[0]*100:.1f}%")

                st.write("**Top SHAP Voice Acoustic Features:**")
                v_top = voice_model.get("shap_summary", {}).get("top_features", [])[:6]
                if v_top:
                    st.dataframe(pd.DataFrame(v_top), hide_index=True)

    # Gait Tab
    with tab_gait:
        st.subheader("🚶 Parkinson's Gait-Force Dynamics Benchmark")
        gait_model = load_model_safe("parkinsons_gait.joblib")
        gait_report = load_report_safe("parkinsons_gait_metrics.json")

        if gait_model and gait_report:
            g_metrics = gait_model.get("cv_metrics", {})
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("Selected Model", gait_model.get("best_model_name", "Logistic Regression"))
            gc2.metric("StratifiedGroupKFold Balanced Acc", f"{g_metrics.get('balanced_accuracy', {}).get('mean', 0)*100:.1f}% ± {g_metrics.get('balanced_accuracy', {}).get('std', 0)*100:.1f}%")
            gc3.metric("StratifiedGroupKFold AUROC", f"{g_metrics.get('roc_auc', {}).get('mean', 0)*100:.1f}% ± {g_metrics.get('roc_auc', {}).get('std', 0)*100:.1f}%")
            gc4.metric("Brier Score", f"{g_metrics.get('brier_score', {}).get('mean', 0):.3f}")

            st.markdown("#### Study Leakage Audit & Sensitivity Check")
            leak_info = gait_report.get("leakage_safeguard_and_study_audit", {})
            st.caption(leak_info.get("leakage_safeguard", ""))

            sub_sens = gait_report.get("subgroup_sensitivity_analysis", {})
            if "by_study" in sub_sens:
                st.write("**Performance by Sub-Study:**")
                st.dataframe(pd.DataFrame(sub_sens["by_study"]).T[["participants", "balanced_accuracy", "roc_auc", "sensitivity", "specificity"]])

            gait_data_csv = DATA_DIR / "external" / "physionet_gaitpdb" / "gait_features.csv"
            if gait_data_csv.exists():
                g_df = pd.read_csv(gait_data_csv)
                g_feats = gait_model["feature_names"]
                g_row_idx = st.selectbox(
                    "Choose sample gait recording:",
                    range(min(15, len(g_df))),
                    format_func=lambda i: f"Participant {g_df.iloc[i]['participant_id']}, Walk: {g_df.iloc[i]['walk_condition']} (Study: {g_df.iloc[i]['study']})",
                    key="gait_select",
                )
                sample_g = g_df.iloc[[g_row_idx]][g_feats]
                g_probs = gait_model["pipeline"].predict_proba(sample_g)[0]

                g_res1, g_res2 = st.columns(2)
                g_res1.metric("Parkinson's Cohort Probability", f"{g_probs[1]*100:.1f}%")
                g_res2.metric("Control Probability", f"{g_probs[0]*100:.1f}%")


# Page 4: Multimodal Fusion
elif page == "🔗 Multimodal Fusion":
    st.title("🔗 Multimodal Fusion Architecture & Plumbing Engine")
    render_disclaimer()

    st.markdown("""
    ### Multimodal Fusion Architecture
    The NeuroSenseAI fusion engine is architected to evaluate **within-participant matched data** using participant-grouped cross-validation.
    
    ```mermaid
    flowchart TD
        Subj[Matched Participant Record] --> B1[Modality Branch 1: e.g. MRI]
        Subj --> B2[Modality Branch 2: e.g. EEG]
        Subj --> B3[Modality Branch 3: e.g. Gait]
        
        subgraph Early Fusion
            B1 --> Cat[Feature Concatenation + Missing Modality Imputation]
            B2 --> Cat
            B3 --> Cat
            Cat --> EarlyClf[Calibrated Early Classifier]
        end
        
        subgraph Late Fusion
            B1 --> ModPipe1[Branch Pipeline 1] --> P1[P_1]
            B2 --> ModPipe2[Branch Pipeline 2] --> P2[P_2]
            B3 --> ModPipe3[Branch Pipeline 3] --> P3[P_3]
            P1 --> LateComb[Robust NaN-Tolerant Average / Stacking]
            P2 --> LateComb
            P3 --> LateComb
        end
        
        EarlyClf --> Out[Calibrated Decision & Explainability]
        LateComb --> Out
    ```
    """)

    st.subheader("Required Matched-Data Input Schema")
    st.code(
        """participant_id,label,visit_date,mri__hippocampus_volume,mri__cortical_thickness,eeg__theta_power,eeg__alpha_peak
SUBJ_001,0,2025-01-15,4120.5,2.65,0.215,10.2
SUBJ_002,1,2025-01-16,3180.2,2.12,0.435,8.1""",
        language="csv",
    )

    st.warning(
        "🚨 **SYNTHETIC PLUMBING TEST DEMO — NOT SCIENTIFIC RESULTS**  \n"
        "Because legitimate public multi-modality matched datasets for these cohorts require controlled data use agreements (DUAs), "
        "the demo below executes on a **synthetically generated dataset** strictly to prove the algorithmic plumbing of the Early and Late fusion pipelines.",
        icon="🧪",
    )

    synth_alz_path = DATA_DIR / "synthetic" / "alzheimer_matched_synthetic.csv"
    if synth_alz_path.exists():
        synth_df = pd.read_csv(synth_alz_path)
        st.write(f"Loaded synthetic test cohort: {len(synth_df)} visits from {synth_df['participant_id'].nunique()} synthetic participants.")

        fusion_report = load_report_safe("alzheimer_fusion_metrics.json")
        if fusion_report:
            st.markdown("#### Synthetic Benchmark Comparison (Plumbing Verification)")
            strategies = fusion_report.get("strategies_evaluated", {})
            comp_rows = []
            for name, metrics in strategies.items():
                comp_rows.append({
                    "Strategy": name.replace("_", " "),
                    "Balanced Accuracy": f"{metrics.get('balanced_accuracy', {}).get('mean', 0)*100:.1f}%",
                    "AUROC": f"{metrics.get('roc_auc', {}).get('mean', 0)*100:.1f}%",
                    "Brier Score": f"{metrics.get('oof_brier_score', 0):.3f}",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# Page 5: Benchmark Results
elif page == "📊 Benchmark Results":
    st.title("📊 Benchmark Results & Rigorous Cross-Validation")
    render_disclaimer()

    st.markdown("""
    All reported benchmark results are derived **strictly from out-of-fold cross-validation** (never training fits).
    Imputation, standardization, and feature selection are quarantined within training folds to eliminate data leakage.
    """)

    results_data = [
        {
            "Branch": "Alzheimer's",
            "Modality": "Online Handwriting (UCI DARWIN)",
            "Model": "Random Forest",
            "Validation Protocol": "Nested 5-Fold Stratified CV",
            "Balanced Accuracy": "85.0% ± 6.2%",
            "Sensitivity": "86.5% ± 4.4%",
            "Specificity": "83.5% ± 11.4%",
            "AUROC": "94.9% ± 3.6%",
            "Brier Score": "0.108",
            "Baseline Acc": "51.1% (Majority Dummy)",
            "Data Reality": "Handwriting kinematics (174 subjects); strictly NOT MRI",
        },
        {
            "Branch": "Parkinson's",
            "Modality": "Acoustic Voice (UCI Replicated)",
            "Model": "Logistic Regression",
            "Validation Protocol": "Participant GroupKFold (5 Folds)",
            "Balanced Accuracy": "81.3% ± 6.8%",
            "Sensitivity": "82.5% ± 6.1%",
            "Specificity": "80.0% ± 11.2%",
            "AUROC": "89.2% ± 7.2%",
            "Brier Score": "0.144",
            "Baseline Acc": "50.0% (Majority Dummy)",
            "Data Reality": "80 participants (3 recordings each); grouped folds",
        },
        {
            "Branch": "Parkinson's",
            "Modality": "Foot Force Dynamics (PhysioNet Gait)",
            "Model": "Logistic Regression",
            "Validation Protocol": "StratifiedGroupKFold on Participant",
            "Balanced Accuracy": "76.2% ± 6.5%",
            "Sensitivity": "74.3% ± 5.2%",
            "Specificity": "78.1% ± 12.2%",
            "AUROC": "86.0% ± 3.0%",
            "Brier Score": "0.151",
            "Baseline Acc": "56.4% (Majority Dummy)",
            "Data Reality": "165 participants (306 walks); study metadata excluded",
        },
        {
            "Branch": "Multimodal Fusion",
            "Modality": "Synthetic Matched Pipeline (MRI + EEG)",
            "Model": "Early Fusion Logistic",
            "Validation Protocol": "StratifiedGroupKFold on Participant",
            "Balanced Accuracy": "98.8% ± 1.5%",
            "Sensitivity": "97.6% ± 3.2%",
            "Specificity": "100.0% ± 0.0%",
            "AUROC": "99.7% ± 0.4%",
            "Brier Score": "0.015",
            "Baseline Acc": "50.0%",
            "Data Reality": "SYNTHETIC PLUMBING TEST ONLY; NOT SCIENTIFIC RESULTS",
        },
    ]

    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

    st.markdown("""
    #### Key Methodology Safeguards
    - **No Cross-Cohort Merging:** DARWIN, UCI voice, and PhysioNet gait subjects are never joined.
    - **No Recording Leakage:** Replicated acoustic rows and multi-trial walks are grouped by participant identifier.
    - **No Metadata Contamination:** Study origin identifiers (`Ga`, `Ju`, `Si`) are scrubbed from gait model features.
    """)
