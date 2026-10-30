"""
Streamlit Web Application: Generative AI-Based Personalized Feedback System
for Academic Staff Stress-Level Assessment (Akwa Ibom State University)

Author: OKON, GOODNEWS ASUQUO (AK22/PHS/CSC/078)
        Department of Computer Science, Akwa Ibom State University
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from stress_system_core import (
    ALL_23_FEATURE_COLS,
    EXCEL_PATH,
    FEATURE_DICTIONARY,
    FEATURE_DOMAINS,
    FEATURE_LABELS,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PEARSON_15_FEATURES,
    TABLES_DIR,
    VALUE_DECODERS,
    decode_feature_value,
    decode_staff_responses,
    encode_feature_value,
    encode_staff_responses,
    explain_individual_staff,
    generate_personalized_feedback,
    load_all_project_sheets,
)

# Resolve dataset path with fallback
DATASET_PATH = TABLES_DIR / "academic_staff_stress_dataset.csv"
if not DATASET_PATH.exists():
    _fallback_csv = TABLES_DIR / "academic_staff_stress_augmented_110.csv"
    if _fallback_csv.exists():
        DATASET_PATH = _fallback_csv

st.set_page_config(
    page_title="STRESS ASSESSMENT TEST PAGE",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_bundle_and_data():
    bundle_path = MODELS_DIR / "stress_assessment_bundle.joblib"
    if not bundle_path.exists():
        st.error(f"Model bundle not found at {bundle_path}. Please run train_and_build_pipeline.py first.")
        st.stop()
    bundle = joblib.load(bundle_path)

    if DATASET_PATH.exists():
        df_cohort = pd.read_csv(DATASET_PATH)
    else:
        sheets_temp = load_all_project_sheets(EXCEL_PATH)
        df_cohort = sheets_temp["sheet3_digitized"]

    return bundle, df_cohort


def main():
    bundle, df_cohort = load_bundle_and_data()

    # -----------------------------------------------------------------------
    # SIDEBAR: MODEL & GENERATIVE AI SETTINGS
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.subheader("Model Selection")
        model_names = list(bundle["trained_models"].keys())
        default_model = bundle.get("best_model_name", "Ridge Logistic Regression")
        model_choice = st.selectbox(
            "Classification Model:",
            options=model_names,
            index=model_names.index(default_model) if default_model in model_names else 0,
        )
        active_pipeline = bundle["trained_models"][model_choice]

        st.divider()
        st.subheader("Generative AI Configuration")
        api_key_input = st.text_input(
            "Gemini API Key (Optional):",
            type="password",
            placeholder="AIzaSy...",
            help="Enter key for live Gemini 3.8 Flash generation. Otherwise, the local generative synthesis engine is used.",
        )
        use_live = st.toggle("Enable Live Gemini API", value=bool(api_key_input or os.environ.get("GEMINI_API_KEY")))

    # -----------------------------------------------------------------------
    # TOP BAR: STRESS ASSESSMENT TEST PAGE
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%); padding: 14px 24px; border-radius: 8px; margin-bottom: 24px;">
            <h1 style="color: white; margin: 0; text-align: center; letter-spacing: 1.5px; font-weight: 800; font-size: 26px;">
                STRESS ASSESSMENT TEST PAGE
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_ref, c_spacer = st.columns([1.5, 2.5])
    with c_ref:
        test_staff_id = st.text_input("Staff Reference Code / ID:", value="TEST_STAFF_01")

    custom_vals = {}

    # Category 1: Socio-Demographic & Academic Profile
    with st.expander("🎓 Socio-Demographic & Academic Profile", expanded=True):
        dom1_cols = ["academic_rank", "educational_qualification", "teaching_experience", "age", "gender"]
        c1, c2 = st.columns(2)
        for idx_c, f_key in enumerate(dom1_cols):
            f_info = FEATURE_DICTIONARY[f_key]
            with (c1 if idx_c % 2 == 0 else c2):
                chosen_opt = st.selectbox(
                    f_info["label"],
                    options=f_info["options"],
                    index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                    help=f_info["question_text"],
                    key=f"new_{f_key}",
                )
                custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

    # Category 2: Occupational & Institutional Stressors
    with st.expander("📚 Occupational & Institutional Stressors", expanded=True):
        dom2_cols = [
            "teaching_workload", "research_responsibilities",
            "administrative_responsibilities", "exam_supervision_marking",
            "teaching_hours"
        ]
        c3, c4 = st.columns(2)
        for idx_c, f_key in enumerate(dom2_cols):
            f_info = FEATURE_DICTIONARY[f_key]
            with (c3 if idx_c % 2 == 0 else c4):
                chosen_opt = st.selectbox(
                    f_info["label"],
                    options=f_info["options"],
                    index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                    help=f_info["question_text"],
                    key=f"new_{f_key}",
                )
                custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

    # Category 3: Lifestyle, Health & Coping Mechanisms
    with st.expander("🏃 Lifestyle, Health & Coping Mechanisms", expanded=False):
        dom3_cols = [
            "physical_exercise", "physical_exercise_frequency",
            "break_frequency", "health_problems_affect_coping",
            "smoking", "alcohol_consumption", "alcohol_frequency"
        ]
        c5, c6 = st.columns(2)
        for idx_c, f_key in enumerate(dom3_cols):
            f_info = FEATURE_DICTIONARY[f_key]
            with (c5 if idx_c % 2 == 0 else c6):
                chosen_opt = st.selectbox(
                    f_info["label"],
                    options=f_info["options"],
                    index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                    help=f_info["question_text"],
                    key=f"new_{f_key}",
                )
                custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

    # Category 4: Generative AI Adoption & Feedback Preferences
    with st.expander("🤖 Generative AI Adoption & Feedback Preferences", expanded=False):
        dom4_cols = [
            "genai_tool_frequency", "genai_research_teaching_usefulness",
            "genai_admin_use", "genai_system_trust",
            "feedback_preference", "feedback_frequency"
        ]
        c7, c8 = st.columns(2)
        for idx_c, f_key in enumerate(dom4_cols):
            f_info = FEATURE_DICTIONARY[f_key]
            with (c7 if idx_c % 2 == 0 else c8):
                chosen_opt = st.selectbox(
                    f_info["label"],
                    options=f_info["options"],
                    index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                    help=f_info["question_text"],
                    key=f"new_{f_key}",
                )
                custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

    # Build complete test respondent series
    custom_vals["staff_code"] = test_staff_id.strip() or "TEST_STAFF"
    custom_vals["total_score"] = sum(custom_vals[c] for c in ALL_23_FEATURE_COLS)
    custom_vals["stress_class"] = 1 if custom_vals["total_score"] >= 40 else 0
    staff_row = pd.Series(custom_vals)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # ACTION BUTTON: PREDICT RISK LEVEL AND GENERATE FEEDBACK
    # -----------------------------------------------------------------------
    predict_clicked = st.button("🔮 Predict Risk Level & Generate Feedback", type="primary", use_container_width=True)

    if "has_evaluated" not in st.session_state:
        st.session_state.has_evaluated = False

    if predict_clicked:
        st.session_state.has_evaluated = True

    if st.session_state.has_evaluated:
        # Execute Model Classification
        explanation = explain_individual_staff(staff_row, active_pipeline, df_cohort, features=PEARSON_15_FEATURES)

        st.divider()

        # Results Summary Cards
        st.subheader("Assessment Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Assessed Risk Level",
            explanation["predicted_label"],
            delta="High Priority Intervention" if explanation["predicted_class"] == 1 else "Equilibrium Maintenance",
            delta_color="inverse" if explanation["predicted_class"] == 1 else "normal",
        )
        m2.metric("High-Stress Probability", f"{explanation['probability_high_stress']*100:.1f}%")
        m3.metric("Composite Score (`TOTAL`)", f"{int(staff_row['total_score'])}")
        m4.metric("Active Model", model_choice)

        st.divider()

        # Generative AI Feedback Section
        st.subheader("🤖 Generative AI Personalized Feedback")
        fb_result = generate_personalized_feedback(
            staff_row=staff_row,
            explanation=explanation,
            selected_model_name=model_choice,
            api_key=api_key_input.strip() or None,
            use_live_gemini=use_live,
        )
        st.info(f"**Generation Engine:** `{fb_result['generation_mode']}`")
        st.markdown(fb_result["feedback_text"])
    else:
        st.info("👆 Review or modify the questionnaire responses above and click **'Predict Risk Level & Generate Feedback'** to run the assessment.")


if __name__ == "__main__":
    main()
