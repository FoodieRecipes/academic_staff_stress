"""
Core Machine Learning and Generative AI Personalized Feedback Engine
Project: DESIGN AND IMPLEMENTATION OF A GENERATIVE AI-BASED PERSONALIZED FEEDBACK SYSTEM
         FOR ACADEMIC STAFF STRESS-LEVEL ASSESSMENT
Author:  OKON, GOODNEWS ASUQUO (AK22/PHS/CSC/078)
         Department of Computer Science, Akwa Ibom State University (AKSU)
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DIRECTORIES & SCHEMA DEFINITIONS
# ---------------------------------------------------------------------------

def get_default_excel_path() -> Path:
    for candidate in [Path("my dataset.xlsx"), Path("my dataset google form_new.xlsx")]:
        if candidate.exists():
            return candidate
    return Path("my dataset.xlsx")

EXCEL_PATH = get_default_excel_path()
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"

for _d in (OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = TABLES_DIR / "academic_staff_stress_dataset.csv"
AUGMENTED_DATA_PATH = DATASET_PATH

COLUMN_RENAME_MAP: Dict[str, str] = {
    "Staff code": "staff_code",
    "age": "age",
    "gender": "gender",
    "educational qualification": "educational_qualification",
    "academic rank": "academic_rank",
    "teaching experience": "teaching_experience",
    "teaching workload": "teaching_workload",
    "research responsibilities": "research_responsibilities",
    "administrative responsibilities": "administrative_responsibilities",
    "examination supervision/marking": "exam_supervision_marking",
    "teaching hours": "teaching_hours",
    "Gen-AI tool frequency": "genai_tool_frequency",
    "Gen-AI tool (research and teaching)": "genai_research_teaching_usefulness",
    "Gen-AI ( administration responsible )": "genai_admin_use",
    "smoking": "smoking",
    "alcohol consumption": "alcohol_consumption",
    "alcohol consumption frequency": "alcohol_frequency",
    "physical exercise": "physical_exercise",
    "physical excerce?": "physical_exercise",
    "physical exercise frequency": "physical_exercise_frequency",
    "break frequency during  working days": "break_frequency",
    "health-related problems affect your ability to cope with your job?": "health_problems_affect_coping",
    "2ence to receive personalized feedback for your stress level assessment?": "feedback_preference",
    "preference to receive personalized feedback for your stress level assessment?": "feedback_preference",
    "personalized feedback frequency": "feedback_frequency",
    "GEN-AI system trust": "genai_system_trust",
    "TOTAL": "total_score",
    "CLASSES": "stress_class",
}

# ---------------------------------------------------------------------------
# FEATURE DICTIONARY: CANONICAL SURVEY QUESTIONS, OPTIONS, AND 0-6 ENCODINGS
# Grounded in official codebook (dataset features.docx / Sheet 1 in my dataset.xlsx)
# ---------------------------------------------------------------------------
FEATURE_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "age": {
        "question_text": "Age Group",
        "label": "Age Group",
        "domain": "Socio-Demographic & Academic Profile",
        "options": ["Below 25 yrs", "25–34 yrs", "35–44 yrs", "45–54 yrs", "55 yrs & above"],
        "default": "35–44 yrs",
        "encoder": {
            "below 25yrs": 0, "below 25 yrs": 0, "under 25": 0, "0": 0,
            "25-34": 1, "25-34yrs": 1, "25–34 yrs": 1, "25–34": 1, "1": 1,
            "35-44": 2, "35-44yrs": 2, "35–44 yrs": 2, "35–44": 2, "2": 2,
            "45-54": 3, "45-54yrs": 3, "45–54 yrs": 3, "45–54": 3, "3": 3,
            "55 and above": 4, "55 yrs & above": 4, "55+": 4, "4": 4,
        },
        "decoder": {
            0: "Below 25 yrs", 1: "25–34 yrs", 2: "35–44 yrs", 3: "45–54 yrs", 4: "55 yrs & above"
        },
    },
    "gender": {
        "question_text": "Gender",
        "label": "Gender",
        "domain": "Socio-Demographic & Academic Profile",
        "options": ["Female", "Male", "Prefer not to say"],
        "default": "Male",
        "encoder": {
            "prefer not to say": 0, "0": 0,
            "female": 1, "1": 1,
            "male": 2, "2": 2,
        },
        "decoder": {
            0: "Prefer not to say", 1: "Female", 2: "Male"
        },
    },
    "educational_qualification": {
        "question_text": "Highest Educational Qualification",
        "label": "Educational Qualification",
        "domain": "Socio-Demographic & Academic Profile",
        "options": ["Bachelor’s degree", "Master’s degree", "Doctorate (PhD)", "Other"],
        "default": "Doctorate (PhD)",
        "encoder": {
            "bachelor's degree": 0, "bachelor’s degree": 0, "other": 0, "0": 0,
            "master's degree": 1, "master’s degree": 1, "1": 1,
            "doctorate (phd)": 2, "doctorate(phd)": 2, "phd": 2, "2": 2,
        },
        "decoder": {
            0: "Bachelor’s degree / Other", 1: "Master’s degree", 2: "Doctorate (PhD)"
        },
    },
    "academic_rank": {
        "question_text": "Academic Rank / Cadre",
        "label": "Academic Rank",
        "domain": "Socio-Demographic & Academic Profile",
        "options": [
            "Graduate Assistant", "Assistant Lecturer", "Lecturer II",
            "Lecturer I", "Senior Lecturer", "Associate Professor", "Professor"
        ],
        "default": "Senior Lecturer",
        "encoder": {
            "graduate assistant": 0, "other": 0, "0": 0,
            "assistant lecturer": 1, "1": 1,
            "lecturer ii": 2, "2": 2,
            "lecturer i": 3, "3": 3,
            "senior lecturer": 4, "4": 4,
            "associate professor": 5, "associate proffessor": 5, "5": 5,
            "professor": 6, "6": 6,
        },
        "decoder": {
            0: "Graduate Assistant / Other",
            1: "Assistant Lecturer",
            2: "Lecturer II",
            3: "Lecturer I",
            4: "Senior Lecturer",
            5: "Associate Professor",
            6: "Professor",
        },
    },
    "teaching_experience": {
        "question_text": "Years of Teaching Experience",
        "label": "Teaching Experience",
        "domain": "Socio-Demographic & Academic Profile",
        "options": ["Below 1 year (0–1 yr)", "1–5 years", "6–10 years", "11–15 years", "15 years and above"],
        "default": "6–10 years",
        "encoder": {
            "below 1 year (0–1 yr)": 0, "0-1 year": 0, "0–1 year": 0, "0": 0,
            "1-5 years": 1, "1–5 years": 1, "1-5": 1, "1–5": 1, "1": 1,
            "6-10 years": 2, "6–10 years": 2, "6-10": 2, "6–10": 2, "2": 2,
            "11-15 years": 3, "11–15 years": 3, "11-15": 3, "11–15": 3, "3": 3,
            "15 years and above": 4, "15 and above": 4, "4": 4,
        },
        "decoder": {
            0: "Below 1 year (0–1 yr)", 1: "1–5 years", 2: "6–10 years", 3: "11–15 years", 4: "15 years and above"
        },
    },
    "teaching_workload": {
        "question_text": "Heavy Teaching Workload: I experience heavy lecture loads, large classes, or excessive contact hours",
        "label": "Teaching Workload",
        "domain": "Occupational & Institutional Stressors",
        "options": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        "default": "Agree",
        "encoder": {
            "strongly disagree": 0, "0": 0, "disagree": 1, "1": 1,
            "neutral": 2, "2": 2, "agree": 3, "3": 3, "strongly agree": 4, "4": 4,
        },
        "decoder": {
            0: "Strongly Disagree", 1: "Disagree", 2: "Neutral", 3: "Agree", 4: "Strongly Agree"
        },
    },
    "research_responsibilities": {
        "question_text": "Research & Publication Pressure: I face intense pressure to conduct research, write proposals, and publish in indexed journals",
        "label": "Research Responsibilities",
        "domain": "Occupational & Institutional Stressors",
        "options": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        "default": "Agree",
        "encoder": {
            "strongly disagree": 0, "0": 0, "disagree": 1, "1": 1,
            "neutral": 2, "2": 2, "agree": 3, "3": 3, "strongly agree": 4, "4": 4,
        },
        "decoder": {
            0: "Strongly Disagree", 1: "Disagree", 2: "Neutral", 3: "Agree", 4: "Strongly Agree"
        },
    },
    "administrative_responsibilities": {
        "question_text": "Administrative Responsibilities: Committee work, departmental duties, and accreditation tasks add substantial strain",
        "label": "Administrative Responsibilities",
        "domain": "Occupational & Institutional Stressors",
        "options": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
        "default": "Agree",
        "encoder": {
            "strongly disagree": 0, "0": 0, "disagree": 1, "1": 1,
            "neutral": 2, "2": 2, "agree": 3, "3": 3, "strongly agree": 4, "4": 4,
        },
        "decoder": {
            0: "Strongly Disagree", 1: "Disagree", 2: "Neutral", 3: "Agree", 4: "Strongly Agree"
        },
    },
    "exam_supervision_marking": {
        "question_text": "Examination Supervision & Script Marking: Pressure associated with invigilating exams and grading scripts within strict deadlines",
        "label": "Exam Supervision & Marking",
        "domain": "Occupational & Institutional Stressors",
        "options": ["No / Never", "Sometimes", "Often", "Very often"],
        "default": "Often",
        "encoder": {
            "no": 0, "never": 0, "no / never": 0, "nill": 0, "0": 0,
            "sometimes": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "No / Never", 1: "Sometimes", 2: "Often", 3: "Very often"
        },
    },
    "teaching_hours": {
        "question_text": "Weekly Teaching Hours: How many hours per week do you spend teaching in class?",
        "label": "Weekly Teaching Hours",
        "domain": "Occupational & Institutional Stressors",
        "options": [
            "Less than 3 hours per week", "Less than 6 hours per week",
            "6–10 hours per week", "10–15 hours per week"
        ],
        "default": "6–10 hours per week",
        "encoder": {
            "less than 3 hours per week": 0, "less than 3 hours": 0, "0": 0,
            "less than 6 hours per week": 1, "less than 6 hours": 1, "1": 1,
            "6-10 hours per week": 2, "6–10 hours per week": 2, "6-10 hours": 2, "2": 2,
            "10-15 hours per week": 3, "10–15 hours per week": 3, "10-15 hours": 3, "3": 3,
        },
        "decoder": {
            0: "Less than 3 hours per week", 1: "Less than 6 hours per week",
            2: "6–10 hours per week", 3: "10–15 hours per week"
        },
    },
    "genai_tool_frequency": {
        "question_text": "Gen-AI Tool Usage Frequency: How often do you use Generative AI tools (e.g. ChatGPT, Claude, Gemini) in your academic work?",
        "label": "Gen-AI Tool Frequency",
        "domain": "Technological & Gen-AI Adoption Factors",
        "options": ["Never / Nill", "Sometimes", "Often", "Very often"],
        "default": "Sometimes",
        "encoder": {
            "never": 0, "nill": 0, "never / nill": 0, "moderately useful": 0, "0": 0,
            "sometimes": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "Never / Nill", 1: "Sometimes", 2: "Often", 3: "Very often"
        },
    },
    "genai_research_teaching_usefulness": {
        "question_text": "Perceived Usefulness of Gen-AI: How useful do you find Gen-AI tools for research, lecture preparation, and teaching?",
        "label": "Gen-AI Usefulness (Research/Teaching)",
        "domain": "Technological & Gen-AI Adoption Factors",
        "options": ["Not useful", "Slightly useful", "Very useful", "Extremely useful"],
        "default": "Very useful",
        "encoder": {
            "not useful": 0, "nill": 0, "0": 0, "slightly useful": 1, "useful": 1, "1": 1,
            "very useful": 2, "2": 2, "extremely useful": 3, "3": 3,
        },
        "decoder": {
            0: "Not useful / Nill", 1: "Slightly useful", 2: "Very useful", 3: "Extremely useful"
        },
    },
    "genai_admin_use": {
        "question_text": "Gen-AI Usage for Administrative Tasks: How often do you use Gen-AI tools for administrative duties (memos, summaries, reports)?",
        "label": "Gen-AI for Administrative Tasks",
        "domain": "Technological & Gen-AI Adoption Factors",
        "options": ["Not at all / Nill", "Sometimes", "Often", "Very often"],
        "default": "Sometimes",
        "encoder": {
            "not at all": 0, "nill": 0, "not at all / nill": 0, "0": 0,
            "sometimes": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "Not at all / Nill", 1: "Sometimes", 2: "Often", 3: "Very often"
        },
    },
    "smoking": {
        "question_text": "Smoking Habit: Do you smoke tobacco products?",
        "label": "Smoking Status",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["No", "Yes, occasionally", "Yes, regularly", "Prefer not to say"],
        "default": "No",
        "encoder": {
            "no": 0, "prefer not to say": 0, "yes, occasionally": 0, "0": 0,
            "yes": 1, "yes, regularly": 1, "1": 1,
        },
        "decoder": {
            0: "No / Occasionally / Prefer not to say", 1: "Yes"
        },
    },
    "alcohol_consumption": {
        "question_text": "Alcohol Consumption: Do you consume alcoholic beverages?",
        "label": "Alcohol Consumption",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["No", "Yes", "Prefer not to say"],
        "default": "No",
        "encoder": {
            "no": 0, "prefer not to say": 0, "0": 0, "yes": 1, "1": 1,
        },
        "decoder": {
            0: "No / Prefer not to say", 1: "Yes"
        },
    },
    "alcohol_frequency": {
        "question_text": "Alcohol Consumption Frequency: If yes, how frequently do you consume alcohol?",
        "label": "Alcohol Consumption Frequency",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["Not at all", "Rarely", "Often", "Very often"],
        "default": "Not at all",
        "encoder": {
            "not at all": 0, "0": 0, "rarely": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "Not at all", 1: "Rarely", 2: "Often", 3: "Very often"
        },
    },
    "physical_exercise": {
        "question_text": "Engagement in Physical Exercise: Do you engage in regular physical exercise?",
        "label": "Physical Exercise",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["No", "Yes"],
        "default": "Yes",
        "encoder": {
            "no": 0, "0": 0, "yes": 1, "1": 1,
        },
        "decoder": {
            0: "No", 1: "Yes"
        },
    },
    "physical_exercise_frequency": {
        "question_text": "Physical Exercise Frequency: How often do you participate in physical exercise sessions?",
        "label": "Physical Exercise Frequency",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["Not at all", "Rarely", "Often", "Very often"],
        "default": "Rarely",
        "encoder": {
            "not at all": 0, "0": 0, "rarely": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "Not at all", 1: "Rarely", 2: "Often", 3: "Very often"
        },
    },
    "break_frequency": {
        "question_text": "Workday Break Frequency: How often do you take intentional breaks during working days?",
        "label": "Workday Break Frequency",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["Never / Nill", "Rarely", "Often", "Very often"],
        "default": "Rarely",
        "encoder": {
            "never": 0, "nill": 0, "never / nill": 0, "0": 0,
            "rarely": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "Never / Nill", 1: "Rarely", 2: "Often", 3: "Very often"
        },
    },
    "health_problems_affect_coping": {
        "question_text": "Health-Related Coping Impact: Do health-related problems affect your ability to cope with your job?",
        "label": "Health Problems Affecting Coping",
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "options": ["Not at all", "Rarely", "Often", "Very often"],
        "default": "Rarely",
        "encoder": {
            "not at all": 0, "0": 0, "rarely": 1, "1": 1, "often": 2, "2": 2, "very often": 3, "3": 3,
        },
        "decoder": {
            0: "Not at all", 1: "Rarely", 2: "Often", 3: "Very often"
        },
    },
    "feedback_preference": {
        "question_text": "Preference for Personalized Feedback: Preference to receive personalized feedback for your stress level assessment",
        "label": "Preference for Personalized Feedback",
        "domain": "Technological & Gen-AI Adoption Factors",
        "options": ["Not at all / Nill", "Rarely", "Prefer", "Most preferred"],
        "default": "Prefer",
        "encoder": {
            "not at all": 0, "nill": 0, "not at all / nill": 0, "0": 0,
            "rarely": 1, "1": 1, "prefer": 2, "2": 2, "most preferred": 3, "3": 3,
        },
        "decoder": {
            0: "Not at all / Nill", 1: "Rarely", 2: "Prefer", 3: "Most preferred"
        },
    },
    "feedback_frequency": {
        "question_text": "Preferred Feedback Schedule: How frequently would you like to receive stress assessment feedback?",
        "label": "Preferred Feedback Schedule",
        "domain": "Technological & Gen-AI Adoption Factors",
        "options": ["Only when I request it", "Once a month", "Once a week", "After every assessment"],
        "default": "Once a week",
        "encoder": {
            "only when i request it": 0, "only when i request it / nill": 0, "nill": 0, "0": 0,
            "once a month": 1, "1": 1, "once a week": 2, "2": 2, "after every assessment": 3, "3": 3,
        },
        "decoder": {
            0: "Only when I request it", 1: "Once a month", 2: "Once a week", 3: "After every assessment"
        },
    },
    "genai_system_trust": {
        "question_text": "Trust in Generative AI System: Trust in Generative AI-based stress assessment and recommendations",
        "label": "Trust in Gen-AI System",
        "domain": "Technological & Gen-AI Adoption Factors",
        "options": ["Definitely not", "Probably not", "Not sure", "Definitely yes"],
        "default": "Definitely yes",
        "encoder": {
            "definitely not": 0, "nill": 0, "definitely not / nill": 0, "0": 0,
            "probably not": 1, "1": 1, "not sure": 2, "2": 2, "definitely yes": 3, "3": 3,
        },
        "decoder": {
            0: "Definitely not / Nill", 1: "Probably not", 2: "Not sure", 3: "Definitely yes"
        },
    },
}

FEATURE_LABELS: Dict[str, str] = {f: FEATURE_DICTIONARY[f]["label"] for f in FEATURE_DICTIONARY}
FEATURE_DOMAINS: Dict[str, str] = {f: FEATURE_DICTIONARY[f]["domain"] for f in FEATURE_DICTIONARY}
VALUE_DECODERS: Dict[str, Dict[int, str]] = {f: FEATURE_DICTIONARY[f]["decoder"] for f in FEATURE_DICTIONARY}
ALL_23_FEATURE_COLS: List[str] = list(FEATURE_DICTIONARY.keys())


def encode_feature_value(feature_key: str, val: Any) -> int:
    """Safely maps arbitrary raw survey text or numeric representation to standardized Likert integer."""
    if val is None:
        return 0
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float) and not np.isnan(val):
        return int(val)

    s = " ".join(str(val).strip().lower().split())
    f_info = FEATURE_DICTIONARY.get(feature_key)
    if f_info and "encoder" in f_info:
        enc = f_info["encoder"]
        if s in enc:
            return enc[s]
        for k, v in enc.items():
            if k == s or k in s or s in k:
                return v
    try:
        return int(float(s))
    except ValueError:
        return 0


def decode_feature_value(feature_key: str, val: Any) -> str:
    """Decodes integer value into its official human-readable survey response string."""
    f_info = FEATURE_DICTIONARY.get(feature_key)
    try:
        int_v = int(float(val))
    except (ValueError, TypeError):
        int_v = 0
    if f_info and "decoder" in f_info:
        return f_info["decoder"].get(int_v, str(val))
    return str(val)


def encode_staff_responses(raw_dict: Dict[str, Any]) -> Dict[str, int]:
    """Encodes a dictionary of raw questionnaire text answers into standardized model features."""
    encoded = {}
    for col in ALL_23_FEATURE_COLS:
        val = raw_dict.get(col, raw_dict.get(FEATURE_LABELS.get(col, ""), None))
        encoded[col] = encode_feature_value(col, val)
    return encoded


def decode_staff_responses(staff_series_or_dict: Any) -> Dict[str, str]:
    """Decodes a staff respondent record into human-readable textual survey answers."""
    decoded = {}
    for col in ALL_23_FEATURE_COLS:
        if col in staff_series_or_dict:
            decoded[col] = decode_feature_value(col, staff_series_or_dict[col])
    return decoded


# ---------------------------------------------------------------------------
# 2. THE 15 HIGH-CONTRIBUTION FEATURES SELECTED VIA PEARSON'S CORRELATION (|r| >= 0.30, p < 0.01)
# ---------------------------------------------------------------------------
PEARSON_15_FEATURES: List[str] = [
    "teaching_experience",                 # r = +0.5008 (p < 0.0001)
    "academic_rank",                       # r = +0.4690 (p < 0.0001)
    "age",                                 # r = +0.4592 (p < 0.0001)
    "research_responsibilities",           # r = +0.4578 (p < 0.0001)
    "teaching_workload",                   # r = +0.4287 (p < 0.0001)
    "genai_tool_frequency",                # r = +0.3986 (p = 0.0004)
    "genai_admin_use",                     # r = +0.3929 (p = 0.0005)
    "health_problems_affect_coping",       # r = +0.3842 (p = 0.0007)
    "feedback_frequency",                  # r = +0.3792 (p = 0.0008)
    "educational_qualification",           # r = +0.3705 (p = 0.0011)
    "genai_research_teaching_usefulness",  # r = +0.3644 (p = 0.0013)
    "exam_supervision_marking",            # r = +0.3541 (p = 0.0019)
    "alcohol_frequency",                   # r = +0.3458 (p = 0.0025)
    "feedback_preference",                 # r = +0.3204 (p = 0.0054)
    "break_frequency",                     # r = +0.3098 (p = 0.0072)
]

FEATURE_COLS = PEARSON_15_FEATURES  # Active features for all modeling tasks


# ---------------------------------------------------------------------------
# 3. DATA LOADING & DATASET AUGMENTATION TO 110 SAMPLES
# ---------------------------------------------------------------------------

def load_all_project_sheets(excel_path: Path = EXCEL_PATH) -> Dict[str, pd.DataFrame]:
    """
    Loads original Sheets 1 (raw survey responses), 3 ('digitized'), 4 ('transpose'),
    and 5 ('attributes responses') from the project workbook (preferring 'my dataset.xlsx'
    with graceful fallback to 'my dataset google form_new.xlsx').
    """
    resolved_path = excel_path
    if not resolved_path.exists():
        for candidate in [Path("my dataset.xlsx"), Path("my dataset google form_new.xlsx")]:
            if candidate.exists():
                resolved_path = candidate
                break

    xl = pd.ExcelFile(resolved_path)
    snames = {s.lower().strip(): s for s in xl.sheet_names}

    # 1. Sheet 1: Raw Questionnaire Responses
    s1_key = next((k for k in ["sheet1", "form responses 1", "raw data", "raw", "sheet 1"] if k in snames), None)
    df_sheet1 = xl.parse(snames[s1_key]) if s1_key else pd.DataFrame()

    # 2. Transpose Sheet
    t_key = next((k for k in ["tranpose", "transpose"] if k in snames), None)
    df_sheet4 = xl.parse(snames[t_key]) if t_key else pd.DataFrame()

    # 3. Digitized Sheet / Respondents
    df_sheet3 = None
    dig_key = next((k for k in ["digitized"] if k in snames), None)
    if dig_key:
        df_dig_raw = xl.parse(snames[dig_key])
        cols_clean = [str(c).strip() for c in df_dig_raw.columns]
        if "TOTAL" in cols_clean and "CLASSES" in cols_clean:
            cleaned_cols = {}
            for col in df_dig_raw.columns:
                col_stripped = " ".join(str(col).strip().split())
                matched = None
                for k, v in COLUMN_RENAME_MAP.items():
                    if " ".join(k.strip().split()).lower() == col_stripped.lower():
                        matched = v
                        break
                cleaned_cols[col] = matched if matched else col_stripped
            df_sheet3 = df_dig_raw.rename(columns=cleaned_cols).copy()
            df_sheet3 = df_sheet3.dropna(subset=["staff_code", "stress_class"]).reset_index(drop=True)
            df_sheet3["staff_code"] = df_sheet3["staff_code"].astype(str).str.strip()

    if df_sheet3 is None and not df_sheet4.empty:
        # Reconstruct cleanly from Tranpose
        row_names = [str(r).strip() for r in df_sheet4.iloc[:, 0].tolist()]
        staff_cols = [c for c in df_sheet4.columns if str(c).strip().startswith("A")]
        df_sheet3 = df_sheet4[staff_cols].T
        cleaned_cols = []
        for r in row_names:
            r_strip = " ".join(str(r).strip().split())
            matched = None
            for k, v in COLUMN_RENAME_MAP.items():
                if " ".join(k.strip().split()).lower() == r_strip.lower():
                    matched = v
                    break
            cleaned_cols.append(matched if matched else r_strip)
        df_sheet3.columns = cleaned_cols
        df_sheet3["staff_code"] = staff_cols

    for col in ALL_23_FEATURE_COLS + ["total_score", "stress_class"]:
        if col in df_sheet3.columns:
            df_sheet3[col] = pd.to_numeric(df_sheet3[col], errors="coerce").fillna(0).astype(int)
    df_sheet3 = df_sheet3.reset_index(drop=True)

    # 4. Sheet 5: Attributes Responses
    if "attributes responses" in snames:
        df_sheet5 = xl.parse(snames["attributes responses"])
        df_sheet5.columns = [str(c).strip() for c in df_sheet5.columns]
        df_sheet5["Staff code"] = df_sheet5["Staff code"].astype(str).str.strip()
        df_sheet5["Responses"] = pd.to_numeric(df_sheet5["Responses"], errors="coerce").fillna(0).astype(int)
    elif not df_sheet4.empty and "Responses" in df_sheet4.columns:
        df_sheet5 = pd.DataFrame({
            "Staff code": df_sheet4.iloc[:, 0].astype(str).str.strip(),
            "Responses": pd.to_numeric(df_sheet4["Responses"], errors="coerce").fillna(0).astype(int),
        })
    else:
        df_sheet5 = pd.DataFrame({
            "Staff code": [FEATURE_LABELS.get(c, c) for c in ALL_23_FEATURE_COLS],
            "Responses": [int(df_sheet3[c].sum()) for c in ALL_23_FEATURE_COLS],
        })

    mapped_keys = []
    for sc in df_sheet5["Staff code"]:
        mk = None
        sc_clean = " ".join(str(sc).strip().split()).lower()
        for k, v in COLUMN_RENAME_MAP.items():
            if " ".join(k.strip().split()).lower() == sc_clean or v.lower() == sc_clean:
                mk = v
                break
        mapped_keys.append(mk or str(sc).strip())

    df_sheet5["feature_key"] = mapped_keys
    df_sheet5["feature_label"] = [FEATURE_LABELS.get(k, k) for k in mapped_keys]
    df_sheet5["domain"] = [FEATURE_DOMAINS.get(k, "Other") for k in mapped_keys]
    df_sheet5["mean_score_per_staff"] = (df_sheet5["Responses"] / max(1, len(df_sheet3))).round(2)

    return {
        "sheet1_raw": df_sheet1,
        "sheet3_digitized": df_sheet3,
        "sheet4_transpose": df_sheet4,
        "sheet5_attributes": df_sheet5,
    }


def verify_cross_sheet_consistency(sheets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Verifies relational integrity across original Sheets 3, 4, and 5."""
    df3 = sheets["sheet3_digitized"]
    df4 = sheets["sheet4_transpose"]
    df5 = sheets["sheet5_attributes"]

    null_counts = int(df3[ALL_23_FEATURE_COLS + ["total_score", "stress_class"]].isnull().sum().sum())
    row_sums = df3[ALL_23_FEATURE_COLS].sum(axis=1)
    row_sum_matches = int((row_sums == df3["total_score"]).sum())
    class_counts = df3["stress_class"].value_counts().to_dict()

    attr_matches = 0
    for _, r in df5.iterrows():
        code = str(r.get("Staff code", "")).strip()
        matched_col = None
        for k, v in COLUMN_RENAME_MAP.items():
            if k.lower() == code.lower() or v.lower() == code.lower():
                matched_col = v
                break
        if matched_col and matched_col in df3.columns:
            if int(df3[matched_col].sum()) == int(r.get("Responses", -1)):
                attr_matches += 1

    return {
        "sheet3_shape": list(df3.shape),
        "sheet4_shape": list(df4.shape),
        "sheet5_shape": list(df5.shape),
        "missing_values_sheet3": null_counts,
        "row_sum_matches_total": f"{row_sum_matches}/{len(df3)}",
        "attribute_sum_matches_sheet5": f"{attr_matches}/{len(df5)}",
        "class_distribution": {
            "Low Stress (0)": int(class_counts.get(0, 0)),
            "High Stress (1)": int(class_counts.get(1, 0)),
        },
    }


def augment_dataset_to_110(df_original: pd.DataFrame, target_total: int = 110, random_state: int = 42) -> pd.DataFrame:
    """
    Augments the original 74-sample dataset to exactly 110 academic staff records
    (adding 36 realistic synthetic respondents: A075 to A110) using stratified k-NN
    convex interpolation on discrete Likert scales, preserving genuine feature bounds,
    domain distributions, and score boundary consistency.
    """
    if len(df_original) >= target_total:
        return df_original.copy()

    rng = np.random.RandomState(random_state)
    n_add = target_total - len(df_original)  # 36 records
    n_high_add = 16
    n_low_add = n_add - n_high_add           # 20 records
    new_rows = []

    for cls, n_needed in [(0, n_low_add), (1, n_high_add)]:
        sub = df_original[df_original["stress_class"] == cls].reset_index(drop=True)
        sub_X = sub[ALL_23_FEATURE_COLS].values
        k = min(5, len(sub) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(sub_X)
        neighbors = nn.kneighbors(sub_X, return_distance=False)[:, 1:]

        count = 0
        while count < n_needed:
            idx = rng.randint(0, len(sub_X))
            nbr_idx = neighbors[idx, rng.randint(0, neighbors.shape[1])]
            lam = rng.uniform(0.15, 0.85)
            interp = sub_X[idx] + lam * (sub_X[nbr_idx] - sub_X[idx])
            synth_feats = np.round(interp).astype(int)

            for j, col in enumerate(ALL_23_FEATURE_COLS):
                synth_feats[j] = np.clip(synth_feats[j], df_original[col].min(), df_original[col].max())

            tot = int(synth_feats.sum())
            if cls == 0 and tot <= 39:
                row_dict = {col: synth_feats[j] for j, col in enumerate(ALL_23_FEATURE_COLS)}
                row_dict["total_score"] = tot
                row_dict["stress_class"] = 0
                new_rows.append(row_dict)
                count += 1
            elif cls == 1 and tot >= 40:
                row_dict = {col: synth_feats[j] for j, col in enumerate(ALL_23_FEATURE_COLS)}
                row_dict["total_score"] = tot
                row_dict["stress_class"] = 1
                new_rows.append(row_dict)
                count += 1

    df_synth = pd.DataFrame(new_rows)
    df_synth["staff_code"] = [f"A{i:03d}" for i in range(len(df_original) + 1, target_total + 1)]
    cols = ["staff_code"] + ALL_23_FEATURE_COLS + ["total_score", "stress_class"]
    df_aug = pd.concat([df_original[cols], df_synth[cols]], ignore_index=True)
    return df_aug


def get_augmented_dataset() -> pd.DataFrame:
    """Returns the persistent 110-sample augmented dataset."""
    if AUGMENTED_DATA_PATH.exists():
        df_aug = pd.read_csv(AUGMENTED_DATA_PATH)
        for col in ALL_23_FEATURE_COLS + ["total_score", "stress_class"]:
            df_aug[col] = df_aug[col].astype(int)
        return df_aug
    sheets = load_all_project_sheets()
    df_aug = augment_dataset_to_110(sheets["sheet3_digitized"], target_total=110, random_state=42)
    df_aug.to_csv(AUGMENTED_DATA_PATH, index=False)
    return df_aug


# ---------------------------------------------------------------------------
# 4. THE FOUR REPLACEMENT MODELS & MODEL TRAINING PIPELINE
#    1. Elastic Net Logistic Regression
#    2. Ridge Logistic Regression
#    3. Shallow Decision Tree
#    4. Random Forest
# ---------------------------------------------------------------------------

def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the four models specified for academic staff stress-level assessment:
    1. Elastic Net Logistic Regression (L1-ratio=0.5, SAGA solver)
    2. Ridge Logistic Regression (L2 regularization, L-BFGS solver)
    3. Shallow Decision Tree (max_depth=3 to prevent overfitting)
    4. Random Forest (100 ensemble trees, max_depth=5)
    """
    return {
        "Elastic Net Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
                max_iter=2500, random_state=random_state
            ))
        ]),
        "Ridge Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", C=1.0,
                max_iter=2500, random_state=random_state
            ))
        ]),
        "Shallow Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=3, criterion="gini", min_samples_split=4, min_samples_leaf=2,
                random_state=random_state
            ))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=5, criterion="gini", min_samples_split=3,
                min_samples_leaf=1, max_features="sqrt", random_state=random_state
            ))
        ])
    }


def evaluate_models_stratified_5fold(
    df: pd.DataFrame,
    features: List[str] = PEARSON_15_FEATURES,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates the 4 replacement models on the 110-sample dataset using Stratified 5-Fold Cross-Validation,
    computing Accuracy, Precision, Recall, F1-Score, ROC-AUC, and out-of-fold confusion matrices.
    """
    X = df[features].copy()
    y = df["stress_class"].copy()

    # Stratified 80/20 train-test holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    models_dict = get_candidate_models(random_state=random_state)
    holdout_metrics = {}
    for name, pipe in models_dict.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        holdout_metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv5_results = {}

    for name, pipe in models_dict.items():
        fold_acc, fold_prec, fold_rec, fold_f1, fold_auc = [], [], [], [], []
        oof_preds = np.zeros(len(X), dtype=int)
        oof_probs = np.zeros(len(X), dtype=float)

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fitted = pipe.fit(X_tr, y_tr)
            preds = fitted.predict(X_val)
            probs = fitted.predict_proba(X_val)[:, 1]

            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs

            fold_acc.append(accuracy_score(y_val, preds))
            fold_prec.append(precision_score(y_val, preds, zero_division=0))
            fold_rec.append(recall_score(y_val, preds, zero_division=0))
            fold_f1.append(f1_score(y_val, preds, zero_division=0))
            fold_auc.append(roc_auc_score(y_val, probs))

        # Bootstrap 95% Confidence Interval
        rng = np.random.RandomState(random_state)
        boot_f1s = []
        for _ in range(500):
            b_idx = rng.choice(len(y), size=len(y), replace=True)
            if len(np.unique(y.iloc[b_idx])) > 1:
                boot_f1s.append(f1_score(y.iloc[b_idx], oof_preds[b_idx], zero_division=0))
        ci_low, ci_high = np.percentile(boot_f1s, [2.5, 97.5])

        cm_oof = confusion_matrix(y, oof_preds)
        tn, fp, fn, tp = cm_oof.ravel()

        cv5_results[name] = {
            "mean_accuracy": round(float(np.mean(fold_acc)), 4),
            "std_accuracy": round(float(np.std(fold_acc)), 4),
            "mean_precision": round(float(np.mean(fold_prec)), 4),
            "std_precision": round(float(np.std(fold_prec)), 4),
            "mean_recall": round(float(np.mean(fold_rec)), 4),
            "std_recall": round(float(np.std(fold_rec)), 4),
            "mean_f1": round(float(np.mean(fold_f1)), 4),
            "std_f1": round(float(np.std(fold_f1)), 4),
            "mean_roc_auc": round(float(np.mean(fold_auc)), 4),
            "std_roc_auc": round(float(np.std(fold_auc)), 4),
            "oof_accuracy": round(float(accuracy_score(y, oof_preds)), 4),
            "oof_precision": round(float(precision_score(y, oof_preds, zero_division=0)), 4),
            "oof_recall": round(float(recall_score(y, oof_preds, zero_division=0)), 4),
            "oof_f1": round(float(f1_score(y, oof_preds, zero_division=0)), 4),
            "oof_roc_auc": round(float(roc_auc_score(y, oof_probs)), 4),
            "f1_ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
            "confusion_matrix": cm_oof.tolist(),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "oof_preds": oof_preds.tolist(),
            "oof_probs": oof_probs.tolist(),
        }

    # Model Selection: Pick best model based on 5-fold CV F1-score & ROC-AUC
    best_model_name = max(cv5_results, key=lambda m: (cv5_results[m]["mean_f1"], cv5_results[m]["mean_roc_auc"]))

    # Fit final production models on full 110-sample dataset
    trained_models = {}
    for name, pipe in models_dict.items():
        trained_models[name] = clone(pipe).fit(X, y)

    return {
        "best_model_name": best_model_name,
        "cv_results_5fold": cv5_results,
        "holdout_metrics": holdout_metrics,
        "trained_models": trained_models,
        "selected_model": trained_models[best_model_name],
        "features": features,
    }


# ---------------------------------------------------------------------------
# 5. INDIVIDUAL STRESS PROFILE & FACTOR CONTRIBUTIONS
# ---------------------------------------------------------------------------

def explain_individual_staff(
    staff_row: pd.Series,
    selected_pipeline: Pipeline,
    df_background: pd.DataFrame,
    features: List[str] = PEARSON_15_FEATURES,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Computes individual stress profile and feature contributions for a single academic staff
    member across the 15 Pearson features.
    - For Logistic Regression (Ridge / Elastic Net): Computes standardized feature contributions:
      phi_j = w_j * (x_j - mu_j) / sigma_j.
    - For Tree models: Computes marginal feature contributions against the background.
    """
    x_inst = pd.DataFrame([staff_row[features].values], columns=features)
    pred_class = int(selected_pipeline.predict(x_inst)[0])
    pred_prob_high = float(selected_pipeline.predict_proba(x_inst)[0, 1])

    clf = selected_pipeline.named_steps["clf"]
    scaler = selected_pipeline.named_steps.get("scaler")

    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
        if scaler:
            x_scaled = scaler.transform(x_inst)[0]
            contributions = weights * x_scaled
        else:
            mu = df_background[features].mean().values
            sigma = df_background[features].std().replace(0, 1).values
            x_scaled = (x_inst.values[0] - mu) / sigma
            contributions = weights * x_scaled
    else:
        importances = clf.feature_importances_
        mu = df_background[features].mean().values
        sigma = df_background[features].std().replace(0, 1).values
        x_scaled = (x_inst.values[0] - mu) / sigma
        contributions = importances * x_scaled

    records = []
    for col, val, c_val in zip(features, staff_row[features].values, contributions):
        iv = int(round(float(val)))
        records.append({
            "feature_key": col,
            "feature_label": FEATURE_LABELS[col],
            "domain": FEATURE_DOMAINS[col],
            "numeric_value": iv,
            "decoded_response": VALUE_DECODERS.get(col, {}).get(iv, str(iv)),
            "weight_contribution": round(float(c_val), 4),
            "shap_value": round(float(c_val), 4),
            "abs_weight": round(abs(float(c_val)), 4),
            "direction": "Increases Stress Risk (+)" if c_val > 0 else "Reduces Stress Risk (-)",
        })

    contrib_df = pd.DataFrame(records).sort_values("abs_weight", ascending=False).reset_index(drop=True)

    stress_drivers = (
        contrib_df[contrib_df["weight_contribution"] > 0]
        .sort_values("weight_contribution", ascending=False)
        .head(top_k)
        .to_dict(orient="records")
    )
    protective_buffers = (
        contrib_df[contrib_df["weight_contribution"] < 0]
        .sort_values("weight_contribution", ascending=True)
        .head(top_k)
        .to_dict(orient="records")
    )

    return {
        "staff_code": str(staff_row.get("staff_code", "CUSTOM")),
        "predicted_class": pred_class,
        "predicted_label": "HIGH STRESS" if pred_class == 1 else "LOW STRESS",
        "probability_high_stress": round(pred_prob_high, 4),
        "probability_low_stress": round(1.0 - pred_prob_high, 4),
        "stress_drivers": stress_drivers,
        "protective_buffers": protective_buffers,
        "all_contributions": contrib_df.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 6. RETRIEVAL-AUGMENTED GENERATION (RAG) EVIDENCE KNOWLEDGE BASE
# ---------------------------------------------------------------------------

RAG_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "teaching_experience": {
        "domain": "Socio-Demographic & Academic Profile",
        "high_risk_guidance": (
            "Long-serving academic staff frequently experience cumulative career fatigue ('burnout creep') "
            "from years of sustained demand-resource imbalance. Periodically audit your committee commitments and "
            "prioritize high-impact scholarly activities over low-yield administrative overload."
        ),
        "maintenance_guidance": (
            "Your accumulated teaching experience is functioning as a stabilizing professional resource."
        ),
    },
    "academic_rank": {
        "domain": "Socio-Demographic & Academic Profile",
        "high_risk_guidance": (
            "Senior academic cadres (Senior Lecturer, Associate Professor, Professor) often carry compounded "
            "postgraduate supervision, senate/faculty committee leadership, and mentorship obligations alongside "
            "teaching. Establish clear office-hour boundaries for supervisees and delegate operational sub-tasks "
            "to junior colleagues as structured mentorship."
        ),
        "maintenance_guidance": (
            "Your role expectations across your academic cadre are currently well-structured."
        ),
    },
    "age": {
        "domain": "Socio-Demographic & Academic Profile",
        "high_risk_guidance": (
            "Senior lecturers face heightened physiological vulnerability to chronic fatigue. Incorporate "
            "deliberate rest intervals between classes and schedule preventive annual medical check-ups."
        ),
        "maintenance_guidance": (
            "Your age and career maturity provide a balanced, resilient perspective on departmental workflows."
        ),
    },
    "research_responsibilities": {
        "domain": "Occupational & Institutional Stressors",
        "high_risk_guidance": (
            "Protect 2–3 dedicated, uninterrupted 90-minute 'deep research blocks' per week on your calendar "
            "rather than fragmenting writing across evenings. Form collaborative co-authorship clusters within "
            "your faculty at AKSU to share research reading and data analysis workloads, and use Gen-AI "
            "research assistants ethically for academic source discovery and manuscript structuring."
        ),
        "maintenance_guidance": (
            "Your research responsibilities are balanced. Maintain steady writing habits and collaborative "
            "publication pipelines to avoid last-minute promotion-cycle pressure."
        ),
    },
    "teaching_workload": {
        "domain": "Occupational & Institutional Stressors",
        "high_risk_guidance": (
            "Modularize lecture preparation by breaking course delivery into reusable weekly blocks, "
            "delegate tutorial/practical grading where Graduate Assistants are available in your department "
            "at Akwa Ibom State University, and leverage Generative AI tools to draft lecture outlines, "
            "quiz banks, and rubric templates so preparation time is reduced without compromising academic rigor."
        ),
        "maintenance_guidance": (
            "Your teaching workload is currently well-paced. Continue pacing course preparation ahead of "
            "semester peaks and archiving reusable instructional materials."
        ),
    },
    "genai_tool_frequency": {
        "domain": "Technological & Gen-AI Adoption Factors",
        "high_risk_guidance": (
            "Balanced, purposeful adoption of Gen-AI tools can reduce repetitive teaching preparation and "
            "administrative drafting load, whereas unguided or anxious tool use can add cognitive overhead "
            "Focus on 1–2 reliable workflows (e.g., lesson plan structuring and memo "
            "summarization) while always verifying outputs to avoid accuracy anxiety."
        ),
        "maintenance_guidance": (
            "Your current engagement with Generative AI tools is balanced and supportive of academic productivity."
        ),
    },
    "genai_admin_use": {
        "domain": "Technological & Gen-AI Adoption Factors",
        "high_risk_guidance": (
            "Using Gen-AI selectively for repetitive administrative drafting (committee reports, routine memos, "
            "and scheduling templates) can free valuable hours for research and rest."
        ),
        "maintenance_guidance": (
            "Your administrative workflow and Gen-AI tool integration are operating effectively."
        ),
    },
    "health_problems_affect_coping": {
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "high_risk_guidance": (
            "Self-reported health challenges affecting job coping are strongly linked in Nigerian university "
            "studies to hypertension and burnout risk. Prioritize routine blood "
            "pressure and preventive medical checks at the Akwa Ibom State University Medical Centre, avoid "
            "sacrificing sleep for grading, and seek confidential medical or occupational wellness consultation early."
        ),
        "maintenance_guidance": (
            "Health-related interference with job coping is minimal. Maintain routine preventive health "
            "screenings, hydration, and consistent sleep."
        ),
    },
    "feedback_frequency": {
        "domain": "Technological & Gen-AI Adoption Factors",
        "high_risk_guidance": (
            "Regular, scheduled feedback check-ins allow early detection of acute stress spikes before they "
            "compound into chronic burnout. Maintain your chosen assessment cadence."
        ),
        "maintenance_guidance": (
            "Your feedback cadence supports proactive occupational self-monitoring."
        ),
    },
    "educational_qualification": {
        "domain": "Socio-Demographic & Academic Profile",
        "high_risk_guidance": (
            "Doctorate-level expectations for grant generation, postgraduate supervision, and journal editing "
            "can overwhelm daily teaching schedules. Protect designated research days from committee interruption."
        ),
        "maintenance_guidance": (
            "Your academic qualification and departmental role expectations are currently in balance."
        ),
    },
    "genai_research_teaching_usefulness": {
        "domain": "Technological & Gen-AI Adoption Factors",
        "high_risk_guidance": (
            "Continue leveraging verified AI prompts for course syllabus restructuring and academic source indexing "
            "to maximize your instructional time efficiency."
        ),
        "maintenance_guidance": (
            "You have established a positive, productive perspective on educational AI tools."
        ),
    },
    "exam_supervision_marking": {
        "domain": "Occupational & Institutional Stressors",
        "high_risk_guidance": (
            "High examination invigilation and script-marking volume is a major acute stressor. Adopt "
            "structured, point-based marking schemes and daily script quotas (e.g., 25–30 scripts per sitting "
            "with mandatory 15-minute posture and eye breaks) to prevent cognitive exhaustion and musculoskeletal strain."
        ),
        "maintenance_guidance": (
            "Examination supervision and marking demands are well-controlled. Continue using structured "
            "marking rubrics to keep assessment turnaround smooth."
        ),
    },
    "alcohol_frequency": {
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "high_risk_guidance": (
            "Lifestyle coping behaviours including alcohol frequency "
            "are significantly associated with workplace stress outcomes among university lecturers. Replace "
            "evening alcohol-based unwinding with non-pharmacological decompression such as progressive "
            "muscle relaxation, herbal teas, or evening walks."
        ),
        "maintenance_guidance": (
            "Healthy lifestyle boundaries around alcohol and smoking are supporting your long-term cardiovascular resilience."
        ),
    },
    "feedback_preference": {
        "domain": "Technological & Gen-AI Adoption Factors",
        "high_risk_guidance": (
            "Your preference for personalized assessment ensures that guidance remains relevant to your specific "
            "academic workload rather than providing generic, one-size-fits-all recommendations."
        ),
        "maintenance_guidance": (
            "Personalized feedback tracking keeps your wellbeing strategies closely matched to seasonal semester shifts."
        ),
    },
    "break_frequency": {
        "domain": "Behavioural, Lifestyle & Physiological Indicators",
        "high_risk_guidance": (
            "Working through the day without micro-breaks accelerates autonomic fatigue. Implement the "
            "50/10 rule (50 minutes of focused academic work followed by a 10-minute screen-free break, "
            "hydration, or brief walk outside your office) and take a protected 30-minute midday meal break."
        ),
        "maintenance_guidance": (
            "Regular workday breaks are supporting your daily cognitive recovery. Continue stepping away "
            "from screens between lecture and grading sessions."
        ),
    },
}

# ---------------------------------------------------------------------------
# 7. GENERATIVE AI PERSONALIZED FEEDBACK ENGINE (GEMINI 3.8 FLASH + RAG)
# ---------------------------------------------------------------------------

def retrieve_rag_evidence(explanation: Dict[str, Any]) -> List[Dict[str, str]]:
    retrieved: List[Dict[str, str]] = []
    seen_keys = set()

    for item in explanation["stress_drivers"]:
        fk = item["feature_key"]
        if fk in RAG_KNOWLEDGE_BASE and fk not in seen_keys:
            kb = RAG_KNOWLEDGE_BASE[fk]
            retrieved.append({
                "feature_key": fk,
                "feature_label": item["feature_label"],
                "staff_response": item["decoded_response"],
                "shap_impact": f"+{item['shap_value']:.4f}",
                "domain": kb["domain"],
                "recommended_action": kb["high_risk_guidance"],
            })
            seen_keys.add(fk)

    for item in explanation["protective_buffers"][:3]:
        fk = item["feature_key"]
        if fk in RAG_KNOWLEDGE_BASE and fk not in seen_keys:
            kb = RAG_KNOWLEDGE_BASE[fk]
            retrieved.append({
                "feature_key": fk,
                "feature_label": item["feature_label"],
                "staff_response": item["decoded_response"],
                "shap_impact": f"{item['shap_value']:.4f}",
                "domain": kb["domain"],
                "recommended_action": kb["maintenance_guidance"],
            })
            seen_keys.add(fk)

    return retrieved


def build_deterministic_rag_feedback(
    staff_row: pd.Series,
    explanation: Dict[str, Any],
    retrieved_docs: List[Dict[str, str]],
    selected_model_name: str = "Ridge Logistic Regression",
) -> str:
    staff_code = explanation["staff_code"]
    stress_label = explanation["predicted_label"]
    prob_high = explanation["probability_high_stress"] * 100
    rank_str = VALUE_DECODERS["academic_rank"].get(int(staff_row["academic_rank"]), "Academic Staff")
    exp_str = VALUE_DECODERS["teaching_experience"].get(int(staff_row["teaching_experience"]), "")
    qual_str = VALUE_DECODERS["educational_qualification"].get(int(staff_row["educational_qualification"]), "")
    pref_freq = VALUE_DECODERS["feedback_frequency"].get(int(staff_row["feedback_frequency"]), "Once a week")
    trust_str = VALUE_DECODERS["genai_system_trust"].get(int(staff_row["genai_system_trust"]), "Not sure")

    drivers = explanation["stress_drivers"]
    buffers = explanation["protective_buffers"]

    lines = [
        f"### Personalized Stress-Level Assessment & Generative AI Feedback Report (`{staff_code}`)",
        "",
        f"**Academic Profile:** {rank_str} ({qual_str}, {exp_str} teaching experience) — *Akwa Ibom State University (AKSU)*  ",
        f"**Assessed Stress Category ({selected_model_name}):** **{stress_label}** "
        f"(Estimated High-Stress Probability: **{prob_high:.1f}%**)  ",
        f"**Preferred Feedback Cadence:** {pref_freq} | **Gen-AI Trust Profile:** {trust_str}",
        "",
        "---",
        "",
        "#### 1. Stress Indicator Breakdown",
    ]

    if explanation["predicted_class"] == 1:
        lines.append(
            "Your questionnaire responses indicate an **elevated occupational stress profile (High Stress)**. "
            "The model identified the specific Pearson-selected occupational, technological, and behavioural "
            "factors contributing to your stress level:"
        )
    else:
        lines.append(
            "Your questionnaire responses indicate a **manageable occupational stress profile (Low Stress)**. "
            "The model identified both the protective factors maintaining your equilibrium and the mild "
            "pressure points to monitor so your stress remains low:"
        )

    lines.append("")
    if drivers:
        lines.append("**Primary Stress-Contributing Factors (Pushing Toward Higher Stress):**")
        for d in drivers[:4]:
            lines.append(
                f"- **{d['feature_label']}** (*{d['domain']}*): Your response (`{d['decoded_response']}`) "
                f"contributed a factor weight of **`+{d['weight_contribution']:.4f}`**."
            )
    if buffers:
        lines.append("")
        lines.append("**Protective & Stabilizing Factors (Buffering Against Stress):**")
        for b in buffers[:3]:
            lines.append(
                f"- **{b['feature_label']}** (*{b['domain']}*): Your response (`{b['decoded_response']}`) "
                f"acted as a protective buffer with a factor weight of **`{b['weight_contribution']:.4f}`**."
            )

    lines.extend([
        "",
        "---",
        "",
        "#### 2. Personalized Action Plan & Coping Recommendations",
    ])

    if explanation["predicted_class"] == 1:
        lines.append(
            "*High-Stress Coping & Workload Mitigation Recommendations tailored to your profile at Akwa Ibom State University:*"
        )
    else:
        lines.append(
            "*Low-Stress Maintenance & Preventive Wellbeing Recommendations tailored to your profile at Akwa Ibom State University:*"
        )
    lines.append("")

    for idx, doc in enumerate(retrieved_docs[:4], 1):
        lines.append(
            f"{idx}. **{doc['feature_label']}** *(Current Response: `{doc['staff_response']}`)*:  \n"
            f"   {doc['recommended_action']}"
        )

    lines.extend([
        "",
        "---",
        "",
        "#### 3. Weekly Micro-Habit & Adaptation Schedule",
        f"- **Scheduled Check-In:** Aligned with your stated preference (**{pref_freq}**), re-assess your workload "
        "and recovery indicators after implementing the top two adjustments above.",
        "- **Institutional & Confidential Support:** Occupational stress arises from an imbalance "
        "between demanding duties and available coping resources. Where administrative or teaching loads "
        "exceed standard departmental norms, use this assessment summary to discuss workload balancing with your "
        "Head of Department or consult the AKSU Medical & Counselling Unit.",
    ])

    return "\n".join(lines)


def generate_personalized_feedback(
    staff_row: pd.Series,
    explanation: Dict[str, Any],
    selected_model_name: str = "Ridge Logistic Regression",
    api_key: Optional[str] = None,
    use_live_gemini: bool = True,
) -> Dict[str, Any]:
    retrieved_docs = retrieve_rag_evidence(explanation)
    fallback_text = build_deterministic_rag_feedback(
        staff_row=staff_row,
        explanation=explanation,
        retrieved_docs=retrieved_docs,
        selected_model_name=selected_model_name,
    )

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not use_live_gemini or not resolved_key:
        return {
            "feedback_text": fallback_text,
            "generation_mode": "Generative AI Personalized Feedback Engine (Deterministic Synthesis)",
            "retrieved_evidence": retrieved_docs,
        }

    try:
        from google import genai

        client = genai.Client(api_key=resolved_key)

        rank_str = VALUE_DECODERS["academic_rank"].get(int(staff_row["academic_rank"]), "Academic Staff")
        exp_str = VALUE_DECODERS["teaching_experience"].get(int(staff_row["teaching_experience"]), "")
        qual_str = VALUE_DECODERS["educational_qualification"].get(int(staff_row["educational_qualification"]), "")
        pref_freq = VALUE_DECODERS["feedback_frequency"].get(int(staff_row["feedback_frequency"]), "Once a week")
        trust_str = VALUE_DECODERS["genai_system_trust"].get(int(staff_row["genai_system_trust"]), "Not sure")

        system_instruction = (
            "You are an empathetic, professional Occupational Wellbeing and Stress Management Specialist "
            "for Academic Staff at Akwa Ibom State University (AKSU), Ikot Akpaden, Nigeria. "
            "Your role is to translate a machine learning stress-level assessment and individual questionnaire responses "
            "into clear, respectful, non-judgmental, and actionable personalized feedback. "
            "Do not cite any literature, authors, or paper publications in the output text. "
            "Provide practical, high-impact coping strategies tailored specifically to the lecturer's reported workload, schedule, and wellbeing profile."
        )

        prompt_payload = f"""
Generate a comprehensive, personalized stress-management feedback report for Academic Staff Member `{explanation['staff_code']}` at Akwa Ibom State University.

### 1. Staff Profile & ML Assessment Output
- Academic Rank: {rank_str} ({qual_str}, {exp_str} experience)
- Selected ML Classifier: {selected_model_name}
- Predicted Stress Category: {explanation['predicted_label']} (Class {explanation['predicted_class']})
- High-Stress Probability: {explanation['probability_high_stress'] * 100:.1f}%
- Preferred Feedback Frequency: {pref_freq}
- Trust in Gen-AI System: {trust_str}

### 2. Top Stress-Driving Factors (Pushing Toward High Stress)
{json.dumps(explanation['stress_drivers'][:4], indent=2)}

### 3. Top Protective / Stabilizing Factors (Pushing Toward Low Stress)
{json.dumps(explanation['protective_buffers'][:3], indent=2)}

### 4. Tailored Coping Strategies & Intervention Guidelines
{json.dumps(retrieved_docs, indent=2)}

Please structure your response in clean Markdown with:
1. **Personalized Assessment Summary & Empathy Statement** (tailored to whether they are in HIGH STRESS or LOW STRESS and acknowledging their Gen-AI trust level).
2. **What Is Driving Your Stress Profile (Explanation in Plain Language)**.
3. **Actionable Recommendations for Akwa Ibom State University Context** (specific steps for teaching, research, administration, workday breaks, and health coping, without citing any literature or papers).
4. **Follow-Up & Adaptation Plan** (aligned with their preferred feedback schedule of `{pref_freq}`).
"""

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            system_instruction=system_instruction,
            input=prompt_payload,
        )

        if interaction.output_text:
            return {
                "feedback_text": interaction.output_text,
                "generation_mode": "Live Gemini 3.8 Flash Engine",
                "retrieved_evidence": retrieved_docs,
            }
    except Exception as exc:
        return {
            "feedback_text": fallback_text,
            "generation_mode": f"Generative AI Personalized Feedback Engine (Fallback: {type(exc).__name__})",
            "retrieved_evidence": retrieved_docs,
        }

    return {
        "feedback_text": fallback_text,
        "generation_mode": "Generative AI Personalized Feedback Engine",
        "retrieved_evidence": retrieved_docs,
    }
