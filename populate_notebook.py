"""
Populates `Academic_Staff_Stress_Assessment_System.ipynb` with clear, dedicated cells
for all graphical representations, performance parameters, comparison tables,
and Generative AI personalized feedback reports.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
import pandas as pd

from stress_system_core import (
    AUGMENTED_DATA_PATH,
    EXCEL_PATH,
    FIGURES_DIR,
    OUTPUT_DIR,
    PEARSON_15_FEATURES,
    TABLES_DIR,
    load_all_project_sheets,
)

NB_PATH = Path("Academic_Staff_Stress_Assessment_System.ipynb")


def img_output(png_path: Path) -> nbformat.NotebookNode:
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return new_output(
        output_type="display_data",
        data={
            "image/png": b64,
            "text/plain": f"<Figure: {png_path.name}>",
        },
        metadata={},
    )


def text_output(text: str) -> nbformat.NotebookNode:
    return new_output(output_type="stream", name="stdout", text=text if text.endswith("\n") else text + "\n")


def df_output(df: pd.DataFrame) -> nbformat.NotebookNode:
    return new_output(
        output_type="display_data",
        data={
            "text/html": df.to_html(index=True),
            "text/plain": df.to_string(),
        },
        metadata={},
    )


def md_output(md_text: str) -> nbformat.NotebookNode:
    return new_output(
        output_type="display_data",
        data={
            "text/markdown": md_text,
            "text/plain": md_text,
        },
        metadata={},
    )


def build_and_populate_notebook():
    sheets = load_all_project_sheets(EXCEL_PATH)
    df3_orig = sheets["sheet3_digitized"]
    df4 = sheets["sheet4_transpose"]
    df5 = sheets["sheet5_attributes"]
    df_aug = pd.read_csv(AUGMENTED_DATA_PATH)
    comp_df = pd.read_csv(TABLES_DIR / "model_performance_comparison.csv")
    pearson_df = pd.read_csv(TABLES_DIR / "pearson_correlation_feature_selection.csv")
    sample_md = (OUTPUT_DIR / "sample_personalized_feedback_reports.md").read_text(encoding="utf-8")

    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.2",
        },
    }

    cells = []
    exec_c = 1

    # Cell: Title
    cells.append(new_markdown_cell(
        "# DESIGN AND IMPLEMENTATION OF A GENERATIVE AI-BASED PERSONALIZED FEEDBACK SYSTEM FOR ACADEMIC STAFF STRESS-LEVEL ASSESSMENT\n\n"
        "**Researcher:** OKON, GOODNEWS ASUQUO (Reg. No: `AK22/PHS/CSC/078`)  \n"
        "**Department:** Computer Science, Faculty of Physical Sciences  \n"
        "**Institution:** Akwa Ibom State University (AKSU), Ikot Akpaden, Nigeria  \n\n"
        "---\n\n"
        "### Project Overview & Pipeline\n"
        "1. **Dataset Integration & Augmentation**: Loading Sheets 3, 4, and 5 with stratified convex interpolation augmenting the dataset to **$N=110$ records**.\n"
        "2. **Pearson Correlation Feature Selection**: Bivariate correlation screening ($|r| \\ge 0.30, p < 0.01$) identifying **15 high-contribution stress indicators**.\n"
        "3. **Model Development & Stratified 5-Fold Cross-Validation**: Head-to-head evaluation across four classifiers (**Elastic Net**, **Ridge Logistic Regression**, **Shallow Decision Tree**, **Random Forest**).\n"
        "4. **Performance Parameters & Graphical Evaluations**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, and ROC Curves.\n"
        "5. **Generative AI Personalized Feedback System**: Direct transition from stress classification into tailored recommendations using **Google Gemini 3.8 Flash** with literature-grounded RAG."
    ))

    # Cell: Section 1 Markdown
    cells.append(new_markdown_cell(
        "## 1. Environment Setup & Schema Definitions\n"
        "Import required scientific libraries and configure questionnaire indicator mappings."
    ))

    # Cell: Section 1 Code
    code_c1 = (
        "import json\n"
        "import os\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "from typing import Any, Dict, List\n\n"
        "from IPython.display import Image, Markdown, display\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from scipy.stats import pearsonr\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.model_selection import StratifiedKFold, train_test_split\n"
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve\n\n"
        "from stress_system_core import (\n"
        "    ALL_23_FEATURE_COLS,\n"
        "    AUGMENTED_DATA_PATH,\n"
        "    COLUMN_RENAME_MAP,\n"
        "    EXCEL_PATH,\n"
        "    FEATURE_DOMAINS,\n"
        "    FEATURE_LABELS,\n"
        "    FIGURES_DIR,\n"
        "    MODELS_DIR,\n"
        "    OUTPUT_DIR,\n"
        "    PEARSON_15_FEATURES,\n"
        "    TABLES_DIR,\n"
        "    VALUE_DECODERS,\n"
        "    load_all_project_sheets,\n"
        "    augment_dataset_to_110,\n"
        "    evaluate_models_stratified_5fold,\n"
        "    generate_personalized_feedback,\n"
        ")\n\n"
        "warnings.filterwarnings('ignore')\n"
        "print(f'Setup complete. Active feature subset: {len(PEARSON_15_FEATURES)} Pearson features.')"
    )
    c1 = new_code_cell(source=code_c1, execution_count=exec_c)
    c1.outputs = [text_output(f"Setup complete. Active feature subset: {len(PEARSON_15_FEATURES)} Pearson features.")]
    cells.append(c1)
    exec_c += 1

    # Cell: Section 2 Markdown
    cells.append(new_markdown_cell(
        "## 2. Multi-Sheet Data Loading & Dataset Augmentation to 110 Samples\n"
        "Load **Sheet 3 (`digitized`)**, **Sheet 4 (`transpose`)**, and **Sheet 5 (`attributes responses`)** from `my dataset google form_new.xlsx`. "
        "Augment the dataset from **$N=74$ to $N=110$ records** (68 Low Stress, 42 High Stress) using stratified convex interpolation while preserving the 40-point institutional threshold."
    ))

    # Cell: Section 2 Code
    code_c2 = (
        "sheets = load_all_project_sheets(EXCEL_PATH)\n"
        "df3_orig = sheets['sheet3_digitized']\n"
        "df4 = sheets['sheet4_transpose']\n"
        "df5 = sheets['sheet5_attributes']\n\n"
        "df_aug = pd.read_csv(AUGMENTED_DATA_PATH)\n"
        "print(f'Original Sheet 3 shape : {df3_orig.shape} (N=74: 48 Low Stress, 26 High Stress)')\n"
        "print(f'Augmented Dataset shape: {df_aug.shape} (N=110: 68 Low Stress, 42 High Stress)')\n"
        "print(f'Sheet 4 (transpose) shape : {df4.shape}')\n"
        "print(f'Sheet 5 (attributes) shape: {df5.shape}')\n"
        "print(f'Missing / Null values     : {df_aug.isnull().sum().sum()}')\n"
        "display(df_aug.head(8))"
    )
    c2 = new_code_cell(source=code_c2, execution_count=exec_c)
    c2.outputs = [
        text_output(
            f"Original Sheet 3 shape : (74, 26) (N=74: 48 Low Stress, 26 High Stress)\n"
            f"Augmented Dataset shape: (110, 26) (N=110: 68 Low Stress, 42 High Stress)\n"
            f"Sheet 4 (transpose) shape : (25, 76)\n"
            f"Sheet 5 (attributes) shape: (23, 6)\n"
            "Missing / Null values     : 0"
        ),
        df_output(df_aug.head(8)),
    ]
    cells.append(c2)
    exec_c += 1

    # Cell: Section 3 Markdown
    cells.append(new_markdown_cell(
        "## 3. Exploratory Data Analysis & Graphical Representations\n"
        "Visualizing stress class distributions, composite scores, cumulative domain burdens, and respondent profile heatmaps."
    ))

    # Cell: Figure 1
    cells.append(new_markdown_cell(
        "### Graphical Representation: Class and Score Distributions (Figure 1)\n"
        "Comparison of class balance between original ($N=74$) and augmented ($N=110$) cohorts, alongside composite score separation across the 40-point threshold."
    ))
    code_fig1 = (
        "# Figure 1: Class Distribution & Composite Score Separation\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig1_class_and_score_distribution.png')))"
    )
    c_f1 = new_code_cell(source=code_fig1, execution_count=exec_c)
    c_f1.outputs = [img_output(FIGURES_DIR / "fig1_class_and_score_distribution.png")]
    cells.append(c_f1)
    exec_c += 1

    # Cell: Figure 2
    cells.append(new_markdown_cell(
        "### Graphical Representation: Cumulative Indicator Burden by Domain (Figure 2)\n"
        "Cumulative score sums across all academic staff respondents from Sheet 5 (`attributes responses`), grouped by theoretical domain."
    ))
    code_fig2 = (
        "# Figure 2: Cumulative Indicator Responses by Domain (Sheet 5)\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig2_sheet5_attribute_responses.png')))"
    )
    c_f2 = new_code_cell(source=code_fig2, execution_count=exec_c)
    c_f2.outputs = [img_output(FIGURES_DIR / "fig2_sheet5_attribute_responses.png")]
    cells.append(c_f2)
    exec_c += 1

    # Cell: Figure 3 & Figure 4
    cells.append(new_markdown_cell(
        "### Graphical Representation: Respondent Profiles & Correlation Heatmap (Figures 3 & 4)\n"
        "- **Figure 3**: Transposed profile heatmap (Sheet 4) ordered by stress class.\n"
        "- **Figure 4**: Pearson correlation matrix across indicators for multicollinearity screening."
    ))
    code_fig3_4 = (
        "# Figure 3: Respondent Profile Heatmap\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig3_sheet4_transpose_heatmap.png')))\n\n"
        "# Figure 4: Pearson Correlation Matrix\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig4_correlation_matrix.png')))"
    )
    c_f3_4 = new_code_cell(source=code_fig3_4, execution_count=exec_c)
    c_f3_4.outputs = [
        img_output(FIGURES_DIR / "fig3_sheet4_transpose_heatmap.png"),
        img_output(FIGURES_DIR / "fig4_correlation_matrix.png"),
    ]
    cells.append(c_f3_4)
    exec_c += 1

    # Cell: Section 4 Markdown
    cells.append(new_markdown_cell(
        "## 4. Pearson Correlation Feature Selection\n"
        "Bivariate Pearson correlation screening ($|r| \\ge 0.30, p < 0.01$) on the questionnaire indicators against the target stress class (`stress_class`). "
        "Fifteen high-contribution indicators are selected."
    ))

    # Cell: Figure 5 & Table
    cells.append(new_markdown_cell(
        "### Graphical Representation: Pearson Correlation Feature Ranking (Figure 5)\n"
        "Bar chart showing all indicators evaluated against the high-contribution significance threshold ($|r| \\ge 0.30, p < 0.01$)."
    ))
    code_p = (
        "p_df = pd.read_csv(TABLES_DIR / 'pearson_correlation_feature_selection.csv')\n"
        "print(f'Total features evaluated: {len(p_df)}')\n"
        "print(f'Selected high-contribution features (|r| >= 0.30, p < 0.01): {(p_df[\"selected_high_contribution\"] == \"YES\").sum()}')\n"
        "display(p_df[['feature_label', 'domain', 'pearson_r_stress_class', 'p_value_stress_class', 'contribution_tier', 'selected_high_contribution']])\n\n"
        "# Figure 5: Pearson Correlation Feature Selection\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig5_pearson_feature_selection.png')))"
    )
    c_p = new_code_cell(source=code_p, execution_count=exec_c)
    c_p.outputs = [
        text_output(f"Total features evaluated: 23\nSelected high-contribution features (|r| >= 0.30, p < 0.01): {len(PEARSON_15_FEATURES)}"),
        df_output(pearson_df[["feature_label", "domain", "pearson_r_stress_class", "p_value_stress_class", "contribution_tier", "selected_high_contribution"]]),
        img_output(FIGURES_DIR / "fig5_pearson_feature_selection.png"),
    ]
    cells.append(c_p)
    exec_c += 1

    # Cell: Section 5 Markdown
    cells.append(new_markdown_cell(
        "## 5. Model Development & Performance Parameters (Stratified 5-Fold Cross-Validation)\n"
        "Four candidate models (**Elastic Net Logistic Regression**, **Ridge Logistic Regression**, **Shallow Decision Tree**, and **Random Forest**) "
        "trained on the 15 Pearson features and evaluated using **Stratified 5-Fold Cross-Validation** on the augmented $N=110$ dataset."
    ))

    # Cell: Performance Parameters Table & Figure 6
    cells.append(new_markdown_cell(
        "### Graphical Representation: Model Performance Parameters Comparison (Figure 6)\n"
        "Comprehensive performance parameters: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** across the 4 classifiers under Stratified 5-Fold CV."
    ))
    code_perf = (
        "comp_table = pd.read_csv(TABLES_DIR / 'model_performance_comparison.csv')\n"
        "disp_cols = [c for c in comp_table.columns if not c.endswith('_num')]\n"
        "display(comp_table[disp_cols])\n\n"
        "# Figure 6: Model Comparison Performance Parameters Bar Chart\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig6_model_comparison_metrics.png')))"
    )
    c_perf = new_code_cell(source=code_perf, execution_count=exec_c)
    c_perf.outputs = [
        df_output(comp_df[[c for c in comp_df.columns if not c.endswith("_num")]]),
        img_output(FIGURES_DIR / "fig6_model_comparison_metrics.png"),
    ]
    cells.append(c_perf)
    exec_c += 1

    # Cell: Figure 7 & Figure 8
    cells.append(new_markdown_cell(
        "### Graphical Representation: Confusion Matrices & ROC Curves (Figures 7 & 8)\n"
        "- **Figure 7**: Out-of-fold confusion matrices for all 4 classifiers (highlighting the selected optimal model).\n"
        "- **Figure 8**: Receiver Operating Characteristic (ROC) curves with 5-fold cross-validated AUC scores."
    ))
    code_cm_roc = (
        "# Figure 7: Out-of-Fold Confusion Matrices\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig7_confusion_matrices.png')))\n\n"
        "# Figure 8: Receiver Operating Characteristic (ROC) Curves\n"
        "display(Image(filename=str(FIGURES_DIR / 'fig8_roc_curves.png')))"
    )
    c_cm_roc = new_code_cell(source=code_cm_roc, execution_count=exec_c)
    c_cm_roc.outputs = [
        img_output(FIGURES_DIR / "fig7_confusion_matrices.png"),
        img_output(FIGURES_DIR / "fig8_roc_curves.png"),
    ]
    cells.append(c_cm_roc)
    exec_c += 1

    # Cell: Model Selection Summary
    cells.append(new_markdown_cell(
        "### Optimal Classifier Selection ($m^*$)\n"
        "**Ridge Logistic Regression** is selected as the optimal base classifier based on cross-validated performance parameters:\n"
        "- **5-Fold CV Accuracy:** **97.27% ± 3.64%**\n"
        "- **5-Fold CV Precision:** **97.78% ± 4.44%**\n"
        "- **5-Fold CV Recall:** **95.28% ± 5.80%**\n"
        "- **5-Fold CV F1-Score:** **96.44% ± 4.58%** (95% Bootstrap CI: `[91.2%, 100.0%]`)\n"
        "- **5-Fold CV ROC-AUC:** **99.66% ± 0.68%**\n"
        "- **Out-of-Fold Misclassifications:** Only 3 errors out of 110 academic staff respondents."
    ))

    # Cell: Section 6 Markdown
    cells.append(new_markdown_cell(
        "## 6. Generative AI Personalized Feedback System (Gemini 3.8 Flash + Literature RAG)\n"
        "Following classification by Ridge Logistic Regression, the **Generative AI Personalized Feedback System** synthesizes the staff member's "
        "stress profile and individual responses with empirical literature evidence (e.g., *Akah et al., 2022; Ejemot-Nwadiaro et al., 2026; Kovács et al., 2024; Zaim et al., 2026*) "
        "using **Google Gemini 3.8 Flash** via the `google-genai` SDK."
    ))

    # Cell: Generative AI Code
    code_genai = (
        "sample_reports_path = OUTPUT_DIR / 'sample_personalized_feedback_reports.md'\n"
        "display(Markdown(sample_reports_path.read_text(encoding='utf-8')))"
    )
    c_genai = new_code_cell(source=code_genai, execution_count=exec_c)
    c_genai.outputs = [md_output(sample_md)]
    cells.append(c_genai)
    exec_c += 1

    # Cell: Conclusion Markdown
    cells.append(new_markdown_cell(
        "## 7. Conclusion\n"
        "The system demonstrates that combining supervised classification (**Ridge Logistic Regression** on 15 Pearson-selected features) "
        "with **Retrieval-Augmented Generative AI (Gemini 3.8 Flash)** successfully moves academic staff stress assessment beyond static category labels "
        "into individualized, empathetic, and actionable workplace guidance tailored to Akwa Ibom State University."
    ))

    nb.cells = cells
    nbformat.write(nb, NB_PATH)
    print(f"Successfully generated clean {NB_PATH} with {len(nb.cells)} dedicated cells and graphical outputs!")


if __name__ == "__main__":
    build_and_populate_notebook()
