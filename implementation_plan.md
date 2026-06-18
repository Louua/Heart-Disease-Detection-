# Implementation Plan - Local Streamlit App Deployment (Notebook Untouched)

This plan outlines the steps to deploy and run the Streamlit web application locally in the `LouaML` workspace, wired to the heart disease prediction model, without modifying the existing `Model.ipynb` notebook.

---

## User Review Required

> [!IMPORTANT]
> - **Notebook Untouched**: `Model.ipynb` will remain completely unchanged.
> - **Model Generation**: Since the notebook has not been executed locally and its pickles do not exist, we will use a small helper script (`train_local.py`) to download the dataset and execute the exact training pipeline from the notebook to generate `deployment_model.pkl` and `deployment_preprocessor.pkl`.
> - **Streamlit Design**: The Streamlit interface will use premium, glassmorphic styling (dark theme, colored cards, smooth animations, and progress/gauge indicators) and be fully interactive.

---

## Proposed Changes

### LouaML Workspace

---

#### [NEW] [train_local.py](file:///home/sary/LouaML/train_local.py)
A training script that reproduces the model pipeline from the notebook:
1. Uses `kagglehub` to download the `benafialoua/heartdisease` dataset locally.
2. Cleans, preprocesses, and engineers the features.
3. Fits the XGBoost and Logistic Regression models.
4. Saves `deployment_model.pkl` and `deployment_preprocessor.pkl` in the `app/` folder.

#### [NEW] [app.py](file:///home/sary/LouaML/app/streamlit_app.py)
The premium Streamlit application file:
- Loads the generated model and preprocessor directly using `joblib`.
- Features a premium layout with tabs separating clinical and diagnostic metrics.
- Uses custom CSS/HTML for a glassmorphic dashboard design, including a color-coded risk gauge.
- Provides an interactive decision support tool with a downloadable report.

#### [NEW] [requirements.txt](file:///home/sary/LouaML/requirements.txt)
Python packages required for the project:
`streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `kagglehub`.

#### [NEW] [run_local.sh](file:///home/sary/LouaML/run_local.sh)
A script to:
1. Set up a Python virtual environment (`venv`).
2. Install dependencies.
3. Run `train_local.py` to generate the model.
4. Launch the Streamlit application.

---

## Verification Plan

### Manual Verification
- We will execute `./run_local.sh` to initialize the environment and launch the app.
- We will use the browser tool to interact with the Streamlit app, test multiple patient profiles (low risk vs. high risk), and confirm that the predictions are computed correctly and the UI looks premium.
