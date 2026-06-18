import os
import shutil
import kagglehub
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, make_scorer
import xgboost as xgb

# 1. Load dataset
print("Loading heart disease dataset...")
local_csv_path = "data/heart_disease.csv"

if os.path.exists("heart_disease.csv"):
    os.makedirs("data", exist_ok=True)
    shutil.copy("heart_disease.csv", local_csv_path)
    print(f"Dataset copied from root to: {local_csv_path}")
else:
    print("Downloading heart disease dataset from Kaggle...")
    download_path = kagglehub.dataset_download('benafialoua/heartdisease')
    print(f"Dataset downloaded to: {download_path}")

    # Find the heart_disease.csv file
    csv_filename = "heart_disease.csv"
    csv_path = None
    for root, dirs, files in os.walk(download_path):
        if csv_filename in files:
            csv_path = os.path.join(root, csv_filename)
            break

    if not csv_path:
        raise FileNotFoundError("Could not find heart_disease.csv in downloaded files.")

    # Copy the dataset to the local data directory for reproducibility
    os.makedirs("data", exist_ok=True)
    shutil.copy(csv_path, local_csv_path)
    print(f"Dataset copied from download to: {local_csv_path}")

# Load dataset
df = pd.read_csv(local_csv_path)

# 2. Cleaning & Preparation
critical_cols = ['age', 'trestbps', 'chol', 'thalach']
for col in critical_cols:
    df = df[df[col].notna()]
df = df[np.floor(df['age']) == df['age']]
df = df[np.floor(df['trestbps']) == df['trestbps']]
df = df[np.floor(df['chol']) == df['chol']]
df = df[np.floor(df['thalach']) == df['thalach']]
for col in critical_cols:
    df[col] = df[col].astype(int)
df = df[(df['chol'] >= 100) & (df['chol'] <= 600)]
df = df[(df['trestbps'] >= 80) & (df['trestbps'] <= 200)]
df = df[(df['oldpeak'] >= 0) & (df['oldpeak'] <= 6)]

X = df.drop(columns=['num', 'target_binary'])
y = df['target_binary']

# 3. Preprocessing
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal', 'sex', 'fbs', 'exang']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# We fit/transform on train, but remember the ColumnTransformer outputs a numpy array
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# 5. Model Training and Tuning (using Recall as metric)
scorer = make_scorer(recall_score)

# 5.1 Logistic Regression Tuning
print("Tuning Logistic Regression...")
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=5, scoring=scorer, n_jobs=-1)
grid_lr.fit(X_train_proc, y_train)
best_lr = grid_lr.best_estimator_
lr_recall = recall_score(y_test, best_lr.predict(X_test_proc))
print(f"Logistic Regression Recall on Test Set: {lr_recall:.4f}")

# 5.2 XGBoost Tuning
print("Tuning XGBoost...")
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
param_dist_xgb = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1.0, 1.5, 2.0]
}
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb, n_iter=20, cv=5, scoring=scorer, random_state=42, n_jobs=-1)
random_search.fit(X_train_proc, y_train)
best_xgb = random_search.best_estimator_
xgb_recall = recall_score(y_test, best_xgb.predict(X_test_proc))
print(f"XGBoost Recall on Test Set: {xgb_recall:.4f}")

# 6. Select Best Model
if lr_recall >= xgb_recall:
    print(f"Selecting Logistic Regression as final model (Recall: {lr_recall:.4f})")
    final_model = best_lr
else:
    print(f"Selecting XGBoost as final model (Recall: {xgb_recall:.4f})")
    final_model = best_xgb

# Refit the preprocessor on the whole cleaned dataset (as done in Cell 7 deployment step)
X_clean_proc = preprocessor.fit_transform(X)
final_model.fit(X_clean_proc, y)

# 7. Save Deployment Files
os.makedirs("app", exist_ok=True)
joblib.dump(final_model, 'app/deployment_model.pkl')
joblib.dump(preprocessor, 'app/deployment_preprocessor.pkl')
print("Model and preprocessor saved successfully in app/ directory.")
