# -*- coding: utf-8 -*-
"""CSE427_Project.ipynb


# **CSE427 : MACHINE LEARNING [LAB PROJECT]**

--------------------------------------------------------------------------------

# **Multi-Level Explainable Machine Learning for Motorcycle Accident Severity Prediction: A Behavioral Risk Profiling Approach**

--------------------------------------------------------------------------------

# **Group:**

1. **Ahnaf Rahman Brinto [-]**
2. **Fayaz Bin Faruk [-]**

--------------------------------------------------------------------------------

# **LIBRARY IMPORTS**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from google.colab import drive
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, silhouette_score
)
from sklearn.calibration import calibration_curve
from scipy import stats

"""# **DATASET**"""

file_link = 'https://drive.google.com/file/d/1sCJBh-3frXr7trFQj2TsQaaygj8avCzX/view?usp=sharing'
id = file_link.split("/")[-2]

new_link = f'https://drive.google.com/uc?id={id}'
df = pd.read_csv(new_link)

"""# **EXPLORATORY DATA ANALYSIS (EDA)**"""

shape = df.shape # number of observations and features
print(f"Rows: {shape[0]}")
print(f"Columns: {shape[1]}")
totalCells = shape[0] * shape[1]
print(f"Total Cells: {totalCells}")

df.head() # preview a sample

df.info()

dtype_counts = df.dtypes.value_counts()

for dtype, count in dtype_counts.items():
  print(f"{dtype}: {count} columns")

df.describe(include='all')

df.describe(include='object') #some additional information on categorical features

df.describe().T  #This method gives a statistical summary of the DataFrame (Transpose)

df[df.duplicated()] #check duplicated rows

df.nunique()

df.isnull().sum()

missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Column': df.columns,
                           'Missing_Count': missing_count.values,
                           'Missing_Percent': missing_percent.values})

missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:

        print(f"Missing Values:")
        display(missing_df)

        # missing value visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # bar plot
        axes[0].barh(missing_df['Column'], missing_df['Missing_Percent'],color='red')
        axes[0].set_xlabel('Missing Percentage (%)')
        axes[0].set_title('Missing Values by Column')
        axes[0].grid(axis='x', alpha=0.8)

        # Heatmap
        plt.sca(axes[1])
        sns.heatmap(df.isnull(), cbar=True, cmap='viridis',
                    yticklabels=False, cbar_kws={'label': 'Missing Data'})
        axes[1].set_title('Missing Values Heatmap')

        plt.tight_layout()
        plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns: {len(numeric_cols)} ")
print(f"Columns: {', '.join(numeric_cols)}")

n_cols = min(3, len(numeric_cols))
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

for idx, col in enumerate(numeric_cols):
  axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', color='grey', alpha=0.7)
  axes[idx].set_title(f'Distribution of {col}')
  axes[idx].set_xlabel(col)
  axes[idx].set_ylabel('Frequency')
  axes[idx].grid(alpha=0.3)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col].dropna(), vert=True)
    axes[idx].set_title(f'Box Plot - {col}')
    axes[idx].set_ylabel(col)
    axes[idx].grid(alpha=0.8)

import math
import matplotlib.pyplot as plt

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
top_n = 10

print(f"Categorical columns: {len(categorical_cols)}")

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(f"Unique values: {df[col].nunique()}")
    print(f"Most common values:")
    print(df[col].value_counts().head(top_n))

plot_cols = [col for col in categorical_cols if df[col].nunique() <= 20]

if len(plot_cols) > 0:
    n_cols = 2   # number of charts per row
    n_rows = math.ceil(len(plot_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        value_counts = df[col].value_counts().head(top_n)

        ax.bar(value_counts.index, value_counts.values,
               color='blue', edgecolor='black')
        ax.set_title(f'Distribution of {col} (Top {min(top_n, len(value_counts))})')
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.8)

    plt.tight_layout()
    plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns

corr_matrix = df[numeric_cols].corr()
# display(corr_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns

outlier_summary = []

for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(df)) * 100

        outlier_summary.append({

            'Column': col,
            'Outlier_Count': outlier_count,
            'Outlier_Percent': outlier_percent,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound

        })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df = outlier_df[outlier_df['Outlier_Count'] > 0].sort_values('Outlier_Count', ascending=False)

if len(outlier_df) > 0:
        print(f"Columns with Outliers:")
        display(outlier_df)

"""
# **TRAIN-TEST SPLIT**"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

# copy of the original data
df_clean = df.copy()

# duplicates removal
dup_count = df_clean.duplicated().sum()
if dup_count > 0:
    df_clean.drop_duplicates(inplace=True)
    print(f"Removed {dup_count} duplicate rows")

# rows with missing values --> remove
initial_rows = df_clean.shape[0]
df_clean.dropna(inplace=True)
dropped_rows = initial_rows - df_clean.shape[0]
if dropped_rows > 0:
    print(f"Removed {dropped_rows} rows with NaN values.")

# features and target Separation
X_raw = df_clean.drop('Accident_Severity', axis=1)
y_raw = df_clean['Accident_Severity']

# 80-20 split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

print(f"Training set: {X_train_raw.shape[0]} samples")
print(f"Test set: {X_test_raw.shape[0]} samples")

"""# **DATA PREPROCESSING (On Training Set)**"""

# copies for preprocessing
X_train_processed = X_train_raw.copy()
X_test_processed = X_test_raw.copy()
y_train_processed = y_train_raw.copy()
y_test_processed = y_test_raw.copy()

# Target distribution Visuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=pd.DataFrame({'Accident_Severity': y_train_processed}),
              x='Accident_Severity', palette='viridis')
plt.title('Training Set - Accident Severity Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Accident Severity')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(data=pd.DataFrame({'Accident_Severity': y_test_processed}),
              x='Accident_Severity', palette='plasma')
plt.title('Test Set - Accident Severity Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Accident Severity')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 1. Age categories
X_train_processed['Age_Category'] = pd.cut(
    X_train_processed['Biker_Age'],
    bins=[0, 25, 35, 50, 100],
    labels=['Young', 'Adult', 'Middle-aged', 'Senior']
)
X_test_processed['Age_Category'] = pd.cut(
    X_test_processed['Biker_Age'],
    bins=[0, 25, 35, 50, 100],
    labels=['Young', 'Adult', 'Middle-aged', 'Senior']
)

# 2. Behavioral Risk Score
X_train_processed['Behavioral_Risk_Score'] = (
    X_train_processed['Talk_While_Riding'].map({'Yes': 1, 'No': 0}).fillna(0) +
    X_train_processed['Smoke_While_Riding'].map({'Yes': 1, 'No': 0}).fillna(0) +
    X_train_processed['Wearing_Helmet'].map({'No': 1, 'Yes': 0}).fillna(0) +
    X_train_processed['Valid_Driving_License'].map({'No': 1, 'Yes': 0}).fillna(0) +
    X_train_processed['Biker_Alcohol'] * 2
)

X_test_processed['Behavioral_Risk_Score'] = (
    X_test_processed['Talk_While_Riding'].map({'Yes': 1, 'No': 0}).fillna(0) +
    X_test_processed['Smoke_While_Riding'].map({'Yes': 1, 'No': 0}).fillna(0) +
    X_test_processed['Wearing_Helmet'].map({'No': 1, 'Yes': 0}).fillna(0) +
    X_test_processed['Valid_Driving_License'].map({'No': 1, 'Yes': 0}).fillna(0) +
    X_test_processed['Biker_Alcohol'] * 2
)

# 3. Speed Violation
X_train_processed['Speed_Violation'] = (
    X_train_processed['Bike_Speed'] > X_train_processed['Speed_Limit']
).astype(int)
X_test_processed['Speed_Violation'] = (
    X_test_processed['Bike_Speed'] > X_test_processed['Speed_Limit']
).astype(int)

# 4. Time Category
time_mapping = {
    'Morning': 'Day',
    'Afternoon': 'Day',
    'Noon': 'Day',
    'Evening': 'Evening',
    'Night': 'Night'
}
X_train_processed['Time_Category'] = X_train_processed['Time_of_Day'].map(time_mapping)
X_test_processed['Time_Category'] = X_test_processed['Time_of_Day'].map(time_mapping)

print("Feature engineering completed")

# featureDrop
features_to_drop = [
    'Speed_Limit',  # Speed_Violation captures this
    'Bike_Speed'    # Speed_Violation captures this
]

X_train_processed.drop(columns=features_to_drop, inplace=True)
X_test_processed.drop(columns=features_to_drop, inplace=True)
print(f"Dropped {len(features_to_drop)} redundant features")

# ENCODING
categorical_cols = X_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
    X_test_processed[col] = X_test_processed[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    label_encoders[col] = le

print(f"Encoded {len(categorical_cols)} categorical features")

# Encode target
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train_processed)
y_test = target_encoder.transform(y_test_processed)

print(f"Target encoded: {dict(enumerate(target_encoder.classes_))}")

# SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train_processed.columns,
    index=X_train_processed.index
)
X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test_processed.columns,
    index=X_test_processed.index
)

"""# **`DATA SUMMARY`**"""

print("Final Dataset Summary:")
print("="*50)
print(f"Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
print(f"Test set: {X_test_scaled.shape[0]} samples, {X_test_scaled.shape[1]} features")
print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nTest class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

"""# **`BASELINE MODELS IMPLEMENTATION`**"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import time

# results
model_results = {}

# 1. Logistic Regression
print("1. Logistic Regression...")
start_time = time.time()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)

lr_time = time.time() - start_time

model_results['Logistic Regression'] = {
    'model': lr_model,
    'predictions': lr_pred,
    'probabilities': lr_pred_proba,
    'training_time': lr_time
}

print(f"Training completed in {lr_time:.2f} seconds")

# 2. Decision Tree
print("2. Decision Tree...")
start_time = time.time()

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_pred_proba = dt_model.predict_proba(X_test_scaled)

dt_time = time.time() - start_time

model_results['Decision Tree'] = {
    'model': dt_model,
    'predictions': dt_pred,
    'probabilities': dt_pred_proba,
    'training_time': dt_time
}

print(f"Training completed in {dt_time:.2f} seconds")

# 3. Random Forest
print("3. Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)

rf_time = time.time() - start_time

model_results['Random Forest'] = {
    'model': rf_model,
    'predictions': rf_pred,
    'probabilities': rf_pred_proba,
    'training_time': rf_time
}

print(f"Training completed in {rf_time:.2f} seconds")

"""# **`ADVANCED MODELS`**"""

# 4. XGBoost with Hyperparameter Tuning
print("4. Training XGBoost with Hyperparameter Tuning...")
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

start_time = time.time()

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Base model
xgb_base = xgb.XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

# Grid search with 3-fold cross-validation
print("Performing grid search...")
grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best model
xgb_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)

xgb_time = time.time() - start_time

model_results['XGBoost'] = {
    'model': xgb_model,
    'predictions': xgb_pred,
    'probabilities': xgb_pred_proba,
    'training_time': xgb_time,
    'best_params': grid_search.best_params_
}

print(f"Training completed in {xgb_time:.2f} seconds")

# 5. Multilayer Perceptrons (MLP)
print("5. Training Multilayer Perceptrons (MLP)...")
from sklearn.neural_network import MLPClassifier

start_time = time.time()

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True
)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)
mlp_pred_proba = mlp_model.predict_proba(X_test_scaled)

mlp_time = time.time() - start_time

model_results['Multilayer Perceptrons'] = {
    'model': mlp_model,
    'predictions': mlp_pred,
    'probabilities': mlp_pred_proba,
    'training_time': mlp_time
}

print(f"Training completed in {mlp_time:.2f} seconds")

# 6. Support Vector Machine (SVM)
print("6. Training Support Vector Machine (SVM)...")
from sklearn.svm import SVC

start_time = time.time()

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,  # Enable probability estimates
    random_state=42,
    max_iter=1000
)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_pred_proba = svm_model.predict_proba(X_test_scaled)

svm_time = time.time() - start_time

model_results['SVM'] = {
    'model': svm_model,
    'predictions': svm_pred,
    'probabilities': svm_pred_proba,
    'training_time': svm_time
}

print(f"Training completed in {svm_time:.2f} seconds")

# 7. Gradient Boosting Classifier
print("7. Training Gradient Boosting Classifier...")
from sklearn.ensemble import GradientBoostingClassifier

start_time = time.time()

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    subsample=0.8
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_pred_proba = gb_model.predict_proba(X_test_scaled)

gb_time = time.time() - start_time

model_results['Gradient Boosting'] = {
    'model': gb_model,
    'predictions': gb_pred,
    'probabilities': gb_pred_proba,
    'training_time': gb_time
}

print(f"Training completed in {gb_time:.2f} seconds")

"""# **`MODEL EVALUATION AND COMPARISON`**"""

comparison_data = []

for model_name, results in model_results.items():
    y_pred = results['predictions']
    y_pred_proba = results['probabilities']

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (multi-class)
    try:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0

    comparison_data.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Training Time (s)': results['training_time']
    })

# comparison dataframe
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("Model Performance Comparison:")
display(comparison_df.round(4))

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Accuracy comparison
axes[0, 0].barh(comparison_df['Model'], comparison_df['Accuracy'], color='grey')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].grid(axis='x', alpha=0.5)

# F1-Score comparison
axes[0, 1].barh(comparison_df['Model'], comparison_df['F1-Score'], color='blue')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('Model F1-Score Comparison')
axes[0, 1].grid(axis='x', alpha=0.5)

# ROC-AUC comparison
axes[1, 0].barh(comparison_df['Model'], comparison_df['ROC-AUC'], color='yellow')
axes[1, 0].set_xlabel('ROC-AUC')
axes[1, 0].set_title('Model ROC-AUC Comparison')
axes[1, 0].grid(axis='x', alpha=0.5)

# Training time comparison
axes[1, 1].barh(comparison_df['Model'], comparison_df['Training Time (s)'], color='green')
axes[1, 1].set_xlabel('Training Time (seconds)')
axes[1, 1].set_title('Model Training Time Comparison')
axes[1, 1].grid(axis='x', alpha=0.5)

plt.tight_layout()
plt.show()

# best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = model_results[best_model_name]['model']
best_predictions = model_results[best_model_name]['predictions']

print(f"Best Performing Model: {best_model_name}")
print(f"F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")

# Detailed report for best model
print(f"\nDetailed Report for {best_model_name}:")
print()
print(classification_report(y_test, best_predictions,
                           target_names=target_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

"""# **MULTI-LEVEL EXPLAINABILITY ANALYSIS**

# **`SHAP Analysis (Global Explainability)`**
"""

# Install SHAP
!pip install shap -q

import shap

print("Done installing SHAP!")

# Initialize SHAP explainer based on best model
if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree', 'Gradient Boosting']:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    X_test_shap = X_test_scaled
elif best_model_name == 'SVM':
    # For SVM, use KernelExplainer with a sample
    X_sample = shap.sample(X_train_scaled, 100)
    explainer = shap.KernelExplainer(best_model.predict_proba, X_sample)
    shap_values = explainer.shap_values(X_test_scaled[:100])
    X_test_shap = X_test_scaled[:100]
else:
    # For other models (Neural Network, Logistic Regression)
    X_sample = shap.sample(X_train_scaled, 100)
    explainer = shap.KernelExplainer(best_model.predict_proba, X_sample)
    shap_values = explainer.shap_values(X_test_scaled[:100])
    X_test_shap = X_test_scaled[:100]

print("SHAP values calculated successfully")

# Global Feature Importance
print("\nGlobal Feature Importance (SHAP):")

plt.figure(figsize=(7, 7))
if isinstance(shap_values, list):
    # Multi-class case - use class 0 or average
    shap.summary_plot(shap_values[0], X_test_scaled if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree'] else X_test_shap,
                     plot_type="bar", show=False)
else:
    shap.summary_plot(shap_values, X_test_scaled if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree'] else X_test_shap,
                     plot_type="bar", show=False)
plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Summary Plot
plt.figure(figsize=(12, 20))
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[0], X_test_scaled if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree'] else X_test_shap, show=False)
else:
    shap.summary_plot(shap_values, X_test_scaled if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree'] else X_test_shap, show=False)
plt.title('SHAP Summary Plot', fontsize=9)
plt.tight_layout()
plt.show()

"""# **`LIME Analysis (Local Explainability)`**"""

# Install LIME
!pip install lime -q

import lime
import lime.lime_tabular

# Initialize LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled.values,
    feature_names=X_train_scaled.columns.tolist(),
    class_names=target_encoder.classes_,
    mode='classification',
    random_state=42
)

print("LIME explainer initialized")

# Explain a few instances
num_instances = 3
print(f"\nExplaining {num_instances} individual predictions:")

for i in range(num_instances):
    instance_idx = i * 10  # Select instances at intervals
    if instance_idx >= len(X_test_scaled):
        break

    instance = X_test_scaled.iloc[instance_idx].values
    true_label = target_encoder.classes_[y_test[instance_idx]]
    pred_label = target_encoder.classes_[best_predictions[instance_idx]]

    print(f"\nInstance {i+1}:")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label}")

    # Generate explanation
    explanation = lime_explainer.explain_instance(
        instance,
        best_model.predict_proba,
        num_features=10
    )

    # Show explanation
    fig = explanation.as_pyplot_figure()
    plt.title(f'LIME Explanation - Instance {i+1}\nTrue: {true_label}, Predicted: {pred_label}',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

"""# **`BEHAVIORAL RISK PROFILING`**"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("BEHAVIORAL RISK PROFILING")

# Recreate df_processed by combining train and test
df_processed = pd.concat([
    pd.concat([X_train_raw, y_train_raw], axis=1),
    pd.concat([X_test_raw, y_test_raw], axis=1)
], axis=0).reset_index(drop=True)

print(f"\nDataset size: {len(df_processed)} riders")

# Calculate Behavioral Risk Score
print("Calculating Behavioral Risk Score")

df_processed['Behavioral_Risk_Score'] = (
    df_processed['Talk_While_Riding'].map({'Yes': 1, 'No': 0}).fillna(0) +
    df_processed['Smoke_While_Riding'].map({'Yes': 1, 'No': 0}).fillna(0) +
    df_processed['Wearing_Helmet'].map({'No': 1, 'Yes': 0}).fillna(0) +
    df_processed['Valid_Driving_License'].map({'No': 1, 'Yes': 0}).fillna(0) +
    df_processed['Biker_Alcohol'] * 2
)

print("\nRisk Score Statistics:")
print(df_processed['Behavioral_Risk_Score'].describe())
print("\nRisk Score Distribution:")
print(df_processed['Behavioral_Risk_Score'].value_counts().sort_index().head(10))

# Select Features for Clustering
print("Selecting Features for Clustering")

behavioral_features = ['Behavioral_Risk_Score']

for col in ['Biker_Age', 'Riding_Experience']:
    if col in df_processed.columns:
        behavioral_features.append(col)

print(f"Using {len(behavioral_features)} features:")
for feat in behavioral_features:
    print(f"  - {feat}")

X_behavioral = df_processed[behavioral_features].copy()

# Scale the Data
scaler_cluster = StandardScaler()
X_behavioral_scaled = scaler_cluster.fit_transform(X_behavioral)

# Determine Optimal Clusters
print("Determining Optimal Clusters")

silhouette_scores = []
K_range = range(2, 7)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_behavioral_scaled)
    score = silhouette_score(X_behavioral_scaled, labels)
    silhouette_scores.append(score)
    print(f"  K={k}: Silhouette Score = {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Perform Clustering
optimal_k = 4
print(f"Performing Clustering (K={optimal_k})")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_processed['Rider_Cluster'] = kmeans_final.fit_predict(X_behavioral_scaled)

print("\nSorting clusters by severe accident rate...")
cluster_severity = df_processed.groupby('Rider_Cluster').apply(
    lambda x: (x['Accident_Severity'] == 'Severe Accident').sum() / len(x),
    include_groups=False
).sort_values()

cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_severity.index)}
df_processed['Rider_Profile'] = df_processed['Rider_Cluster'].map(cluster_mapping)

cluster_names = {
    0: 'Safest Riders',
    1: 'Low-Risk Riders',
    2: 'High-Risk Riders',
    3: 'Severe Accident Prone'
}

# Analyzing Clusters
print("Cluster Analysis:")

for cluster_id in range(optimal_k):
    cluster_data = df_processed[df_processed['Rider_Profile'] == cluster_id]
    severe_rate = (cluster_data['Accident_Severity'] == 'Severe Accident').sum() / len(cluster_data) * 100

    print(f"\n{cluster_names[cluster_id]} (Profile {cluster_id}):")
    print(f"  Size: {len(cluster_data):,} riders ({len(cluster_data)/len(df_processed)*100:.1f}%)")
    print(f"  Avg Risk Score: {cluster_data['Behavioral_Risk_Score'].mean():.2f}")
    print(f"  Severe Accident Rate: {severe_rate:.1f}%")
    print(f"  Accident Distribution:")

    severity_dist = cluster_data['Accident_Severity'].value_counts(normalize=True) * 100
    for severity in ['No Accident', 'Moderate Accident', 'Severe Accident']:
        pct = severity_dist.get(severity, 0)
        print(f"    {severity}: {pct:.1f}%")

print("Creating Visualizations\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Cluster Size Distribution
cluster_sizes = [len(df_processed[df_processed['Rider_Profile']==i]) for i in range(optimal_k)]
axes[0, 0].bar([cluster_names[i] for i in range(optimal_k)], cluster_sizes,
               color='steelblue', edgecolor='black')
axes[0, 0].set_title('Rider Profile Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Riders', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=10)
axes[0, 0].grid(axis='y', alpha=0.3)

# Average Risk Score by Profile
risk_by_cluster = [df_processed[df_processed['Rider_Profile']==i]['Behavioral_Risk_Score'].mean()
                   for i in range(optimal_k)]
bars = axes[0, 1].bar([cluster_names[i] for i in range(optimal_k)], risk_by_cluster,
                       color='coral', edgecolor='black')
axes[0, 1].set_title('Average Risk Score by Profile', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Risk Score', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=10)
axes[0, 1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Accident Severity Distribution
severity_data = df_processed.groupby(['Rider_Profile', 'Accident_Severity']).size().unstack(fill_value=0)
severity_data = severity_data[['No Accident', 'Moderate Accident', 'Severe Accident']]
severity_data.plot(kind='bar', stacked=True, ax=axes[1, 0],
                   color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
axes[1, 0].set_title('Accident Severity by Rider Profile', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Rider Profile', fontsize=12)
axes[1, 0].set_ylabel('Number of Riders', fontsize=12)
axes[1, 0].set_xticklabels([cluster_names[i] for i in range(optimal_k)], rotation=45, ha='right')
axes[1, 0].legend(title='Severity', loc='upper left', fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3)

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_behavioral_scaled)

scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1],
                             c=df_processed['Rider_Profile'],
                             cmap='tab10', alpha=0.6, s=40,
                             edgecolors='black', linewidth=0.5)

axes[1, 1].set_title('PCA Visualization of Rider Profiles', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = fig.colorbar(scatter, ax=axes[1, 1], pad=0.02)
cbar.set_label('Profile ID', rotation=270, labelpad=20, fontsize=11)
cbar.set_ticks(range(optimal_k))
cbar.set_ticklabels([f'{i}' for i in range(optimal_k)])

plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_behavioral_scaled)

scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1],
                             c=df_processed['Rider_Profile'],
                             cmap='tab10', alpha=0.6, s=40,
                             edgecolors='black', linewidth=0.5)

axes[1, 1].set_title('PCA Visualization of Rider Profiles', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = fig.colorbar(scatter, ax=axes[1, 1], pad=0.02)
cbar.set_label('Profile ID', rotation=270, labelpad=20, fontsize=11)
cbar.set_ticks(range(optimal_k))
cbar.set_ticklabels([f'{i}' for i in range(optimal_k)])

plt.tight_layout()
plt.show()

print("=== Top Risk Factors (SHAP Analysis) ===")

feature_cols_current = X_test_scaled.columns.tolist()

if isinstance(shap_values, list):
    shap_importance = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
else:
    if shap_values.ndim == 3:
        shap_importance = np.abs(shap_values).mean(axis=(0, 2))
    elif shap_values.ndim == 2:
        shap_importance = np.abs(shap_values).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(0)

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols_current,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

print("\nTop 10 Risk Factors:")
print(feature_importance_df.head(10).to_string(index=False))