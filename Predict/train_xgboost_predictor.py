#!/usr/bin/env python3
"""
Train XGBoost classifier to predict immunotherapy response using extracted features.
- Loads features and binary labels
- Splits into train/test sets
- Trains XGBoost model
- Reports accuracy, ROC AUC, and feature importances
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load features and labels
features = pd.read_csv('Datasets/GSE91061/GSE91061_features.csv', index_col=0)
labels = pd.read_csv('Datasets/GSE91061/GSE91061_labels_binary.csv', index_col=0)

# Align samples
X = features.loc[labels.index]
y = labels['label']

# Class imbalance handling
num_neg = (y == 0).sum()
num_pos = (y == 1).sum()
scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
print(f"Class 0 (non-responder): {num_neg}, Class 1 (responder): {num_pos}, scale_pos_weight: {scale_pos_weight:.2f}")

# Stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []
all_reports = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Apply SMOTE to training data only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Fold {fold}: After SMOTE - Class 0: {sum(y_train_res==0)}, Class 1: {sum(y_train_res==1)}")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=1 # No need for scale_pos_weight after SMOTE
    )
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    accs.append(acc)
    aucs.append(auc)
    report = classification_report(y_test, y_pred, output_dict=True)
    all_reports.append(report)
    print(f"Fold {fold}: Accuracy={acc:.3f}, ROC AUC={auc:.3f}")

# Average metrics
print("\n==== Cross-Validation Results ====")
print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Mean ROC AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# Hyperparameter tuning with RandomizedSearchCV and SMOTE
param_dist = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [2, 3, 4, 5],
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__gamma': [0, 0.5, 1],
}
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])
rs = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
rs.fit(X, y)
print(f"\nBest ROC AUC: {rs.best_score_:.3f}")
print(f"Best Params: {rs.best_params_}")

# Predict with best model using cross_val_predict for more robust estimate
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(rs.best_estimator_, X, y, cv=5, method='predict')
y_proba = cross_val_predict(rs.best_estimator_, X, y, cv=5, method='predict_proba')[:,1]
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba)
print(f"Overall Accuracy (CV): {acc:.3f}")
print(f"Overall ROC AUC (CV): {auc:.3f}")
print(classification_report(y, y_pred))

# Feature importance plot (fit on all data)
rs.best_estimator_.fit(X, y)
importances = rs.best_estimator_.named_steps['xgb'].feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Top 20 Feature Importances (XGBoost, tuned)")
plt.bar(range(20), importances[indices[:20]], align="center")
plt.xticks(range(20), X.columns[indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig('Datasets/GSE91061/xgb_feature_importance_tuned.png')
plt.show()
# Train final model on all data for feature importance
clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
clf.fit(X, y)

# Feature importance plot
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Top 20 Feature Importances (XGBoost)")
plt.bar(range(20), importances[indices[:20]], align="center")
plt.xticks(range(20), X.columns[indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig('Datasets/GSE91061/xgb_feature_importance.png')
plt.show()
