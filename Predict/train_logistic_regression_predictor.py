import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE

# Load features and labels
df = pd.read_csv('Datasets/GSE91061/GSE91061_features.csv', index_col=0)
labels = pd.read_csv('Datasets/GSE91061/GSE91061_labels_binary.csv', index_col=0)
X = df.values
y = labels.values.ravel()

# Stratified 5-fold CV with SMOTE
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []
all_preds, all_trues = [], []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    accs.append(accuracy_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_proba))
    all_preds.extend(y_pred)
    all_trues.extend(y_test)

print("==== Logistic Regression CV Results ====")
print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Mean ROC AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(classification_report(all_trues, all_preds))
