import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- PARAMETERS ---
TPM_FILE = 'Datasets/GSE91061/GSE91061_norm_counts_TPM_GRCh38.p13_NCBI.tsv'
IMMUNE_FILE = 'Datasets/GSE91061/GSE91061_features.csv'
LABEL_FILE = 'Datasets/GSE91061/GSE91061_labels_binary.csv'
N_TOP_GENES = 2000
N_PCS = 30

# --- LOAD DATA ---
tpm = pd.read_csv(TPM_FILE, sep='\t', index_col=0)
immune = pd.read_csv(IMMUNE_FILE, index_col=0)
labels = pd.read_csv(LABEL_FILE, index_col=0)

# --- FILTER SAMPLES TO MATCH LABELS ---
common_samples = [s for s in labels.index if s in tpm.columns and s in immune.index]
tpm = tpm[common_samples]
immune = immune.loc[common_samples]
labels = labels.loc[common_samples]

# --- FILTER GENES: REMOVE LOW VARIANCE ---
gene_vars = tpm.var(axis=1)
top_genes = gene_vars.sort_values(ascending=False).head(N_TOP_GENES).index
tpm = tpm.loc[top_genes]

# --- LOG TRANSFORM ---
tpm_log = np.log2(tpm + 1)

# --- PCA ---
pca = PCA(n_components=N_PCS, random_state=42)
X_pca = pca.fit_transform(tpm_log.T)

# --- MERGE PCA + IMMUNE FEATURES ---
X_immune = immune.values
X = np.concatenate([X_pca, X_immune], axis=1)
y = labels.values.ravel()

# --- CV + SMOTE + XGBOOST ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []
all_preds, all_trues = [], []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    accs.append(accuracy_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_proba))
    all_preds.extend(y_pred)
    all_trues.extend(y_test)

print("==== XGBoost (PCA+Immune) CV Results ====")
print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Mean ROC AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(classification_report(all_trues, all_preds))
