#!/usr/bin/env python3
"""
Preprocess GSE91061 for feature extraction pipeline.
- Filters to pre-treatment samples only
- Keeps only samples with clear response (PD or PRCR)
- Converts Entrez IDs to gene symbols
- Outputs processed expression matrix and label file
"""
import pandas as pd
import mygene

# Load TPM matrix (Entrez IDs x samples)
tpm = pd.read_csv('GSE91061_norm_counts_TPM_GRCh38.p13_NCBI.tsv', sep='\t', index_col=0)

# Load metadata
def parse_metadata():
    import re
    with open('GSE91061_series_matrix.txt', 'r') as f:
        lines = f.readlines()
    samples = None
    titles = None
    visits = None
    responses = None
    for line in lines:
        if line.startswith('!Sample_geo_accession'):
            samples = re.findall(r'"([^"]+)"', line)
        elif line.startswith('!Sample_title'):
            titles = re.findall(r'"([^"]+)"', line)
        elif 'visit (pre or on treatment)' in line:
            visits = re.findall(r'visit \(pre or on treatment\): (\w+)', line)
        elif 'response:' in line and line.startswith('!Sample_characteristics'):
            responses = re.findall(r'response: (\w+)', line)
    meta_df = pd.DataFrame({
        'sample_id': samples,
        'title': titles,
        'visit': visits,
        'response': responses
    })
    meta_df['patient'] = meta_df['title'].str.extract(r'(Pt\d+)')
    return meta_df

meta = parse_metadata()

# Filter: pre-treatment only, response PD or PRCR
meta = meta[(meta['visit'] == 'Pre') & (meta['response'].isin(['PD', 'PRCR']))]

# Subset TPM matrix to relevant samples
keep_samples = meta['sample_id'].tolist()
expr = tpm[keep_samples]

# Convert Entrez IDs to gene symbols
mg = mygene.MyGeneInfo()
query = mg.querymany(expr.index.tolist(), scopes='entrezgene', fields='symbol', species='human')
# Build mapping
entrez2symbol = {str(q['query']): q.get('symbol','') for q in query if 'symbol' in q}
expr.index = [entrez2symbol.get(str(eid), '') for eid in expr.index]
expr = expr[expr.index != '']  # Remove genes with no symbol
expr = expr[~expr.index.duplicated(keep='first')]

# Save processed expression matrix
target_expr = 'GSE91061_preprocessed_for_pipeline.tsv'
expr.to_csv(target_expr, sep='\t')
print(f'Processed expression matrix saved: {target_expr} ({expr.shape[0]} genes x {expr.shape[1]} samples)')

# Save label file
labels = meta[['sample_id', 'response']].copy()
labels['label'] = labels['response'].map({'PRCR': 1, 'PD': 0})
labels[['sample_id', 'label']].to_csv('GSE91061_labels_binary.csv', index=False)
print('Label file saved: GSE91061_labels_binary.csv')
