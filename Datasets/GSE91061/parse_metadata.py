#!/usr/bin/env python3
"""Parse GSE91061 metadata to understand the dataset."""

import pandas as pd
import numpy as np
import re

# Read TPM data  
tpm = pd.read_csv('GSE91061_norm_counts_TPM_GRCh38.p13_NCBI.tsv', sep='\t', index_col=0)
print('='*60)
print('TPM EXPRESSION MATRIX')
print('='*60)
print(f'Shape: {tpm.shape[0]} genes x {tpm.shape[1]} samples')
print(f'Gene ID format: NCBI Entrez IDs (e.g., {tpm.index[:3].tolist()})')
print(f'Sample IDs: GSM format (e.g., {tpm.columns[:3].tolist()})')
print()

# Parse metadata from series matrix  
with open('GSE91061_series_matrix.txt', 'r') as f:
    content = f.read()

# Get sample IDs
sample_ids = tpm.columns.tolist()
print(f'Total samples: {len(sample_ids)}')
print()

# Parse sample characteristics  
lines = content.split('\n')
metadata = {}
for line in lines:
    if line.startswith('!Sample_title'):
        titles = re.findall(r'"([^"]+)"', line)
        metadata['title'] = titles
    elif line.startswith('!Sample_characteristics_ch1') and 'visit' in line:
        visits = re.findall(r'visit \(pre or on treatment\): (\w+)', line)
        metadata['visit'] = visits
    elif line.startswith('!Sample_characteristics_ch1') and 'response:' in line:
        responses = re.findall(r'response: (\w+)', line)
        metadata['response'] = responses

# Create metadata DataFrame
meta_df = pd.DataFrame({
    'sample_id': sample_ids,
    'title': metadata.get('title', [])[:len(sample_ids)],
    'visit': metadata.get('visit', [])[:len(sample_ids)],
    'response': metadata.get('response', [])[:len(sample_ids)]
})

# Parse patient ID from title
meta_df['patient'] = meta_df['title'].str.extract(r'(Pt\d+)')

print('='*60)
print('METADATA SUMMARY')
print('='*60)
print()
print('Visit (Pre vs On treatment):')
print(meta_df['visit'].value_counts())
print()
print('Response categories:')
print(meta_df['response'].value_counts())
print()
print('Number of unique patients:', meta_df['patient'].nunique())
print()

# Binary response label analysis
print('='*60)
print('USABLE LABELS FOR ML')
print('='*60)
print()
print('Response can be converted to BINARY:')
print('  - Responders (R): PRCR (Partial Response / Complete Response)')
print('  - Non-responders (NR): PD (Progressive Disease), SD (Stable Disease)')
print('  - Exclude: UNK (Unknown)')
print()
responders = (meta_df['response'] == 'PRCR').sum()
non_responders = meta_df['response'].isin(['PD', 'SD']).sum()
unknown = (meta_df['response'] == 'UNK').sum()
print(f'Responders (PRCR): {responders} samples')
print(f'Non-responders (PD+SD): {non_responders} samples')
print(f'Unknown (exclude): {unknown} samples')
print()

# Pre-treatment only (for prediction)
pre_df = meta_df[meta_df['visit'] == 'Pre']
print('='*60)
print('PRE-TREATMENT SAMPLES ONLY (Best for ICB prediction)')
print('='*60)
print(f'Total pre-treatment samples: {len(pre_df)}')
print('Response distribution (pre-treatment):')
print(pre_df['response'].value_counts())
print()
pre_responders = (pre_df['response'] == 'PRCR').sum()
pre_non_resp = pre_df['response'].isin(['PD', 'SD']).sum()
pre_unknown = (pre_df['response'] == 'UNK').sum()
print(f'Pre-treatment Responders: {pre_responders}')
print(f'Pre-treatment Non-responders: {pre_non_resp}')
print(f'Pre-treatment Unknown (exclude): {pre_unknown}')

# Save processed metadata
meta_df.to_csv('GSE91061_metadata.csv', index=False)
print()
print('Saved: GSE91061_metadata.csv')
