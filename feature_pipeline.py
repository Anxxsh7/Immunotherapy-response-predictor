#!/usr/bin/env python3
"""
Feature Extraction Pipeline - Unified Interface

This script provides a unified interface to run all feature extraction modules
on a single expression matrix and merge the outputs into a combined feature vector.

INPUT FORMAT REQUIREMENTS:
--------------------------
The input expression matrix should be:
  - Format: Tab-separated file (.tsv or .txt)
  - Orientation: Genes (rows) x Samples (columns)
  - Row index: Gene symbols (e.g., CD8A, PDCD1, etc.)
  - Values: TPM (Transcripts Per Million) - NOT log-transformed
  
The pipeline will automatically convert to the format required by each module:
  - IMMUCELL (ImmuCellAI): TPM, genes x samples ✓ (native)
  - IMMUNESUBTYPES: log2(TPM+1), genes x samples (auto-converted)
  - IMPRES: Any expression, samples x genes (auto-transposed)
  - TIDE: Normalized log2, genes x samples (auto-converted + centered)

OUTPUT:
-------
A merged feature matrix with samples as rows and all features as columns.

Usage:
    python feature_pipeline.py --input expression_tpm.txt --output features.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import warnings

# ============================================================================
# Input Format Converters
# ============================================================================

def load_expression_matrix(filepath: str) -> pd.DataFrame:
    """
    Load expression matrix from file.
    
    Expected format:
        - Tab-separated
        - Genes as rows (index)
        - Samples as columns
        - Values: TPM (not log-transformed)
    
    Returns:
        DataFrame with genes as rows, samples as columns
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, index_col=0)
    else:
        df = pd.read_csv(filepath, sep='\t', index_col=0)
    
    print(f"Loaded expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def convert_to_log2(tpm_df: pd.DataFrame) -> pd.DataFrame:
    """Convert TPM to log2(TPM + 1)."""
    return np.log2(tpm_df + 1)


def convert_to_sample_centered(log2_df: pd.DataFrame) -> pd.DataFrame:
    """Center each gene by subtracting the mean across samples."""
    return log2_df.sub(log2_df.mean(axis=1), axis=0)


def transpose_for_impres(df: pd.DataFrame, sample_col: str = "sample_id") -> pd.DataFrame:
    """
    Transpose expression matrix for IMPRES (samples as rows, genes as columns).
    Adds a sample_id column.
    """
    transposed = df.T.copy()
    transposed.insert(0, sample_col, transposed.index)
    transposed = transposed.reset_index(drop=True)
    return transposed


# ============================================================================
# Module Runners
# ============================================================================

def run_immucell(expression_tpm: pd.DataFrame, sample_type: str = "tumor") -> pd.DataFrame:
    """
    Run ImmuCellAI for immune cell deconvolution.
    
    Input: TPM expression matrix (genes x samples)
    Output: Immune cell abundances (samples x cell types)
    """
    try:
        import immucellai2
    except ImportError:
        print("WARNING: immucellai2 not installed. Skipping IMMUCELL.")
        return None
    
    print("\n" + "="*60)
    print("Running IMMUCELL (ImmuCellAI 2.0)")
    print("="*60)
    
    # Save to temp file (ImmuCellAI expects a file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        expression_tpm.to_csv(f, sep='\t')
        temp_input = f.name
    
    try:
        # Load reference
        if sample_type == "tumor":
            reference = immucellai2.load_tumor_reference_data()
        else:
            reference = immucellai2.load_normal_reference_data()
        
        # Run deconvolution
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_output = f.name
        
        result_obj = immucellai2.run_ImmuCellAI2(
            reference_file=reference,
            sample_file=temp_input,
            output_file=temp_output,
            thread_num=4
        )
        
        # Get results
        cell_fractions = result_obj.get_result(ResultIndex=0)
        
        # Add prefix to columns
        cell_fractions.columns = ["IMMUCELL_" + col for col in cell_fractions.columns]
        
        print(f"IMMUCELL output: {cell_fractions.shape[1]} features")
        return cell_fractions
        
    finally:
        # Cleanup
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)


def run_tide(expression_tpm: pd.DataFrame, cancer_type: str = "Other") -> pd.DataFrame:
    """
    Run TIDE analysis.
    
    Input: TPM expression matrix (genes x samples) - will be log2 + centered
    Output: TIDE scores (samples x features)
    """
    try:
        from tidepy.pred import TIDE
    except ImportError:
        print("WARNING: tidepy not installed. Skipping TIDE.")
        return None
    
    print("\n" + "="*60)
    print("Running TIDE")
    print("="*60)
    
    # TIDE expects normalized data - convert TPM to log2 and center
    log2_expr = convert_to_log2(expression_tpm)
    centered_expr = convert_to_sample_centered(log2_expr)
    
    # Run TIDE
    result = TIDE(centered_expr, cancer=cancer_type, pretreat=False)
    
    # Select numeric columns only and add prefix
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    tide_features = result[numeric_cols].copy()
    tide_features.columns = ["TIDE_" + col for col in tide_features.columns]
    
    print(f"TIDE output: {tide_features.shape[1]} features")
    return tide_features


def run_immunesubtypes_r(expression_tpm: pd.DataFrame) -> pd.DataFrame:
    """
    Run ImmuneSubtypeClassifier via R.
    
    Input: TPM expression matrix (genes x samples) - will be log2 transformed
    Output: Subtype probabilities (samples x 6 subtypes)
    """
    print("\n" + "="*60)
    print("Running IMMUNESUBTYPES (R)")
    print("="*60)
    
    # Convert to log2
    log2_expr = convert_to_log2(expression_tpm)
    
    # Save to temp file for R
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        log2_expr.to_csv(f, sep='\t')
        temp_input = f.name
    
    temp_output = tempfile.mktemp(suffix='.csv')
    
    # Create R script
    r_script = f'''
    library(ImmuneSubtypeClassifier)
    
    expr <- read.table("{temp_input}", header=TRUE, row.names=1, sep="\\t")
    results <- callEnsemble(X = as.matrix(expr), geneids = "symbol")
    write.csv(results, "{temp_output}", row.names=FALSE)
    '''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        r_script_path = f.name
    
    try:
        import subprocess
        result = subprocess.run(['Rscript', r_script_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"WARNING: R script failed: {result.stderr}")
            return None
        
        # Load results
        subtype_df = pd.read_csv(temp_output)
        subtype_df.index = subtype_df['SampleIDs']
        
        # Keep only probability columns and rename
        prob_cols = [col for col in subtype_df.columns if col not in ['SampleIDs', 'BestCall']]
        subtype_features = subtype_df[prob_cols].copy()
        
        # Rename columns
        subtype_names = {
            '1': 'C1_Wound_Healing',
            '2': 'C2_IFN_gamma_Dominant',
            '3': 'C3_Inflammatory',
            '4': 'C4_Lymphocyte_Depleted',
            '5': 'C5_Immunologically_Quiet',
            '6': 'C6_TGF_beta_Dominant'
        }
        subtype_features.columns = ["IMMUNESUBTYPE_" + subtype_names.get(col, col) 
                                    for col in subtype_features.columns]
        
        print(f"IMMUNESUBTYPES output: {subtype_features.shape[1]} features")
        return subtype_features
        
    except FileNotFoundError:
        print("WARNING: Rscript not found. Skipping IMMUNESUBTYPES.")
        return None
    finally:
        for f in [temp_input, temp_output, r_script_path]:
            if os.path.exists(f):
                os.remove(f)


def run_impres_r(expression_tpm: pd.DataFrame) -> pd.DataFrame:
    """
    Run IMPRES score calculation via R.
    
    Input: TPM expression matrix (genes x samples)
    Output: IMPRES scores and gene pair differences (samples x features)
    """
    print("\n" + "="*60)
    print("Running IMPRES (R)")
    print("="*60)
    
    # Transpose for IMPRES (samples x genes) and add sample_id column
    impres_input = transpose_for_impres(expression_tpm)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        impres_input.to_csv(f, sep='\t', index=False)
        temp_input = f.name
    
    temp_output = tempfile.mktemp(suffix='.csv')
    
    # Get the IMPRES.R path
    script_dir = Path(__file__).parent
    impres_r_path = script_dir / "IMPRES.R"
    
    # Create R script
    r_script = f'''
    library(data.table)
    source("{impres_r_path}")
    
    ge_df <- fread("{temp_input}")
    results <- calc_impres(ge_df, sample_key = "sample_id", include_continuous = TRUE)
    write.csv(results, "{temp_output}", row.names=FALSE)
    '''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_script)
        r_script_path = f.name
    
    try:
        import subprocess
        result = subprocess.run(['Rscript', r_script_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"WARNING: R script failed: {result.stderr}")
            return None
        
        # Load results
        impres_df = pd.read_csv(temp_output)
        impres_df.index = impres_df['sample_id']
        
        # Remove sample_id column and add prefix
        impres_features = impres_df.drop(columns=['sample_id'])
        impres_features.columns = ["IMPRES_" + col if not col.startswith("IMPRES") else col 
                                   for col in impres_features.columns]
        
        print(f"IMPRES output: {impres_features.shape[1]} features")
        return impres_features
        
    except FileNotFoundError:
        print("WARNING: Rscript not found. Skipping IMPRES.")
        return None
    finally:
        for f in [temp_input, temp_output, r_script_path]:
            if os.path.exists(f):
                os.remove(f)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_feature_pipeline(
    expression_file: str,
    output_file: str,
    sample_type: str = "tumor",
    cancer_type: str = "Other",
    modules: list = None
) -> pd.DataFrame:
    """
    Run the complete feature extraction pipeline.
    
    Args:
        expression_file: Path to TPM expression matrix (genes x samples)
        output_file: Path to save merged features
        sample_type: "tumor" or "normal" for ImmuCellAI
        cancer_type: Cancer type for TIDE ("Melanoma", "NSCLC", "Other")
        modules: List of modules to run. Default: all available
    
    Returns:
        Merged feature DataFrame (samples x features)
    """
    if modules is None:
        modules = ['immucell', 'tide', 'immunesubtypes', 'impres']
    
    print("="*60)
    print("Feature Extraction Pipeline")
    print("="*60)
    print(f"Input: {expression_file}")
    print(f"Modules: {', '.join(modules)}")
    print("="*60)
    
    # Load expression data
    expression_tpm = load_expression_matrix(expression_file)
    samples = expression_tpm.columns.tolist()
    
    # Run each module and collect results
    all_features = []
    
    if 'immucell' in modules:
        result = run_immucell(expression_tpm, sample_type)
        if result is not None:
            all_features.append(result)
    
    if 'tide' in modules:
        result = run_tide(expression_tpm, cancer_type)
        if result is not None:
            all_features.append(result)
    
    if 'immunesubtypes' in modules:
        result = run_immunesubtypes_r(expression_tpm)
        if result is not None:
            all_features.append(result)
    
    if 'impres' in modules:
        result = run_impres_r(expression_tpm)
        if result is not None:
            all_features.append(result)
    
    # Merge all features
    if not all_features:
        print("\nERROR: No features were extracted!")
        return None
    
    print("\n" + "="*60)
    print("Merging Features")
    print("="*60)
    
    # Concatenate all feature DataFrames
    merged = pd.concat(all_features, axis=1)
    
    # Ensure all samples are present
    merged = merged.reindex(samples)
    
    # Save results
    merged.to_csv(output_file)
    
    print(f"\nMerged feature matrix: {merged.shape[0]} samples x {merged.shape[1]} features")
    print(f"Saved to: {output_file}")
    
    # Print feature summary
    print("\nFeature Summary:")
    print("-"*40)
    for prefix in ['IMMUCELL', 'TIDE', 'IMMUNESUBTYPE', 'IMPRES']:
        cols = [c for c in merged.columns if c.startswith(prefix)]
        if cols:
            print(f"  {prefix}: {len(cols)} features")
    
    return merged


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified feature extraction pipeline for ICB response prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Format:
  The input should be a TPM expression matrix:
    - Tab-separated file (.txt or .tsv)
    - Genes as rows (gene symbols as index)
    - Samples as columns
    - Values: TPM (NOT log-transformed)

Example:
  python feature_pipeline.py -i expression_tpm.txt -o features.csv
  python feature_pipeline.py -i expression_tpm.txt -o features.csv --modules immucell tide
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input TPM expression matrix (genes x samples, tab-separated)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path for merged features"
    )
    parser.add_argument(
        "--sample-type",
        choices=["tumor", "normal"],
        default="tumor",
        help="Sample type for ImmuCellAI (default: tumor)"
    )
    parser.add_argument(
        "--cancer-type",
        choices=["Melanoma", "NSCLC", "Other"],
        default="Other",
        help="Cancer type for TIDE (default: Other)"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["immucell", "tide", "immunesubtypes", "impres"],
        default=None,
        help="Modules to run (default: all)"
    )
    
    args = parser.parse_args()
    
    run_feature_pipeline(
        expression_file=args.input,
        output_file=args.output,
        sample_type=args.sample_type,
        cancer_type=args.cancer_type,
        modules=args.modules
    )
