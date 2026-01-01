#!/usr/bin/env python3
"""
TIDE Analysis using Official tidepy Package

This script provides a complete interface to the official TIDE (Tumor Immune 
Dysfunction and Exclusion) package for predicting immunotherapy response.

Installation:
    pip install tidepy

Usage:
    python tidepy_analysis.py --input expression_data.csv --output results/
    
Reference:
    Jiang et al., "Signatures of T cell dysfunction and exclusion predict 
    cancer immunotherapy response", Nature Medicine, 2018
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import tidepy
    from tidepy import TIDE
    TIDEPY_AVAILABLE = True
except ImportError:
    TIDEPY_AVAILABLE = False
    print("Warning: tidepy not installed. Install with: pip install tidepy")


def check_tidepy_installation():
    """Check if tidepy is properly installed."""
    if not TIDEPY_AVAILABLE:
        raise ImportError(
            "tidepy is not installed. Please install it with:\n"
            "    pip install tidepy\n"
            "Or visit: https://github.com/jingxinfu/TIDEpy"
        )
    return True


def load_expression_data(filepath: str) -> pd.DataFrame:
    """
    Load gene expression data from file.
    
    Args:
        filepath: Path to expression data (CSV or TSV)
        
    Returns:
        DataFrame with genes as rows and samples as columns
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, index_col=0)
    elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
        df = pd.read_csv(filepath, sep='\t', index_col=0)
    else:
        # Try to infer delimiter
        df = pd.read_csv(filepath, index_col=0, sep=None, engine='python')
    
    print(f"Loaded expression data: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def run_tide_analysis(
    expression_data: pd.DataFrame,
    cancer_type: str = None,
    pretreatment: bool = True,
    output_dir: str = None
) -> dict:
    """
    Run complete TIDE analysis using the official tidepy package.
    
    Args:
        expression_data: Gene expression DataFrame (genes x samples)
        cancer_type: Cancer type for normalization (e.g., 'Melanoma', 'NSCLC')
                    If None, uses pan-cancer model
        pretreatment: Whether samples are pre-treatment (True) or on-treatment (False)
        output_dir: Directory to save output files
        
    Returns:
        Dictionary containing all TIDE results and signatures
    """
    check_tidepy_installation()
    
    results = {}
    
    # Initialize TIDE
    print("\n" + "="*60)
    print("Running TIDE Analysis")
    print("="*60)
    
    # Run TIDE prediction
    print("\nCalculating TIDE scores...")
    tide_result = TIDE(
        expression_data,
        cancer=cancer_type,
        pretreat=pretreatment
    )
    
    results['tide_scores'] = tide_result
    
    # Extract individual components
    print("\nExtracting component scores...")
    
    # The TIDE result contains multiple columns:
    # - Dysfunction: T cell dysfunction score
    # - Exclusion: T cell exclusion score  
    # - MDSC: MDSC exclusion score
    # - CAF: CAF exclusion score
    # - TAM M2: TAM M2 exclusion score
    # - TIDE: Final TIDE score
    # - Responder: Predicted response (True/False)
    
    print(f"\nTIDE Results Summary:")
    print(f"  - Number of samples: {len(tide_result)}")
    print(f"  - Columns: {list(tide_result.columns)}")
    
    if 'Responder' in tide_result.columns:
        responders = tide_result['Responder'].sum()
        non_responders = len(tide_result) - responders
        print(f"  - Predicted Responders: {responders}")
        print(f"  - Predicted Non-responders: {non_responders}")
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main TIDE results
        tide_output = output_path / "tide_results.csv"
        tide_result.to_csv(tide_output)
        print(f"\nSaved TIDE results to: {tide_output}")
        
        # Save summary statistics
        summary = tide_result.describe()
        summary_output = output_path / "tide_summary_stats.csv"
        summary.to_csv(summary_output)
        print(f"Saved summary statistics to: {summary_output}")
        
        results['output_files'] = {
            'tide_results': str(tide_output),
            'summary_stats': str(summary_output)
        }
    
    return results


def get_tide_signatures() -> dict:
    """
    Get the gene signatures used by TIDE.
    
    Returns:
        Dictionary containing dysfunction and exclusion signatures
    """
    check_tidepy_installation()
    
    signatures = {}
    
    # TIDE uses these key signatures:
    # 1. Dysfunction signature - genes associated with T cell dysfunction
    # 2. Exclusion signatures - genes associated with immune exclusion
    #    - MDSC (Myeloid-derived suppressor cells)
    #    - CAF (Cancer-associated fibroblasts)  
    #    - TAM M2 (Tumor-associated macrophages M2)
    
    # CTL (Cytotoxic T Lymphocyte) markers used for stratification
    ctl_markers = ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1']
    signatures['CTL_markers'] = ctl_markers
    
    print("\nTIDE Gene Signatures:")
    print("-" * 40)
    print(f"CTL Markers: {', '.join(ctl_markers)}")
    print("\nNote: Full signatures are embedded in the tidepy package.")
    print("The package uses pre-computed weights from the original TIDE paper.")
    
    return signatures


def batch_tide_analysis(
    expression_files: list,
    cancer_types: list = None,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Run TIDE analysis on multiple expression files.
    
    Args:
        expression_files: List of paths to expression data files
        cancer_types: List of cancer types (same length as expression_files)
                     If None, uses pan-cancer for all
        output_dir: Directory to save combined results
        
    Returns:
        Combined DataFrame with all TIDE results
    """
    check_tidepy_installation()
    
    all_results = []
    
    if cancer_types is None:
        cancer_types = [None] * len(expression_files)
    
    for i, (expr_file, cancer) in enumerate(zip(expression_files, cancer_types)):
        print(f"\nProcessing file {i+1}/{len(expression_files)}: {expr_file}")
        
        expr_data = load_expression_data(expr_file)
        result = run_tide_analysis(expr_data, cancer_type=cancer)
        
        # Add source file info
        result['tide_scores']['source_file'] = os.path.basename(expr_file)
        all_results.append(result['tide_scores'])
    
    # Combine all results
    combined = pd.concat(all_results, axis=0)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        combined_output = output_path / "combined_tide_results.csv"
        combined.to_csv(combined_output)
        print(f"\nSaved combined results to: {combined_output}")
    
    return combined


def extract_features_for_model(tide_results: pd.DataFrame) -> pd.DataFrame:
    """
    Extract TIDE-derived features for use in downstream ML models.
    
    Args:
        tide_results: DataFrame from run_tide_analysis
        
    Returns:
        DataFrame with processed features ready for ML
    """
    features = pd.DataFrame(index=tide_results.index)
    
    # Core TIDE features
    numeric_cols = ['Dysfunction', 'Exclusion', 'MDSC', 'CAF', 'TAM M2', 'TIDE']
    
    for col in numeric_cols:
        if col in tide_results.columns:
            features[f'TIDE_{col}'] = tide_results[col]
    
    # Binary response prediction
    if 'Responder' in tide_results.columns:
        features['TIDE_Responder'] = tide_results['Responder'].astype(int)
    
    # Derived features
    if 'Dysfunction' in tide_results.columns and 'Exclusion' in tide_results.columns:
        # Ratio features
        features['TIDE_Dys_Exc_ratio'] = (
            tide_results['Dysfunction'] / (tide_results['Exclusion'] + 1e-6)
        )
        # Dominant mechanism
        features['TIDE_dominant_mechanism'] = np.where(
            tide_results['Dysfunction'] > tide_results['Exclusion'],
            'Dysfunction', 'Exclusion'
        )
    
    print(f"\nExtracted {len(features.columns)} TIDE-derived features")
    return features


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="TIDE Analysis using official tidepy package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python tidepy_analysis.py --input expression.csv --output results/
    
    # Specify cancer type
    python tidepy_analysis.py --input expression.csv --cancer Melanoma --output results/
    
    # Batch analysis
    python tidepy_analysis.py --batch file1.csv file2.csv --output results/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to gene expression data (genes x samples)'
    )
    parser.add_argument(
        '--batch', '-b',
        nargs='+',
        help='Multiple expression files for batch processing'
    )
    parser.add_argument(
        '--cancer', '-c',
        type=str,
        default=None,
        help='Cancer type (e.g., Melanoma, NSCLC, BLCA). Default: pan-cancer'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='tide_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--pretreatment',
        action='store_true',
        default=True,
        help='Samples are pre-treatment (default: True)'
    )
    parser.add_argument(
        '--on-treatment',
        action='store_true',
        help='Samples are on-treatment'
    )
    parser.add_argument(
        '--signatures',
        action='store_true',
        help='Print TIDE signature information'
    )
    parser.add_argument(
        '--extract-features',
        action='store_true',
        help='Extract ML-ready features from TIDE results'
    )
    
    args = parser.parse_args()
    
    # Check tidepy installation
    if not TIDEPY_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: tidepy is not installed")
        print("="*60)
        print("\nInstall tidepy with:")
        print("    pip install tidepy")
        print("\nOr visit: https://github.com/jingxinfu/TIDEpy")
        return 1
    
    # Print signatures if requested
    if args.signatures:
        get_tide_signatures()
        return 0
    
    # Determine pretreatment status
    pretreatment = not args.on_treatment
    
    # Run analysis
    if args.batch:
        # Batch processing
        results = batch_tide_analysis(
            args.batch,
            output_dir=args.output
        )
    elif args.input:
        # Single file processing
        expr_data = load_expression_data(args.input)
        result = run_tide_analysis(
            expr_data,
            cancer_type=args.cancer,
            pretreatment=pretreatment,
            output_dir=args.output
        )
        
        # Extract features if requested
        if args.extract_features:
            features = extract_features_for_model(result['tide_scores'])
            features_output = Path(args.output) / "tide_features.csv"
            features.to_csv(features_output)
            print(f"Saved ML features to: {features_output}")
    else:
        parser.print_help()
        return 1
    
    print("\n" + "="*60)
    print("TIDE Analysis Complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    exit(main())
