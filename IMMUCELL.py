"""
IMMUCELL (Immune Cell Deconvolution) Feature Extraction using ImmuCellAI 2.0

This script extracts immune cell composition features using ImmuCellAI 2.0 deconvolution.
ImmuCellAI 2.0 estimates proportions of 9 major immune cell types and 53 minor subtypes
from bulk RNA-seq data (TPM format).

Reference: https://github.com/GuoBioinfoLab/ImmuCellAI-2.0

Installation:
    pip install immucellai2
"""

import pandas as pd
import numpy as np
import os
import tempfile
from typing import Optional, Union, Tuple

# Try to import immucellai2
try:
    import immucellai2
    IMMUCELLAI_AVAILABLE = True
except ImportError:
    IMMUCELLAI_AVAILABLE = False
    print("Warning: immucellai2 not installed. Install with: pip install immucellai2")


# Major immune cell types from ImmuCellAI 2.0
MAJOR_IMMUNE_CELL_TYPES = [
    "T_cell",
    "B_cell", 
    "NK",
    "Monocyte",
    "Macrophage",
    "DC",
    "Neutrophil",
    "Mast",
    "Plasma"
]

# Selected minor subtypes commonly used for ICB response prediction
ICB_RELEVANT_SUBTYPES = [
    "CD8_T",
    "CD4_T",
    "Treg",
    "Th1",
    "Th2",
    "Th17",
    "Tfh",
    "Exhausted_CD8_T",
    "NK",
    "M1_Macrophage",
    "M2_Macrophage",
    "DC",
    "MDSC",
    "B_cell"
]


def check_immucellai_available() -> bool:
    """Check if ImmuCellAI 2.0 is available."""
    if not IMMUCELLAI_AVAILABLE:
        raise ImportError(
            "immucellai2 is not installed. Please install with: pip install immucellai2"
        )
    return True


def run_immucellai(
    expression_file: str,
    output_file: Optional[str] = None,
    sample_type: str = "tumor",
    thread_num: int = 8,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run ImmuCellAI 2.0 deconvolution on RNA-seq TPM data.
    
    Args:
        expression_file: Path to TPM expression matrix file (tab-separated).
                        Format: genes (rows) x samples (columns), with gene names as index.
        output_file: Optional path to save results. If None, uses temp file.
        sample_type: Either "tumor" or "normal" to use appropriate reference.
        thread_num: Number of threads for parallel processing.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with immune cell type abundances (samples x cell types).
    """
    check_immucellai_available()
    
    # Load appropriate reference data
    if sample_type == "tumor":
        reference_data = immucellai2.load_tumor_reference_data()
    else:
        reference_data = immucellai2.load_normal_reference_data()
    
    # Set up output file
    if output_file is None:
        output_file = tempfile.mktemp(suffix=".xlsx")
        cleanup_output = True
    else:
        cleanup_output = False
    
    # Run ImmuCellAI 2.0
    result_obj = immucellai2.run_ImmuCellAI2(
        reference_file=reference_data,
        sample_file=expression_file,
        output_file=output_file,
        thread_num=thread_num,
        seed=seed
    )
    
    # Get results - CellTypeRatioResult contains the deconvolution results
    cell_fractions = result_obj.get_result(ResultIndex=0)
    
    # Clean up temp file if needed
    if cleanup_output and os.path.exists(output_file):
        os.remove(output_file)
    
    return cell_fractions


def run_immucellai_from_dataframe(
    expression_df: pd.DataFrame,
    sample_type: str = "tumor",
    thread_num: int = 8,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run ImmuCellAI 2.0 on expression data provided as a DataFrame.
    
    Args:
        expression_df: TPM expression DataFrame with genes as rows and samples as columns.
                      Index should be gene symbols.
        sample_type: Either "tumor" or "normal".
        thread_num: Number of threads.
        seed: Random seed.
    
    Returns:
        DataFrame with immune cell abundances (samples x cell types).
    """
    check_immucellai_available()
    
    # Save to temp file
    temp_file = tempfile.mktemp(suffix=".txt")
    expression_df.to_csv(temp_file, sep="\t")
    
    try:
        result = run_immucellai(
            expression_file=temp_file,
            sample_type=sample_type,
            thread_num=thread_num,
            seed=seed
        )
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return result


def extract_immucell_features(
    expression_data: Union[str, pd.DataFrame],
    sample_type: str = "tumor",
    thread_num: int = 8,
    seed: int = 42,
    include_derived_scores: bool = True
) -> pd.DataFrame:
    """
    Extract immune cell composition features from gene expression data.
    
    This is the main entry point for feature extraction.
    
    Args:
        expression_data: Either path to TPM expression file, or DataFrame.
                        Format: genes (rows) x samples (columns).
        sample_type: "tumor" or "normal" sample type.
        thread_num: Number of threads for processing.
        seed: Random seed for reproducibility.
        include_derived_scores: If True, calculate additional derived scores.
    
    Returns:
        DataFrame with immune cell fractions and optional derived scores.
        Rows are samples, columns are features.
    """
    # Run deconvolution
    if isinstance(expression_data, str):
        cell_fractions = run_immucellai(
            expression_file=expression_data,
            sample_type=sample_type,
            thread_num=thread_num,
            seed=seed
        )
    else:
        cell_fractions = run_immucellai_from_dataframe(
            expression_df=expression_data,
            sample_type=sample_type,
            thread_num=thread_num,
            seed=seed
        )
    
    # Add derived scores if requested
    if include_derived_scores:
        cell_fractions = add_derived_immune_scores(cell_fractions)
    
    return cell_fractions


def add_derived_immune_scores(cell_fractions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional derived immune scores from cell fractions.
    
    These scores are commonly used for ICB response prediction.
    
    Args:
        cell_fractions: DataFrame with immune cell fractions.
    
    Returns:
        DataFrame with original fractions plus derived scores.
    """
    result = cell_fractions.copy()
    
    # Helper to safely get column or return zeros
    def safe_get(df, col):
        return df[col] if col in df.columns else pd.Series(0, index=df.index)
    
    # Cytotoxic Score: CD8 T cells + NK cells
    cd8 = safe_get(result, "CD8_T") + safe_get(result, "Exhausted_CD8_T")
    nk = safe_get(result, "NK")
    result["Cytotoxic_Score"] = cd8 + nk
    
    # Immunosuppressive Score: Tregs + M2 Macrophages + MDSCs
    treg = safe_get(result, "Treg")
    m2 = safe_get(result, "M2_Macrophage")
    mdsc = safe_get(result, "MDSC")
    result["Immunosuppressive_Score"] = treg + m2 + mdsc
    
    # Cytotoxic/Immunosuppressive Ratio
    suppressive = result["Immunosuppressive_Score"].replace(0, np.nan)
    result["Cytotoxic_Suppressive_Ratio"] = result["Cytotoxic_Score"] / suppressive
    result["Cytotoxic_Suppressive_Ratio"] = result["Cytotoxic_Suppressive_Ratio"].fillna(0)
    
    # T cell infiltration score
    t_cell_cols = ["CD8_T", "CD4_T", "Treg", "Th1", "Th2", "Th17", "Tfh", 
                   "Exhausted_CD8_T", "T_cell"]
    t_cell_sum = sum(safe_get(result, col) for col in t_cell_cols if col in result.columns)
    result["T_Cell_Infiltration"] = t_cell_sum
    
    # Hot vs Cold tumor proxy (high CD8 + low Treg = "hot")
    result["Hot_Tumor_Score"] = safe_get(result, "CD8_T") - safe_get(result, "Treg")
    
    # M1/M2 Macrophage Ratio
    m1 = safe_get(result, "M1_Macrophage")
    m2_safe = m2.replace(0, np.nan)
    result["M1_M2_Ratio"] = m1 / m2_safe
    result["M1_M2_Ratio"] = result["M1_M2_Ratio"].fillna(0)
    
    # Th1/Th2 Ratio
    th1 = safe_get(result, "Th1")
    th2 = safe_get(result, "Th2").replace(0, np.nan)
    result["Th1_Th2_Ratio"] = th1 / th2
    result["Th1_Th2_Ratio"] = result["Th1_Th2_Ratio"].fillna(0)
    
    return result


def get_icb_relevant_features(cell_fractions: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the features most relevant for ICB response prediction.
    
    Args:
        cell_fractions: Full DataFrame from ImmuCellAI deconvolution.
    
    Returns:
        DataFrame with ICB-relevant features only.
    """
    # Get available ICB-relevant columns
    available_cols = [col for col in ICB_RELEVANT_SUBTYPES if col in cell_fractions.columns]
    
    # Also include any derived scores if present
    derived_scores = [
        "Cytotoxic_Score", 
        "Immunosuppressive_Score",
        "Cytotoxic_Suppressive_Ratio",
        "T_Cell_Infiltration",
        "Hot_Tumor_Score",
        "M1_M2_Ratio",
        "Th1_Th2_Ratio"
    ]
    available_derived = [col for col in derived_scores if col in cell_fractions.columns]
    
    return cell_fractions[available_cols + available_derived]


def calculate_immune_score(cell_fractions: pd.DataFrame) -> pd.Series:
    """
    Calculate overall immune infiltration score.
    
    Args:
        cell_fractions: DataFrame with immune cell fractions.
    
    Returns:
        Series of immune scores per sample.
    """
    # Sum all immune cell fractions (excluding derived scores)
    immune_cols = [col for col in cell_fractions.columns 
                   if not col.endswith("_Score") and not col.endswith("_Ratio")]
    return cell_fractions[immune_cols].sum(axis=1)


def calculate_cytotoxic_score(cell_fractions: pd.DataFrame) -> pd.Series:
    """
    Calculate cytotoxic immune score from CD8 T cells and NK cells.
    
    Args:
        cell_fractions: DataFrame with immune cell fractions.
    
    Returns:
        Series of cytotoxic scores per sample.
    """
    score = pd.Series(0, index=cell_fractions.index)
    
    for col in ["CD8_T", "Exhausted_CD8_T", "NK"]:
        if col in cell_fractions.columns:
            score += cell_fractions[col]
    
    return score


def save_features(
    features: pd.DataFrame,
    output_path: str,
    format: str = "csv"
) -> None:
    """
    Save extracted features to file.
    
    Args:
        features: DataFrame with features.
        output_path: Path to save file.
        format: Output format ("csv", "tsv", or "excel").
    """
    if format == "csv":
        features.to_csv(output_path)
    elif format == "tsv":
        features.to_csv(output_path, sep="\t")
    elif format == "excel":
        features.to_excel(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract immune cell features using ImmuCellAI 2.0"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input TPM expression file (genes x samples, tab-separated)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path for immune cell features"
    )
    parser.add_argument(
        "--sample-type",
        choices=["tumor", "normal"],
        default="tumor",
        help="Sample type for reference selection (default: tumor)"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
        help="Number of threads (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv", "excel"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--icb-only",
        action="store_true",
        help="Output only ICB-relevant features"
    )
    
    args = parser.parse_args()
    
    print(f"Running ImmuCellAI 2.0 on {args.input}...")
    
    # Extract features
    features = extract_immucell_features(
        expression_data=args.input,
        sample_type=args.sample_type,
        thread_num=args.threads,
        seed=args.seed,
        include_derived_scores=True
    )
    
    # Filter to ICB-relevant if requested
    if args.icb_only:
        features = get_icb_relevant_features(features)
    
    # Save results
    save_features(features, args.output, args.format)
    
    print(f"Features saved to {args.output}")
    print(f"Extracted {features.shape[1]} features for {features.shape[0]} samples")
