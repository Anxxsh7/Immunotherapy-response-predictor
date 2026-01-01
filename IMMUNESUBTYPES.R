# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMMUNESUBTYPES.R - Immune Subtype Classification Feature Extraction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Uses the ImmuneSubtypeClassifier R package to classify RNA-seq samples
# into one of 6 immune subtypes and extract subtype probabilities as features.
#
# The 6 immune subtypes from "The Immune Landscape of Cancer" (Thorsson et al., 2018):
#   C1: Wound Healing
#   C2: IFN-gamma Dominant
#   C3: Inflammatory
#   C4: Lymphocyte Depleted
#   C5: Immunologically Quiet
#   C6: TGF-beta Dominant
#
# Reference: https://github.com/CRI-iAtlas/ImmuneSubtypeClassifier
#
# Installation:
#   library(devtools)
#   install_github("CRI-iAtlas/ImmuneSubtypeClassifier")
#   devtools::install_version("xgboost", version = "1.0.0.1")
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load required package (will error if not installed)
if (!requireNamespace("ImmuneSubtypeClassifier", quietly = TRUE)) {
  stop("ImmuneSubtypeClassifier package is required. Install with:\n",
       "  library(devtools)\n",
       "  install_github('CRI-iAtlas/ImmuneSubtypeClassifier')\n",
       "  devtools::install_version('xgboost', version = '1.0.0.1')")
}
library(ImmuneSubtypeClassifier)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get_required_genes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Get the 485 genes required by ImmuneSubtypeClassifier
#'
#' @return Character vector of required gene symbols
#'
#' @export
get_required_genes <- function() {
  data("ebpp_genes_sig", package = "ImmuneSubtypeClassifier", envir = environment())
  return(get("ebpp_genes_sig", envir = environment()))
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# check_gene_coverage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Check how many required genes are present in your data
#'
#' @param expression_matrix Matrix with genes as rows (rownames) and samples as columns
#' @param gene_id_type Type of gene IDs: 'symbol', 'entrez', or 'ensembl'
#'
#' @return List with match_error (proportion missing) and missing_genes (data.frame)
#'
#' @export
check_gene_coverage <- function(expression_matrix, gene_id_type = "symbol") {
  report <- ImmuneSubtypeClassifier::geneMatchErrorReport(
    X = as.matrix(expression_matrix), 
    geneid = gene_id_type
  )
  
  cat("Missing gene proportion:", round(report$matchError * 100, 2), "%\n")
  
  if (nrow(report$missingGenes) > 0) {
    cat("Number of missing genes:", nrow(report$missingGenes), "\n")
  }
  
  return(report)
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# extract_immune_subtype_features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Extract immune subtype probability features
#'
#' @description Classifies samples into 6 immune subtypes and returns
#' the probability of belonging to each subtype. These probabilities
#' can be used as features for downstream ML models.
#'
#' @param expression_matrix Matrix or data.frame with genes as rows (rownames = gene IDs)
#'   and samples as columns. Expression values should be log2-transformed.
#' @param gene_id_type Type of gene IDs in rownames: 'symbol', 'entrez', or 'ensembl'
#' @param rename_columns If TRUE, rename probability columns to descriptive names
#'
#' @return Data.frame with:
#'   - SampleIDs: sample identifiers
#'   - BestCall: predicted subtype (1-6)
#'   - Probability columns for each subtype (6 columns)
#'
#' @examples
#' \dontrun{
#' # Load expression data
#' expr <- read.table("expression.txt", header=TRUE, row.names=1, sep="\t")
#' 
#' # Extract features
#' features <- extract_immune_subtype_features(expr, gene_id_type = "symbol")
#' }
#'
#' @export
extract_immune_subtype_features <- function(expression_matrix, 
                                            gene_id_type = "symbol",
                                            rename_columns = TRUE) {
  
  # Ensure matrix format
  expr_mat <- as.matrix(expression_matrix)
  
  # Run the classifier
  results <- ImmuneSubtypeClassifier::callEnsemble(X = expr_mat, geneids = gene_id_type)
  
  # Rename columns to descriptive names if requested
  if (rename_columns) {
    subtype_names <- c(
      "1" = "C1_Wound_Healing",
      "2" = "C2_IFN_gamma_Dominant",
      "3" = "C3_Inflammatory",
      "4" = "C4_Lymphocyte_Depleted",
      "5" = "C5_Immunologically_Quiet",
      "6" = "C6_TGF_beta_Dominant"
    )
    
    # Rename the probability columns
    for (old_name in names(subtype_names)) {
      if (old_name %in% colnames(results)) {
        colnames(results)[colnames(results) == old_name] <- subtype_names[old_name]
      }
    }
  }
  
  return(results)
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get_probability_features_only
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Get only the probability features (no SampleIDs or BestCall)
#'
#' @param results Output from extract_immune_subtype_features
#'
#' @return Data.frame with only the 6 probability columns, rownames = SampleIDs
#'
#' @export
get_probability_features_only <- function(results) {
  # Set sample IDs as rownames
  prob_features <- results
  rownames(prob_features) <- prob_features$SampleIDs
  
 # Keep only probability columns (exclude SampleIDs and BestCall)
  prob_cols <- grep("^C[1-6]_|^[1-6]$", colnames(prob_features), value = TRUE)
  prob_features <- prob_features[, prob_cols, drop = FALSE]
  
  return(prob_features)
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# run_immune_subtype_classification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Main function to run classification from file
#'
#' @param input_file Path to expression file (genes x samples, tab-separated)
#' @param output_file Path to save output features
#' @param gene_id_type Type of gene IDs: 'symbol', 'entrez', or 'ensembl'
#' @param check_genes If TRUE, report gene coverage before classification
#'
#' @return Data.frame with immune subtype features
#'
#' @export
run_immune_subtype_classification <- function(input_file,
                                               output_file,
                                               gene_id_type = "symbol",
                                               check_genes = TRUE) {
  
  cat("Loading expression data from:", input_file, "\n")
  expression_matrix <- read.table(input_file, header = TRUE, row.names = 1, sep = "\t")
  
  cat("Expression matrix dimensions:", nrow(expression_matrix), "genes x", 
      ncol(expression_matrix), "samples\n")
  
  # Check gene coverage if requested
  if (check_genes) {
    cat("\nChecking gene coverage...\n")
    check_gene_coverage(expression_matrix, gene_id_type)
  }
  
  # Run classification
  cat("\nRunning ImmuneSubtypeClassifier...\n")
  features <- extract_immune_subtype_features(expression_matrix, gene_id_type)
  
  # Save results
  write.csv(features, output_file, row.names = FALSE)
  cat("\nFeatures saved to:", output_file, "\n")
  cat("Output dimensions:", nrow(features), "samples x", ncol(features), "columns\n")
  
  return(features)
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example Usage (when running script directly)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# # Option 1: Simple usage
# expr <- read.table("expression.txt", header=TRUE, row.names=1, sep="\t")
# results <- callEnsemble(X = as.matrix(expr), geneids = "symbol")
# write.csv(results, "immune_subtype_features.csv", row.names = FALSE)
#
# # Option 2: Using wrapper functions
# features <- run_immune_subtype_classification(
#   input_file = "expression.txt",
#   output_file = "immune_subtype_features.csv",
#   gene_id_type = "symbol"
# )
#
# # Option 3: Get just probability features for ML
# prob_features <- get_probability_features_only(features)
# # Returns 6 columns: C1_Wound_Healing, C2_IFN_gamma_Dominant, etc.
#
