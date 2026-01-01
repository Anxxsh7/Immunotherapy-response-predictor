# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPRES.R - IMmune Predictive Response Score Feature Extraction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Calculates the IMPRES score based on checkpoint gene pair comparisons.
# Reference: https://www.nature.com/articles/s41591-018-0157-9
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load required libraries
library(data.table)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# default_ckpt_pairs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Get the default gene pairs for the impres calculation
#' 
#' @return Returns data frame/table with 2 columns: gene1_list & gene2_list
#' 
#' @export
default_ckpt_pairs <- function() {
  gene1_list <- c("PDCD1", "CD27", "CTLA4", "CD40", "CD86", "CD28", "CD80", "CD274", "CD86", "CD40",
                  "CD86", "CD40", "CD28", "CD40", "TNFRSF14")
  gene2_list <- c("TNFSF4", "PDCD1", "TNFSF4", "CD28", "TNFSF4", "CD86", "TNFSF9", "VSIR", "HAVCR2",
                  "PDCD1", "CD200", "CD80", "CD276", "CD274", "CD86")
  return(data.table(gene1_list = gene1_list, gene2_list = gene2_list))
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# calc_impres
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#' @title Calculate an IMPRES score
#' 
#' @description The IMPRES score was created by this study: https://www.nature.com/articles/s41591-018-0157-9.  
#' Where melanoma patients on with higher IMPRES scores had better survival when treated with ICI 
#' across multiple datasets. A score of 1 means that - out of all of the pairs of genes in ckpt_pair_dt - in only 
#' one case was the expression of the first gene higher than the second.  2 means that 2 pairs were 
#' higher in the first gene than the second. Etc... NA's in gene expression will be treated as zeroes.
#' 
#' @param ge_df Gene expression data.frame or data.table. Each column is a gene, each row is a sample.
#'   Must include the sample column indicated by \code{sample_key}.
#' @param ckpt_pair_dt optional data frame with user defined gene comparisons
#' @param require_all_genes Boolean to indicate if the function should proceed if some genes are missing.
#' @param sample_key Character string to specify the column that is the sample key.
#' @param gene_element For gene names that have a pipe in them, which position should be used ("1|2 etc"). Integer.
#' @param include_continuous Boolean to include continuous difference features (gene1 - gene2) for each pair.
#'   Positive values indicate gene1 > gene2, negative values indicate gene1 < gene2.
#' 
#' @return Returns data frame/table with sample ID, IMPRES score, and optionally continuous difference features
#'   for each gene pair (if include_continuous = TRUE).
#' 
#' @export
calc_impres <- function(
  ge_df,
  ckpt_pair_dt = default_ckpt_pairs(),
  gene_element = 1,
  require_all_genes = TRUE,
  sample_key = "sample_id",
  include_continuous = TRUE
) {
  
  # Convert to data.table if needed and track input type
  if (is.data.frame(ge_df) && !is.data.table(ge_df)) {
    input_type <- "DF"
    ge_df <- as.data.table(ge_df)
  } else if (is.data.table(ge_df)) {
    input_type <- "DT"
  } else {
    stop("ge_df must be a data.frame or data.table")
  }
  
  # Check sample_key exists
  if (!(sample_key %in% names(ge_df))) {
    stop(paste0(sample_key, " must be a column in ge_df."))
  }
  
  # Validate ckpt_pair_dt
  if (!is.data.frame(ckpt_pair_dt) && !is.data.table(ckpt_pair_dt)) {
    stop("ckpt_pair_dt must be a data.table or data.frame.")
  }
  ckpt_pair_dt <- as.data.frame(ckpt_pair_dt)

  # Get the first part of the names in case these are piped-names 
  names(ge_df) <- sapply(names(ge_df), function(x) {
    strsplit(x, '|', fixed = TRUE)[[1]][gene_element]
  })
  
  # Check for missing genes first, and list all the ones that are missing
  all_genes <- unlist(ckpt_pair_dt)
  missing_genes <- all_genes[!(all_genes %in% names(ge_df))]
  
  if (length(missing_genes) > 0) {
    warning(paste0("Your gene data is missing the following columns: \n", paste0(missing_genes, collapse = "\n")))
    if (require_all_genes) {
      stop("Terminating calc_impres.")
    } else {
      # Drop the missing rows
      cat("Proceeding without the missing gene(s).\n")
      rows_of_missing_genes <- unique(c(
        which(ckpt_pair_dt[[1]] %in% missing_genes), 
        which(ckpt_pair_dt[[2]] %in% missing_genes)
      ))
      if (length(rows_of_missing_genes) > 0) {
        ckpt_pair_dt <- ckpt_pair_dt[-rows_of_missing_genes, ]
      }
      if (nrow(ckpt_pair_dt) == 0) {
        stop("You have no valid gene pair combinations to compute a score.")
      }
    }
  }
  
  
  return_df <- ge_df[, sample_key, with = FALSE]
  return_df$IMPRES_Score <- NA
  return_df <- as.data.frame(return_df)
  
  # Initialize continuous difference columns if requested
  if (include_continuous) {
    for (pair_index in seq_len(nrow(ckpt_pair_dt))) {
      gene1 <- ckpt_pair_dt[[1]][pair_index]
      gene2 <- ckpt_pair_dt[[2]][pair_index]
      col_name <- paste0(gene1, "_minus_", gene2)
      return_df[[col_name]] <- NA
    }
    # Also add a sum of all differences as an aggregate continuous score
    return_df$IMPRES_Continuous_Sum <- NA
  }

  for (sample_index in seq_len(nrow(ge_df))) {
    my_value <- 0
    continuous_sum <- 0
    for (pair_index in seq_len(nrow(ckpt_pair_dt))) {
      gene1 <- ckpt_pair_dt[[1]][pair_index]
      gene2 <- ckpt_pair_dt[[2]][pair_index]
      gene1_value <- ge_df[[gene1]][sample_index]
      gene2_value <- ge_df[[gene2]][sample_index]
      
      # Handle NA values - treat as zero
      if (is.na(gene1_value)) gene1_value <- 0
      if (is.na(gene2_value)) gene2_value <- 0
      
      # Calculate continuous difference (gene1 - gene2)
      difference <- gene1_value - gene2_value
      
      # Store continuous difference if requested
      if (include_continuous) {
        col_name <- paste0(gene1, "_minus_", gene2)
        return_df[[col_name]][sample_index] <- difference
        continuous_sum <- continuous_sum + difference
      }
      
      # Binary comparison for IMPRES score
      if (gene1_value > gene2_value) {
        my_value <- my_value + 1
      }
    }      
    return_df$IMPRES_Score[sample_index] <- my_value
    if (include_continuous) {
      return_df$IMPRES_Continuous_Sum[sample_index] <- continuous_sum
    }
  }
  

  if (input_type == "DF") {
    return(return_df)
  } else {
    return(as.data.table(return_df))
  }
}