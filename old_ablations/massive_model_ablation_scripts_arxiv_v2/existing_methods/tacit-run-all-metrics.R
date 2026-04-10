# # --- SETUP ---
# lib_loc <- "/hpc/home/vk93/R_libs"
# conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
# conda_r_lib <- paste0(conda_env, "/lib/R/library")
# .libPaths(c(lib_loc, conda_r_lib, .libPaths()))
# Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

# # --- LIBRARIES ---
# library(Seurat)
# library(future)    
# library(TACIT)
# library(readr)
# library(mclust)
# library(fossil)
# library(dplyr)

# options(future.globals.maxSize = 8000 * 1024^2)

# # --- METRICS FUNCTION ---
# calc_weighted_f1 <- function(true_labels, pred_labels) {
#   classes <- unique(true_labels)
#   total_cells <- length(true_labels)
#   f1_scores <- numeric(length(classes))
#   weights <- numeric(length(classes))
#   for (i in seq_along(classes)) {
#     cls <- classes[i]
#     tp <- sum(true_labels == cls & pred_labels == cls)
#     fp <- sum(true_labels != cls & pred_labels == cls)
#     fn <- sum(true_labels == cls & pred_labels != cls)
#     precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
#     recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
#     f1_scores[i] <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
#     weights[i] <- sum(true_labels == cls) / total_cells
#   }
#   return(sum(f1_scores * weights))
# }

# # --- LOAD DATA ---
# print("Loading B004 Ground Truth Data...")
# df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)
# df_b004 <- df %>% filter(grepl("^B004", unique_region))

# marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

# true_labels <- df_b004$`Cell Type`

# # Pure data frame to protect rownames (NO VIRTUAL_ID added to the biological matrix)
# CELLxFEATURE <- as.data.frame(df_b004[, marker_cols])
# rownames(CELLxFEATURE) <- paste0("C", seq_len(nrow(CELLxFEATURE)))

# results <- list()
# donors <- c("B008", "B012")

# # --- RUN TACIT ---
# for (d in donors) {
#   cat(sprintf("\n\n======================================================\n"))
#   cat(sprintf("Evaluating TACIT with %s Prior...\n", d))
#   cat(sprintf("======================================================\n"))
  
#   rds_file <- sprintf("tacit_results_%s.rds", d)
  
#   if (file.exists(rds_file)) {
#     cat(sprintf("[DEBUG] Found cached results for %s! Loading instantly...\n", d))
#     tacit_res <- readRDS(rds_file)
#   } else {
#     cat(sprintf("[DEBUG] No cache found for %s. Running full TACIT pipeline...\n", d))
#     tacit_prior <- read_csv(sprintf("tacit_prior_%s.csv", d), show_col_types = FALSE)
#     tacit_prior <- tacit_prior %>% rename(cell_type = `Cell Type`)
#     tacit_res <- TACIT(CELLxFEATURE, tacit_prior, r=10, p=10)
#     saveRDS(tacit_res, rds_file)
#     cat(sprintf("[DEBUG] Pipeline finished. Saved to %s\n", rds_file))
#   }
  
#   # Initialize master array with "Unassigned"
#   pred_tacit_named <- rep("Unassigned", nrow(CELLxFEATURE))
#   names(pred_tacit_named) <- rownames(CELLxFEATURE)
  
#   # 1. Map the CLEAN cells precisely by Row Name
#   clean_cells <- c()
#   if (!is.null(tacit_res$data_clean_final) && nrow(tacit_res$data_clean_final) > 0) {
#     clean_cells <- rownames(tacit_res$data_clean_final)
#     pred_tacit_named[clean_cells] <- tacit_res$data_clean_final$Cell_Type
#   }
#   cat(sprintf("[DEBUG] Extracted %d CLEAN cells.\n", length(clean_cells)))
  
#   # 2. Map the MIXED cells precisely by Row Name
#   mixed_cells <- c()
#   if (!is.null(tacit_res$data_mixed_final) && nrow(tacit_res$data_mixed_final) > 0) {
#     mixed_cells <- rownames(tacit_res$data_mixed_final)
#     pred_tacit_named[mixed_cells] <- tacit_res$data_mixed_final$Cell_Type
#   }
#   cat(sprintf("[DEBUG] Extracted %d MIXED cells.\n", length(mixed_cells)))
  
#   # Create absolute boolean masks based on exact Cell IDs
#   mask_clean <- names(pred_tacit_named) %in% clean_cells
#   mask_mixed <- names(pred_tacit_named) %in% mixed_cells
#   mask_total <- rep(TRUE, length(pred_tacit_named))
  
#   # Strip names for pure vector math
#   pred_tacit <- unname(pred_tacit_named)
  
#   # Verify lengths before math
#   cat(sprintf("[DEBUG] Total Labels Length: %d | Total Preds Length: %d\n", length(true_labels), length(pred_tacit)))
  
#   # Print a sneak peek of the mapping
#   cat(sprintf("[DEBUG] Sneak peek of the first 5 cells:\n"))
#   print(head(data.frame(GroundTruth=true_labels, TACIT_Pred=pred_tacit), 5))
  
#   # Calculate Metrics
#   calc_ari <- function(mask) {
#     if (sum(mask) == 0) return(0)
#     return(adjustedRandIndex(true_labels[mask], pred_tacit[mask]))
#   }
#   calc_f1 <- function(mask) {
#     if (sum(mask) == 0) return(0)
#     return(calc_weighted_f1(true_labels[mask], pred_tacit[mask]))
#   }
  
#   ari_tot <- calc_ari(mask_total)
#   ari_cln <- calc_ari(mask_clean)
#   ari_mix <- calc_ari(mask_mixed)
  
#   f1_tot <- calc_f1(mask_total)
#   f1_cln <- calc_f1(mask_clean)
#   f1_mix <- calc_f1(mask_mixed)
  
#   # Print intermediate results so you don't have to wait for both donors to finish
#   cat(sprintf("\n[LIVE RESULT for %s] -> [TOTAL] ARI: %.4f | F1: %.4f\n", d, ari_tot, f1_tot))
  
#   results[[sprintf("TACIT_Prior_%s", d)]] <- c(ari_tot, ari_cln, ari_mix, f1_tot, f1_cln, f1_mix)
# }

# cat("\n\n======================================================\n")
# cat("FINAL TACIT LEADERBOARD (Tested on B004)\n")
# cat("======================================================\n")
# for (name in names(results)) {
#   m <- results[[name]]
#   cat(sprintf("%-20s : [TOTAL] ARI=%.4f F1=%.4f | [CLEAN] ARI=%.4f F1=%.4f | [MIXED] ARI=%.4f F1=%.4f\n", 
#               name, m[1], m[4], m[2], m[5], m[3], m[6]))
# }




######### check and debug script post rds

# # --- SETUP ---
# lib_loc <- "/hpc/home/vk93/R_libs"
# conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
# conda_r_lib <- paste0(conda_env, "/lib/R/library")
# .libPaths(c(lib_loc, conda_r_lib, .libPaths()))
# Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

# # --- LIBRARIES ---
# library(Seurat)
# library(future)    
# library(TACIT)
# library(readr)
# library(mclust)
# library(fossil)
# library(dplyr)

# # --- METRICS FUNCTION ---
# calc_weighted_f1 <- function(true_labels, pred_labels) {
#   classes <- unique(true_labels)
#   total_cells <- length(true_labels)
#   f1_scores <- numeric(length(classes))
#   weights <- numeric(length(classes))
#   for (i in seq_along(classes)) {
#     cls <- classes[i]
#     tp <- sum(true_labels == cls & pred_labels == cls)
#     fp <- sum(true_labels != cls & pred_labels == cls)
#     fn <- sum(true_labels == cls & pred_labels != cls)
#     precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
#     recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
#     f1_scores[i] <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
#     weights[i] <- sum(true_labels == cls) / total_cells
#   }
#   return(sum(f1_scores * weights))
# }

# # --- LOAD DATA ---
# print("Loading B004 Ground Truth Data...")
# df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)
# df_b004 <- df %>% filter(grepl("^B004", unique_region))

# marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

# true_labels <- df_b004$`Cell Type`

# CELLxFEATURE <- as.data.frame(df_b004[, marker_cols])
# target_ids <- paste0("C", seq_len(nrow(CELLxFEATURE)))
# rownames(CELLxFEATURE) <- target_ids

# results <- list()
# donors <- c("B008", "B012")

# # --- SEARCH AND RESCUE FUNCTION ---
# rescue_ids <- function(df, df_name) {
#   if (is.null(df) || nrow(df) == 0) return(NULL)
  
#   # 1. Check Rownames first
#   if (any(rownames(df) %in% target_ids)) {
#     cat(sprintf("[DEBUG] Found original IDs in ROWNAMES of %s.\n", df_name))
#     return(rownames(df))
#   }
  
#   # 2. Check every single column
#   for (col in colnames(df)) {
#     vals <- as.character(df[[col]])
#     if (any(vals %in% target_ids)) {
#       cat(sprintf("[DEBUG] Found original IDs in COLUMN '%s' of %s.\n", col, df_name))
#       return(vals)
#     }
#   }
  
#   # 3. If missing, dump the architecture so we can see what TACIT is doing
#   cat(sprintf("\n[FATAL] IDs completely missing from %s! Dumping structure:\n", df_name))
#   cat("Columns Available: ", paste(colnames(df), collapse=", "), "\n")
#   cat("First 3 rows (first 5 columns):\n")
#   print(head(df[, 1:min(5, ncol(df))], 3))
#   return(NULL)
# }

# # --- RUN METRICS ---
# for (d in donors) {
#   cat(sprintf("\n\n======================================================\n"))
#   cat(sprintf("Evaluating TACIT Cached Results for %s\n", d))
#   cat(sprintf("======================================================\n"))
  
#   rds_file <- sprintf("tacit_results_%s.rds", d)
#   if (!file.exists(rds_file)) {
#       cat("[ERROR] Could not find the .rds file. Skipping...\n")
#       next
#   }
  
#   tacit_res <- readRDS(rds_file)
  
#   cat("[DEBUG] Objects available inside tacit_res:\n")
#   print(names(tacit_res))
  
#   # Initialize master array
#   pred_tacit_named <- rep("Unassigned", nrow(CELLxFEATURE))
#   names(pred_tacit_named) <- target_ids
  
#   # Execute Search & Rescue
#   clean_cells <- rescue_ids(tacit_res$data_clean_final, "data_clean_final")
#   mixed_cells <- rescue_ids(tacit_res$data_mixed_final, "data_mixed_final")
  
#   if (!is.null(clean_cells)) {
#       pred_tacit_named[clean_cells] <- tacit_res$data_clean_final$Cell_Type
#       cat(sprintf("[DEBUG] Successfully mapped %d CLEAN cells.\n", length(clean_cells)))
#   }
#   if (!is.null(mixed_cells)) {
#       pred_tacit_named[mixed_cells] <- tacit_res$data_mixed_final$Cell_Type
#       cat(sprintf("[DEBUG] Successfully mapped %d MIXED cells.\n", length(mixed_cells)))
#   }
  
#   # Masks
#   mask_clean <- names(pred_tacit_named) %in% clean_cells
#   mask_mixed <- names(pred_tacit_named) %in% mixed_cells
#   mask_total <- rep(TRUE, length(pred_tacit_named))
  
#   # Strip names for pure vector math
#   pred_tacit <- unname(pred_tacit_named)
  
#   # Calculate Metrics
#   calc_ari <- function(mask) {
#     if (sum(mask) == 0) return(0)
#     return(adjustedRandIndex(true_labels[mask], pred_tacit[mask]))
#   }
#   calc_f1 <- function(mask) {
#     if (sum(mask) == 0) return(0)
#     return(calc_weighted_f1(true_labels[mask], pred_tacit[mask]))
#   }
  
#   ari_tot <- calc_ari(mask_total)
#   ari_cln <- calc_ari(mask_clean)
#   ari_mix <- calc_ari(mask_mixed)
#   f1_tot <- calc_f1(mask_total)
#   f1_cln <- calc_f1(mask_clean)
#   f1_mix <- calc_f1(mask_mixed)
  
#   results[[sprintf("TACIT_Prior_%s", d)]] <- c(ari_tot, ari_cln, ari_mix, f1_tot, f1_cln, f1_mix)
# }

# cat("\n\n======================================================\n")
# cat("FINAL TACIT LEADERBOARD (Tested on B004)\n")
# cat("======================================================\n")
# for (name in names(results)) {
#   m <- results[[name]]
#   cat(sprintf("%-20s : [TOTAL] ARI=%.4f F1=%.4f | [CLEAN] ARI=%.4f F1=%.4f | [MIXED] ARI=%.4f F1=%.4f\n", 
#               name, m[1], m[4], m[2], m[5], m[3], m[6]))
# }



# --- SETUP ---
lib_loc <- "/hpc/home/vk93/R_libs"
conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
conda_r_lib <- paste0(conda_env, "/lib/R/library")
.libPaths(c(lib_loc, conda_r_lib, .libPaths()))
Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

# --- LIBRARIES ---
library(mclust)
library(fossil)
library(dplyr)
library(readr)

# --- METRICS FUNCTION ---
calc_weighted_f1 <- function(true_labels, pred_labels) {
  classes <- unique(true_labels)
  total_cells <- length(true_labels)
  f1_scores <- numeric(length(classes))
  weights <- numeric(length(classes))
  for (i in seq_along(classes)) {
    cls <- classes[i]
    tp <- sum(true_labels == cls & pred_labels == cls)
    fp <- sum(true_labels != cls & pred_labels == cls)
    fn <- sum(true_labels == cls & pred_labels != cls)
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    f1_scores[i] <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    weights[i] <- sum(true_labels == cls) / total_cells
  }
  return(sum(f1_scores * weights))
}

# --- LOAD GROUND TRUTH ---
print("Loading B004 Ground Truth Data...")
df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)
df_b004 <- df %>% filter(grepl("^B004", unique_region))
true_labels <- df_b004$`Cell Type`

results <- list()
donors <- c("B008", "B012")

# --- EXTRACT METRICS FROM CACHE ---
for (d in donors) {
  rds_file <- sprintf("tacit_results_%s.rds", d)
  
  if (!file.exists(rds_file)) {
      cat(sprintf("[ERROR] Could not find %s. Skipping...\n", rds_file))
      next
  }
  
  # Load the cached dataframe
  tacit_res <- readRDS(rds_file)
  
  # Extract the predictions from the "TACIT" column
  pred_tacit <- tacit_res$TACIT
  
  # Clean up any NA or "Unknown" values just in case
  pred_tacit[is.na(pred_tacit) | pred_tacit == "Unknown"] <- "Unassigned"
  
  # Calculate Total Metrics
  ari_tot <- adjustedRandIndex(true_labels, pred_tacit)
  f1_tot <- calc_weighted_f1(true_labels, pred_tacit)
  
  results[[sprintf("TACIT_Prior_%s", d)]] <- c(ari_tot, f1_tot)
}

cat("\n======================================================\n")
cat("FINAL TACIT LEADERBOARD (Tested on B004)\n")
cat("======================================================\n")
for (name in names(results)) {
  m <- results[[name]]
  cat(sprintf("%-20s : [TOTAL] ARI=%.4f | F1=%.4f\n", name, m[1], m[2]))
}