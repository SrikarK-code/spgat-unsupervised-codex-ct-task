# # --- SETUP ---
# lib_loc <- "/hpc/home/vk93/R_libs"
# conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
# conda_r_lib <- paste0(conda_env, "/lib/R/library")
# .libPaths(c(lib_loc, conda_r_lib, .libPaths()))
# Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

# suppressPackageStartupMessages({
#   library(dplyr)
#   library(stringr)
#   library(readr)
# })

# # --- HELPER FUNCTION ---
# binarize_col <- function(x) {
#   q75 <- quantile(x, 0.75, na.rm = TRUE)
#   q25 <- quantile(x, 0.25, na.rm = TRUE)
#   res <- rep(NA, length(x))
#   res[x >= q75] <- 1
#   res[x <= q25] <- 0
#   return(res)
# }

# # --- 1. LOAD AND CLEAN DATA ---
# cat("\n[1] Loading and cleaning raw dataset...\n")
# df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)

# # Clean trailing whitespace from unique_region column
# df <- df %>% mutate(unique_region = str_trim(unique_region))

# marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

# # --- 2. CREATE OUTPUT DIRECTORY ---
# out_dir <- "tacit_priors"
# if (!dir.exists(out_dir)) {
#   dir.create(out_dir)
#   cat(sprintf("[2] Created directory: %s/\n", out_dir))
# } else {
#   cat(sprintf("[2] Directory %s/ already exists.\n", out_dir))
# }

# # --- 3. GENERATE INTRA-DONOR PRIOR (B004_Ascending) ---
# anchor_region <- "B004_Ascending"
# cat(sprintf("\n[3] Generating Intra-Donor Prior from Region: %s\n", anchor_region))

# df_anchor <- df %>% filter(unique_region == anchor_region)
# cat(sprintf("    -> Found %d cells.\n", nrow(df_anchor)))

# if (nrow(df_anchor) > 0) {
#   mean_expr_intra <- df_anchor %>%
#     group_by(`Cell Type`) %>%
#     summarise(across(all_of(marker_cols), ~mean(.x, na.rm = TRUE)))
    
#   intra_prior <- mean_expr_intra
#   intra_prior[marker_cols] <- lapply(intra_prior[marker_cols], binarize_col)
#   intra_prior <- intra_prior %>% rename(cell_type = `Cell Type`)
  
#   out_path <- file.path(out_dir, "tacit_prior_B004_Ascending.csv")
#   write_csv(intra_prior, out_path)
#   cat(sprintf("    -> Saved prior with %d cell types to %s\n", nrow(intra_prior), out_path))
# } else {
#   cat("    -> [ERROR] No cells found! Check region string matching.\n")
# }

# # --- 4. GENERATE INTER-DONOR PRIORS (B008, B012) ---
# macro_donors <- c("B008", "B012")

# for (d in macro_donors) {
#   cat(sprintf("\n[4] Generating Inter-Donor Prior from Donor: %s\n", d))
  
#   df_donor <- df %>% filter(donor == d)
#   cat(sprintf("    -> Found %d cells.\n", nrow(df_donor)))
  
#   if (nrow(df_donor) > 0) {
#     mean_expr_macro <- df_donor %>%
#       group_by(`Cell Type`) %>%
#       summarise(across(all_of(marker_cols), ~mean(.x, na.rm = TRUE)))
      
#     macro_prior <- mean_expr_macro
#     macro_prior[marker_cols] <- lapply(macro_prior[marker_cols], binarize_col)
#     macro_prior <- macro_prior %>% rename(cell_type = `Cell Type`)
    
#     out_path <- file.path(out_dir, sprintf("tacit_prior_%s.csv", d))
#     write_csv(macro_prior, out_path)
#     cat(sprintf("    -> Saved prior with %d cell types to %s\n", nrow(macro_prior), out_path))
#   } else {
#     cat("    -> [ERROR] No cells found for donor!\n")
#   }
# }

# cat("\n=== ALL PRIORS GENERATED SUCCESSFULLY ===\n")

# --- SETUP ---
lib_loc <- "/hpc/home/vk93/R_libs"
conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
conda_r_lib <- paste0(conda_env, "/lib/R/library")
.libPaths(c(lib_loc, conda_r_lib, .libPaths()))
Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

suppressPackageStartupMessages({
  library(Seurat)
  library(future)
  library(mclust)
  library(fossil)
  library(dplyr)
  library(readr)
  library(TACIT)
})

# Configure Future Plan
plan(sequential)
options(future.globals.maxSize = 8000 * 1024^2)

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

binarize_col <- function(x) {
  q75 <- quantile(x, 0.75, na.rm = TRUE)
  q25 <- quantile(x, 0.25, na.rm = TRUE)
  res <- rep(NA, length(x))
  res[x >= q75] <- 1
  res[x <= q25] <- 0
  return(res)
}

# --- LOAD DATA ---
cat("\n[1] Loading Base Data...\n")
df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)
marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

# --- CONFIG ---
anchor_reg <- "B004_Ascending"
df_b004_full <- df %>% filter(grepl("^B004", unique_region))
exact_b004_regions <- unique(df_b004_full$unique_region)

csv_file <- "tacit_meso_vs_macro_baselines.csv"

# Optional: Add resume logic so it doesn't overwrite if you stop and start
if(!file.exists(csv_file)) {
    writeLines("Prior_Type,Prior_Source,Test_Region,Total_ARI,Total_F1", csv_file)
}

# --- 1. GENERATE INTRA-DONOR PRIOR ---
cat(sprintf("\n[2] Generating Intra-Donor Prior from %s...\n", anchor_reg))

df_anchor <- df_b004_full %>% filter(unique_region == anchor_reg) 
if(nrow(df_anchor) == 0) stop("FATAL ERROR: Anchor region not found!")

mean_expr <- df_anchor %>%
  group_by(`Cell Type`) %>%
  summarise(across(all_of(marker_cols), ~mean(.x, na.rm = TRUE)))

intra_prior <- mean_expr
intra_prior[marker_cols] <- lapply(intra_prior[marker_cols], binarize_col)
intra_prior <- intra_prior %>% rename(cell_type = `Cell Type`)

# --- 2. RUN EVALUATIONS ---
priors_to_test <- c("B008", "B012")

for (target_reg in exact_b004_regions) {
  
  if (target_reg == anchor_reg) {
      cat(sprintf("\n--- Skipping Anchor Region (%s) to prevent data leakage ---\n", target_reg))
      next 
  }
  
  df_sub <- df_b004_full %>% filter(unique_region == target_reg)
  n_cells <- nrow(df_sub)
  
  if (n_cells < 10) {
      cat(sprintf("\n[WARNING] Skipping '%s' because it only has %d cells.\n", target_reg, n_cells))
      next
  }
  
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("Testing on Region: '%s' (%d cells)\n", target_reg, n_cells))
  cat(sprintf("==================================================\n"))
  
  true_labels <- df_sub$`Cell Type`
  CELLxFEATURE <- as.data.frame(df_sub[, marker_cols])
  rownames(CELLxFEATURE) <- paste0("C", seq_len(nrow(CELLxFEATURE)))
  
  # A. Test Intra-Donor Prior (Meso)
  cat(sprintf("  -> Testing Intra-Donor Prior (%s)...\n", anchor_reg))
  tryCatch({
      tacit_res_intra <- TACIT(CELLxFEATURE, intra_prior, r=10, p=10)
      pred_intra <- tacit_res_intra$TACIT
      pred_intra[is.na(pred_intra) | pred_intra == "Unknown"] <- "Unassigned"
      
      ari_intra <- adjustedRandIndex(true_labels, pred_intra)
      f1_intra <- calc_weighted_f1(true_labels, pred_intra)
      write(sprintf("INTRA_DONOR,%s,%s,%.4f,%.4f", anchor_reg, target_reg, ari_intra, f1_intra), file=csv_file, append=TRUE)
      cat(sprintf("     [SUCCESS] ARI: %.4f | F1: %.4f\n", ari_intra, f1_intra))
  }, error = function(e) {
      cat(sprintf("     [ERROR] TACIT Failed on Intra-Donor: %s\n", e$message))
  })
  
  # B. Test Inter-Donor Priors (Macro)
  for (donor_prior in priors_to_test) {
    prior_file <- file.path("tacit_priors", sprintf("tacit_prior_%s.csv", donor_prior))
    
    if (!file.exists(prior_file)) {
        cat(sprintf("  -> [ERROR] Cannot find file: %s\n", prior_file))
        next
    }
    
    cat(sprintf("  -> Testing Inter-Donor Prior (%s)...\n", donor_prior))
    
    # THE FIX IS HERE: No rename() attached to read_csv()
    tacit_prior_macro <- read_csv(prior_file, show_col_types = FALSE) 
    
    tryCatch({
        tacit_res_macro <- TACIT(CELLxFEATURE, tacit_prior_macro, r=10, p=10)
        pred_macro <- tacit_res_macro$TACIT
        pred_macro[is.na(pred_macro) | pred_macro == "Unknown"] <- "Unassigned"
        
        ari_macro <- adjustedRandIndex(true_labels, pred_macro)
        f1_macro <- calc_weighted_f1(true_labels, pred_macro)
        write(sprintf("INTER_DONOR,%s,%s,%.4f,%.4f", donor_prior, target_reg, ari_macro, f1_macro), file=csv_file, append=TRUE)
        cat(sprintf("     [SUCCESS] ARI: %.4f | F1: %.4f\n", ari_macro, f1_macro))
    }, error = function(e) {
        cat(sprintf("     [ERROR] TACIT Failed on Inter-Donor %s: %s\n", donor_prior, e$message))
    })
  }
}

cat("\n[3] Finished all TACIT Meso vs Macro baselines.\n")