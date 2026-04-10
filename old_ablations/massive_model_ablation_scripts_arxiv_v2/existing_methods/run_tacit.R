# # 1. Set up custom HPC library path
# lib_loc <- "/hpc/home/vk93/R_libs"
# if (!dir.exists(lib_loc)) dir.create(lib_loc, recursive = TRUE)
# .libPaths(c(lib_loc, .libPaths()))

# # 2. Smart Installer for Base Dependencies
# required_pkgs <- c("readr", "mclust", "fossil", "dplyr", "devtools", "caret", "cowplot", "segmented", "uwot", "Seurat")
# missing_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
# if(length(missing_pkgs) > 0) {
#   print(paste("Installing missing packages:", paste(missing_pkgs, collapse=", ")))
#   install.packages(missing_pkgs, lib = lib_loc, repos = "http://cran.us.r-project.org")
# }

# # 3. AUTO-PATCH AND INSTALL THE BROKEN TACIT PACKAGE
# if (!"TACIT" %in% installed.packages()[,"Package"]) {
#   print("Downloading and patching broken TACIT package from GitHub...")
  
#   # Download repo as zip
#   download.file("https://github.com/huynhkl953/TACIT/archive/refs/heads/main.zip", destfile = "TACIT.zip")
#   unzip("TACIT.zip")
  
#   # Read the broken NAMESPACE file
#   ns_file <- "TACIT-main/NAMESPACE"
#   ns_lines <- readLines(ns_file)
  
#   # Remove the 3 lines causing the crash
#   bad_lines <- c("export(TACIT_visualization)", "export(heatmap_anb)", "export(plot_marker)")
#   ns_lines <- ns_lines[!(ns_lines %in% bad_lines)]
  
#   # Write the fixed file back
#   writeLines(ns_lines, ns_file)
  
#   # Install the patched package locally
#   install.packages("TACIT-main", repos = NULL, type = "source", lib = lib_loc)
  
#   # Clean up downloaded files
#   unlink("TACIT.zip")
#   unlink("TACIT-main", recursive = TRUE)
# }

# # 4. Load Libraries
# library(TACIT)
# library(readr)
# library(mclust)
# library(fossil)
# library(dplyr)




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

# # Raise memory limit for parallel processing
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
# CELLxFEATURE <- df_b004[, marker_cols]

# results <- list()
# donors <- c("B008", "B012")

# # --- RUN TACIT ---
# for (d in donors) {
#   print(sprintf("Running TACIT with %s Prior...", d))
#   tacit_prior <- read_csv(sprintf("tacit_prior_%s.csv", d), show_col_types = FALSE)
  
#   # FIX: Rename the column so TACIT's internal code doesn't crash
#   tacit_prior <- tacit_prior %>% rename(cell_type = `Cell Type`)
  
#   tacit_res <- TACIT(CELLxFEATURE, tacit_prior, r=10, p=10)
#   pred_tacit <- tacit_res$data_clean_final$Cell_Type
#   pred_tacit[is.na(pred_tacit)] <- "Unassigned"
  
#   ari_tacit <- adjustedRandIndex(true_labels, pred_tacit)
#   ri_tacit <- rand.index(as.numeric(as.factor(true_labels)), as.numeric(as.factor(pred_tacit)))
#   f1_tacit <- calc_weighted_f1(true_labels, pred_tacit)
  
#   results[[sprintf("TACIT_Prior_%s", d)]] <- c(ari_tacit, ri_tacit, f1_tacit)
# }

# cat("\n======================================================\n")
# cat("TACIT LEADERBOARD (Tested on B004)\n")
# cat("======================================================\n")
# for (name in names(results)) {
#   metrics <- results[[name]]
#   cat(sprintf("%-20s : ARI=%.4f | RI=%.4f | Weighted F1=%.4f\n", name, metrics[1], metrics[2], metrics[3]))
# }












# --- SETUP ---
lib_loc <- "/hpc/home/vk93/R_libs"
conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
conda_r_lib <- paste0(conda_env, "/lib/R/library")
.libPaths(c(lib_loc, conda_r_lib, .libPaths()))
Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

# --- LIBRARIES ---
library(Seurat)
library(future)    
library(TACIT)
library(readr)
library(mclust)
library(fossil)
library(dplyr)

# Raise memory limit for parallel processing
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

# --- LOAD DATA ---
print("Loading B004 Ground Truth Data...")
df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)
df_b004 <- df %>% filter(grepl("^B004", unique_region))

marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

true_labels <- df_b004$`Cell Type`

# FIX: Force explicit rownames so TACIT doesn't lose track of which cell is which
CELLxFEATURE <- as.data.frame(df_b004[, marker_cols])
rownames(CELLxFEATURE) <- paste0("C", seq_len(nrow(CELLxFEATURE)))

results <- list()
donors <- c("B008", "B012")

# --- RUN TACIT ---
for (d in donors) {
  print(sprintf("Running TACIT with %s Prior...", d))
  tacit_prior <- read_csv(sprintf("tacit_prior_%s.csv", d), show_col_types = FALSE)
  
  # Rename the column so TACIT's internal code doesn't crash
  tacit_prior <- tacit_prior %>% rename(cell_type = `Cell Type`)
  
  tacit_res <- TACIT(CELLxFEATURE, tacit_prior, r=10, p=10)
  
  # FIX: Create an array of "Unassigned" for ALL 248,287 cells
  pred_tacit_named <- rep("Unassigned", nrow(CELLxFEATURE))
  names(pred_tacit_named) <- rownames(CELLxFEATURE)
  
  # Map the 92k "clean" cells TACIT found back to their exact original position
  clean_cells <- rownames(tacit_res$data_clean_final)
  pred_tacit_named[clean_cells] <- tacit_res$data_clean_final$Cell_Type
  
  # Strip the names to create a pure vector for metric calculation
  pred_tacit <- unname(pred_tacit_named)
  
  ari_tacit <- adjustedRandIndex(true_labels, pred_tacit)
  ri_tacit <- rand.index(as.numeric(as.factor(true_labels)), as.numeric(as.factor(pred_tacit)))
  f1_tacit <- calc_weighted_f1(true_labels, pred_tacit)
  
  results[[sprintf("TACIT_Prior_%s", d)]] <- c(ari_tacit, ri_tacit, f1_tacit)
}

cat("\n======================================================\n")
cat("TACIT LEADERBOARD (Tested on B004)\n")
cat("======================================================\n")
for (name in names(results)) {
  metrics <- results[[name]]
  cat(sprintf("%-20s : ARI=%.4f | RI=%.4f | Weighted F1=%.4f\n", name, metrics[1], metrics[2], metrics[3]))
}