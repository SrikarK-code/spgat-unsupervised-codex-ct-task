# # 1. Automatically clear the broken locks from the last attempt
# unlink("/hpc/home/vk93/R_libs/00LOCK*", recursive = TRUE)
# unlink("/hpc/home/vk93/R_libs/sf", recursive = TRUE)
# unlink("/hpc/home/vk93/R_libs/units", recursive = TRUE)
# unlink("/hpc/home/vk93/R_libs/spdep", recursive = TRUE)

# # 2. Set up custom HPC library path
# lib_loc <- "/hpc/home/vk93/R_libs"
# if (!dir.exists(lib_loc)) dir.create(lib_loc, recursive = TRUE)

# # 3. The Magic Bridge: Tell the HPC R where your Micromamba environment is
# conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
# conda_r_lib <- paste0(conda_env, "/lib/R/library")

# # Add both your local lib and the conda lib to R's search path
# .libPaths(c(lib_loc, conda_r_lib, .libPaths()))

# # Force the Linux compiler to look inside micromamba for the missing .so files
# Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))
# Sys.setenv(PROJ_LIB = paste0(conda_env, "/share/proj"))
# Sys.setenv(GDAL_DATA = paste0(conda_env, "/share/gdal"))

# # Provide the exact paths just in case it still tries to compile
# configure_args <- c(
#   sf = paste0("--with-gdal-config=", conda_env, "/bin/gdal-config --with-proj-include=", conda_env, "/include --with-proj-lib=", conda_env, "/lib"),
#   units = paste0("--with-udunits2-include=", conda_env, "/include --with-udunits2-lib=", conda_env, "/lib")
# )

# # 4. Smart Installer
# celesta_deps <- c("Rmixmod", "units", "sf", "spdep", "ggplot2", "reshape2", "zeallot", "FNN", "Matrix", "devtools")

# # Because we added conda_r_lib to .libPaths, R will now realize sf and spdep are already installed!
# missing_pkgs <- celesta_deps[!(celesta_deps %in% installed.packages()[,"Package"])]

# if(length(missing_pkgs) > 0) {
#   print(paste("Installing missing CELESTA dependencies:", paste(missing_pkgs, collapse=", ")))
#   install.packages(missing_pkgs, lib = lib_loc, repos = "http://cran.us.r-project.org", configure.args = configure_args)
# } else {
#   print("All CELESTA base dependencies already installed. Skipping...")
# }

# # 5. Install CELESTA from GitHub
# if (!"CELESTA" %in% installed.packages()[,"Package"]) {
#   print("Installing CELESTA from GitHub...")
#   devtools::install_github("plevritis/CELESTA", lib = lib_loc)
# }

# print("CELESTA installation complete! You can now run 'Rscript run_baselines.R'")
# 1. Set up custom HPC library path & Conda Paths
lib_loc <- "/hpc/home/vk93/R_libs"
conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
conda_r_lib <- paste0(conda_env, "/lib/R/library")

.libPaths(c(lib_loc, conda_r_lib, .libPaths()))

# CRITICAL: Force R to use Conda's newer C++ library (Fixes the GLIBCXX error)
Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

library(Seurat)    
library(future)
library(TACIT)
library(CELESTA)
library(readr)
library(mclust)
library(fossil)
library(dplyr)
options(future.globals.maxSize = 8000 * 1024^2)

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

print("Loading B004 Ground Truth Data...")
df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)
df_b004 <- df %>% filter(grepl("^B004", unique_region))

marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

true_labels <- df_b004$`Cell Type`
CELLxFEATURE <- df_b004[, marker_cols]

imaging_data <- df_b004[, c("x", "y", marker_cols)]
names(imaging_data)[1:2] <- c("X", "Y")

results <- list()
donors <- c("B008", "B012")

for (d in donors) {
  # ==========================================
  # TACIT
  # ==========================================
  print(sprintf("Running TACIT with %s Prior...", d))
  tacit_prior <- read_csv(sprintf("tacit_prior_%s.csv", d), show_col_types = FALSE)
  
  tacit_res <- TACIT(CELLxFEATURE, tacit_prior, r=10, p=10)
  pred_tacit <- tacit_res$data_clean_final$Cell_Type
  pred_tacit[is.na(pred_tacit)] <- "Unassigned"
  
  ari_tacit <- adjustedRandIndex(true_labels, pred_tacit)
  ri_tacit <- rand.index(as.numeric(as.factor(true_labels)), as.numeric(as.factor(pred_tacit)))
  f1_tacit <- calc_weighted_f1(true_labels, pred_tacit)
  
  results[[sprintf("TACIT_Prior_%s", d)]] <- c(ari_tacit, ri_tacit, f1_tacit)
  
  # ==========================================
  # CELESTA
  # ==========================================
  print(sprintf("Running CELESTA with %s Prior...", d))
  celesta_prior <- read_csv(sprintf("celesta_prior_%s.csv", d), show_col_types = FALSE)
  
  CelestaObj <- CreateCelestaObject(project_title = sprintf("B004_Eval_%s", d), celesta_prior, imaging_data)
  
  num_ct <- nrow(celesta_prior)
  CelestaObj <- AssignCells(CelestaObj, max_iteration=10, cell_change_threshold=0.01,
                            high_expression_threshold_anchor=rep(0.7, num_ct),
                            low_expression_threshold_anchor=rep(1, num_ct),
                            high_expression_threshold_index=rep(0.5, num_ct),
                            low_expression_threshold_index=rep(1, num_ct))
  
  pred_celesta <- CelestaObj@final_cell_type_assignment[, (CelestaObj@total_rounds + 1)]
  pred_celesta[pred_celesta == "Unknown" | is.na(pred_celesta)] <- "Unassigned"
  
  ari_celesta <- adjustedRandIndex(true_labels, pred_celesta)
  ri_celesta <- rand.index(as.numeric(as.factor(true_labels)), as.numeric(as.factor(pred_celesta)))
  f1_celesta <- calc_weighted_f1(true_labels, pred_celesta)
  
  results[[sprintf("CELESTA_Prior_%s", d)]] <- c(ari_celesta, ri_celesta, f1_celesta)
}

# ==========================================
# PRINT LEADERBOARD
# ==========================================
cat("\n======================================================\n")
cat("PRIOR-GUIDED BASELINES LEADERBOARD (Tested on B004)\n")
cat("======================================================\n")
for (name in names(results)) {
  metrics <- results[[name]]
  cat(sprintf("%-20s : ARI=%.4f | RI=%.4f | Weighted F1=%.4f\n", name, metrics[1], metrics[2], metrics[3]))
}
cat("======================================================\n")