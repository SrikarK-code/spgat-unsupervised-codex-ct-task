# # Install required packages if missing

# lib_loc <- "/hpc/home/vk93/R_libs"
# if (!dir.exists(lib_loc)) dir.create(lib_loc, recursive = TRUE)
# .libPaths(c(lib_loc, .libPaths()))

# # install.packages(c("readr", "mclust", "fossil", "dplyr", "devtools"), 
# #                  lib = lib_loc, 
# #                  repos = "http://cran.us.r-project.org")
# devtools::install_github("plevritis/CELESTA", lib = lib_loc)

# --- SETUP ---
lib_loc <- "/hpc/home/vk93/R_libs"
conda_env <- "/hpc/group/yizhanglab/vk93/micromamba-envs/envs/codex-sthd"
conda_r_lib <- paste0(conda_env, "/lib/R/library")
.libPaths(c(lib_loc, conda_r_lib, .libPaths()))
Sys.setenv(LD_LIBRARY_PATH = paste0(conda_env, "/lib:", Sys.getenv("LD_LIBRARY_PATH")))

# --- LIBRARIES ---
library(CELESTA)
library(readr)
library(mclust)
library(fossil)
library(dplyr)
library(zeallot)  
library(Rmixmod)
library(spdep)

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

# FIX: Downgrade the imaging data to a base data.frame
imaging_data <- as.data.frame(df_b004[, c("x", "y", marker_cols)])
names(imaging_data)[1:2] <- c("X", "Y")

results <- list()
donors <- c("B008", "B012")

# --- RUN CELESTA ---
for (d in donors) {
  print(sprintf("Running CELESTA with %s Prior...", d))
  celesta_prior <- read_csv(sprintf("celesta_prior_%s.csv", d), show_col_types = FALSE)
  
  # FIX: Rename column AND downgrade to base data.frame
  celesta_prior <- celesta_prior %>% rename(cell_type = `Cell Type`)
  celesta_prior <- as.data.frame(celesta_prior)
  
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

cat("\n======================================================\n")
cat("CELESTA LEADERBOARD (Tested on B004)\n")
cat("======================================================\n")
for (name in names(results)) {
  metrics <- results[[name]]
  cat(sprintf("%-20s : ARI=%.4f | RI=%.4f | Weighted F1=%.4f\n", name, metrics[1], metrics[2], metrics[3]))
}