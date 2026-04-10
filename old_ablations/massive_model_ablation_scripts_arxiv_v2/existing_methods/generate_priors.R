# 1. Set up custom HPC library path
lib_loc <- "/hpc/home/vk93/R_libs"
.libPaths(c(lib_loc, .libPaths()))

library(readr)
library(dplyr)

print("Loading Data to generate priors...")
df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)

marker_cols <- c('MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161')

binarize_col <- function(x) {
  q75 <- quantile(x, 0.75, na.rm = TRUE)
  q25 <- quantile(x, 0.25, na.rm = TRUE)
  res <- rep(NA, length(x))
  res[x >= q75] <- 1
  res[x <= q25] <- 0
  return(res)
}

donors <- c("B008", "B012")

for (d in donors) {
  print(sprintf("Processing Donor %s...", d))
  
  # Safely filter using the unique_region prefix
  df_donor <- df %>% filter(grepl(paste0("^", d), unique_region))
  
  mean_expr <- df_donor %>%
    group_by(`Cell Type`) %>%
    summarise(across(all_of(marker_cols), mean, na.rm = TRUE))
  
  binary_matrix <- mean_expr
  binary_matrix[marker_cols] <- lapply(binary_matrix[marker_cols], binarize_col)
  
  # TACIT Prior
  tacit_name <- sprintf("tacit_prior_%s.csv", d)
  write_csv(binary_matrix, tacit_name)
  
  # CELESTA Prior
  celesta_matrix <- binary_matrix %>%
    mutate(Lineage = sprintf("1_0_%d", row_number()), .after = `Cell Type`)
  celesta_name <- sprintf("celesta_prior_%s.csv", d)
  write_csv(celesta_matrix, celesta_name)
}

print("Successfully generated all prior matrices!")