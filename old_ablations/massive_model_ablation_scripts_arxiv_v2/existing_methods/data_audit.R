# --- SETUP ---
lib_loc <- "/hpc/home/vk93/R_libs"
.libPaths(c(lib_loc, .libPaths()))
library(dplyr)
library(stringr)
library(readr)

# --- AUDIT SCRIPT ---
cat("\n=== DATA INTEGRITY AUDIT ===\n")
cat("1. Loading raw CSV...\n")
df <- read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', show_col_types = FALSE)

cat("2. Trimming invisible whitespace from regions...\n")
df <- df %>% mutate(unique_region = str_trim(unique_region))

cat("3. Isolating B004...\n")
df_b004 <- df %>% filter(str_detect(unique_region, "^B004"))
unique_regions <- unique(df_b004$unique_region)

cat(sprintf("   -> Found %d unique B004 regions.\n\n", length(unique_regions)))

cat("4. Testing Filter Integrity Loop:\n")
total_found <- 0
for (reg in unique_regions) {
  # This tests the exact string-matching logic we will use in the main script
  matched_cells <- df_b004 %>% filter(unique_region == reg) %>% nrow()
  
  if (matched_cells == 0) {
      cat(sprintf("   [FAIL] R cannot match region: '%s'\n", reg))
  } else {
      cat(sprintf("   [PASS] Region: '%-30s' | Cells: %d\n", reg, matched_cells))
      total_found <- total_found + matched_cells
  }
}

cat(sprintf("\n=== AUDIT COMPLETE ===\n"))
cat(sprintf("Total B004 Cells in memory: %d\n", nrow(df_b004)))
cat(sprintf("Total B004 Cells matched in loop: %d\n", total_found))
if (nrow(df_b004) == total_found) {
    cat("STATUS: PERFECT MATCH. Safe to proceed with pipeline.\n")
} else {
    cat("STATUS: CRITICAL LEAK. Do not proceed.\n")
}