I will start with my CSV files, which already contain properly segmented cells—each row represents one cell, and baseline cell types are included (for example, Cell common for Melanoma or Cell Type for Intestine).

I will convert these CSVs into AnnData objects and make sure the spatial coordinates are properly loaded.

Next, I will use refscrna.py to calculate the mean intensity (μ) for every existing cell type, generating a profile for each type based on the spatial transcriptomics data.

Then, I will run patchify.py to partition the large cell graphs into smaller, memory-safe chunks using the KD-Tree method.

After that, I will train the model by running train.py. This step refines the spatial microenvironment labels by looking at each cell and its neighbors to smooth and adjust the neighborhood assignments.

Finally, I will visualize the results by running sthdviz.py, which will plot continuous (X, Y) scatter plots showing the refined spatial neighborhoods.

