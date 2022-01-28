# Ni_stereoconvergent_coupling
The 'gen_identifiers', 'gen_rxn_data', and/or 'gen_descriptors' excel files contain information on the kraken ligand library and the ligands employed in this study. These files should be kept in the same folder as the scripts.

gen_linear_regression.ipynb: code for univariate correlations and predictive workflow employing forward stepwise linear regression. Requires 'ForwardStepCandidates_updated.py' and 'loo_q2.py'.

gen_threshold_and_multidescriptor_plot.ipynb: code for the threshold analysis and visualization of stereoelectronic space.

gen_umap.ipynb: code for the uniform manifold and approximation (umap) dimensionality reduction and visualization of the chemical space.

gen_pca_clustering.ipynb: code for the principal component analysis (PCA) dimensionality reduction, visualization of the chemical space, and K-means clustering.

**Citation**

This code is released under the MIT license. Commercial use, Modification, Distribution and Private use are all permitted. The use of this workflow can be acknowledged with the following citation: https://pubs.acs.org/articlesonrequest/AOR-4BU5KPI6WVZXCZQIEV98

## Contributors
- ForwardStepCandidates_updated: code by Tobias Gensch with contributions from Cian Kingston. Code is a translation of a MATLAB script from Iris Guo
- gen_linear_regression: code by Tobias Gensch with contributions from Ellyn Peters, Jen Crawford, and Cian Kingston
- gen_pca_clustering: code by Tobias Gensch with contributions from Cian Kingston
- gen_threshold_and_multidescriptor_plot: code by Tobias Gensch with contributions from Cian Kingston
- gen_umap: code by Cian Kingston
