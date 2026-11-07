# Chemical Space and Clustering

A workflow for mapping, visualizing, and clustering molecular chemical space from numerical molecular descriptors.

The repository provides a shared notebook and reusable Python functions for dimensionality reduction, chemical-space visualization, clustering, parameter screening, and export of coordinates and cluster assignments.

## Repository structure

```text
.
├── data/
├── example_output_coordinates/
├── example_plot_outputs/
├── LICENSE
├── README.md
├── chemical_space_and_clustering.ipynb
├── chemical_space_and_clustering.yml
├── chemical_space_and_clustering_windows.yml
└── clustering_functions.py
```

- `chemical_space_and_clustering.ipynb` contains the interactive workflow.
- `clustering_functions.py` contains the reusable plotting, dimensionality-reduction, and clustering functions.
- `data/` contains example input datasets.
- `example_output_coordinates/` and `example_plot_outputs/` contain example exported results.
- Separate conda environment files are provided for the standard and Windows installations.

## Installation

Create the conda environment using the supplied environment file:

```bash
conda env create -f chemical_space_and_clustering.yml -n chem_space_and_clustering
conda activate chem_space_and_clustering
```

For Windows, use:

```bash
conda env create -f chemical_space_and_clustering_windows.yml -n chem_space_and_clustering
conda activate chem_space_and_clustering
```

Using the supplied environment files is recommended because UMAP and its dependencies can be sensitive to package versions.

## Quick start

1. Create and activate the conda environment.
2. Open `chemical_space_and_clustering.ipynb`.
3. In the **Read in Data** section, set the input workbook, sheet name, identifier column, optional SMILES/type columns, and the column at which numerical descriptors begin.
4. Run the preprocessing cells.
5. Select the dimensionality-reduction and clustering methods required for the analysis.
6. Enable Excel or PNG export in the corresponding cells when needed.

The repository includes `data/sample_input.xlsx` and `data/digits_dataset_sample.xlsx` as example inputs.

## Input data

The main workflow reads an Excel workbook containing compound identifiers and numerical molecular descriptors.

The notebook configuration includes:

```python
feature_input_file = './data/sample_input'
feature_sheet_name = 'Sheet1'
feature_start_column = 5
feature_id_column_name = 'Compound_Name'
feature_smiles_column_name = 'smiles'
feature_type_column_name = 'type'
```

`feature_start_column` defines the first descriptor column using zero-based indexing. Columns from that position onward are treated as numerical features.

SMILES are optional for plotting. A categorical type/subgroup column is also optional.

An additional response variable can be merged for response-colored visualizations when required.

Rows with missing values in required columns are removed before analysis.

## Preprocessing

Numerical descriptor features are standardized before dimensionality reduction.

Because dimensionality reduction and clustering are sensitive to preprocessing, use the same descriptor definitions, scaling procedure, and random-state settings when comparing analyses.

## Dimensionality reduction

The workflow supports:

- UMAP
- t-SNE
- PCA

### UMAP

UMAP coordinates can be generated and exported to Excel. The notebook exposes parameters including:

- `n_neighbors`
- `min_dist`
- distance metric
- random state

An optional UMAP parameter-screening function compares combinations of neighborhood size and minimum distance before selecting final settings.

### t-SNE

t-SNE coordinates can be generated with configurable perplexity, initialization, learning rate, distance metric, and random state.

### PCA

PCA coordinates can be generated and exported together with feature loadings/contributions.

## Chemical-space visualization

Static chemical-space plots can be customized using:

- subgroup coloring;
- continuous response heatmaps;
- marker size;
- marker type;
- marker transparency;
- optional 2D or 3D coordinates.

Interactive Dash scatter plots are also available and can display compound identifiers and molecular structures when SMILES are provided.

## Clustering

The workflow includes the following clustering methods:

- K-means
- Gaussian Mixture Models (GMM)
- HDBSCAN
- DBSCAN
- Agglomerative clustering
- BIRCH

For GMM, the exported results include the probability of membership in each Gaussian component.

The notebook also includes tools to help assess or select clustering parameters, including:

- Hopkins statistic;
- silhouette scores;
- AIC/BIC/inertia-based comparisons;
- elbow analysis;
- optional HDBSCAN parameter screening.

## Outputs

Depending on the selected workflow and export settings, outputs may include:

- UMAP, t-SNE, or PCA coordinates;
- PCA feature loadings/contributions;
- cluster assignments;
- GMM component probabilities;
- parameter-screening figures;
- static chemical-space plots;
- clustering plots;
- interactive chemical-space visualizations.

Example coordinate exports and figures are included in the repository.

## Reproducibility

Dimensionality-reduction and clustering results may depend on:

- package versions;
- descriptor definitions;
- preprocessing and scaling;
- random-state settings;
- dimensionality-reduction hyperparameters;
- clustering hyperparameters.

Use the supplied conda environment files and keep random states fixed when reproducibility is required.

UMAP in particular can be sensitive to version changes in NumPy, Numba, UMAP, and related dependencies. If compatibility problems occur, first reproduce the supplied environment rather than changing individual packages independently.

## License

The code in this repository is licensed under the MIT License. You are free to use, modify, and distribute the code under the terms of this license.

Copyright (c) 2026 Sigman Lab, University of Utah


## Citation and contact

If you use or adapt this workflow, please cite the repository and the associated publication when available.

For questions, bug reports, or reproducibility issues, please use the repository issue tracker.
