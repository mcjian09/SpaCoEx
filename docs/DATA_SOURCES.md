# Public data and source attribution

## Human Breast Cancer Block A Section 1

- Dataset: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0
- Raw filtered counts: https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5
- Position archive: same prefix ending `_spatial.tar.gz`; only the tissue-position
  CSV is bundled. Histology images are not needed by this analysis.
- SEDR annotation source: https://github.com/JinmiaoChenLab/SEDR_analyses/blob/master/data/BRCA1/metadata.tsv
- 3,798 fine-annotated spots. Detection in >=3 annotated spots defines the library
  denominator; candidate genes also require >=50% detection and named artifact
  exclusions before selecting the top 200 by pixel-Delaunay Moran's I.
- Strict cancer = IDC or DCIS/LCIS prefix. Tumor-edge and Healthy are non-cancer.

## PEtracer MERFISH M3-2

- Dataset: https://figshare.com/articles/dataset/PEtracer_Data/28473866
- Original M3 file: https://ndownloader.figshare.com/files/55130807
- Original file bytes: 910462070; MD5: 48977f5c3de98b63876d35b13a892618.
- Source preprocessing: https://github.com/jweissmanlab/PEtracer-2025/blob/main/tumor_tracing/M3_process_MERFISH.ipynb
- Source boundary: https://github.com/jweissmanlab/PEtracer-2025/blob/main/tumor_tracing/M3_segment_tumors.ipynb
- The ZIP includes the extracted measured-count M3-2 section, so selecting the
  fixed 1,200-cell window can be repeated offline without the full 910 MB file.
  `prepare_merfish()` reconstructs and checks the same window/candidate pool.
- The public tumor mask derives from malignant-cell positions, not independent
  pathologist annotations. Only the 25-um fitted analysis is included.

## Third-party algorithms

- SpatialCorr 1.2.0: https://github.com/mbernste/spatialcorr
  Installed separately. `pairwise_br.py` implements a two-dimensional algebraic
  expansion and checks native observed/permuted likelihoods; it is not represented
  as unmodified SpatialCorr source.
- SPARK-X: https://github.com/xzhoulab/SPARK ; the exact commit is in README.md.
  Uses raw genes-by-spots counts, image-pixel coordinates and option='mixture'.
- HVG: Scanpy's Seurat dispersion method, 20 mean-expression bins, on CP10K/log1p
  data; it ranks within the shared candidate pool.
- Conley validation: R conleyreg 0.1.9, radial Bartlett kernel, pixel distances,
  main cutoff 2000 pixels. Code and reference numeric results are included.

Data attribution is not a new license grant. Retain the original providers'
terms and citations when distributing or reusing these inputs.

