# SpaCoEx: paper reproducibility package

Curated for the manuscript and response letter dated November 11, 2026.
This is a code-and-input release, not an export of the working directory.
Only the **25 um M3-2 MERFISH run** is supplied, alongside the Visium breast-cancer
analysis and the five-group simulation.

## What to put on GitHub

Upload the contents of `01_upload_first`, commit, then upload the contents of
`02_upload_second` to the same repository root and commit again. Do not upload
the two batch folders themselves. Both batches are required before running code.
Keep the relative folder structure. Do not add your other notebooks, Conda/R
installations, downloaded archives, reviewer letters, manuscripts, or caches.
`.gitignore` excludes newly generated outputs, local libraries and credentials.
No repository is created and nothing is uploaded by these scripts.

The completed repository has this layout:

```text
README.md
requirements.txt
.gitignore
notebooks/   Six entry-point notebooks, in run order
src/         Python/R implementation and the unchanged registered master
data/        Required Visium and M3-2 MERFISH inputs
reference/   Saved results for comparison with fresh runs
docs/        Data sources, environment, checksums, verification and schematics
```

To work locally, clone or download the completed repository after both uploads.
The upload batches are transport folders, not independent projects.

`data/` contains the public inputs needed to rerun preprocessing without large
downloads. `reference/` contains small, clearly labeled historical results for
comparison. `outputs/` is where every new analysis writes. The registered master
files are copied unchanged and their SHA-256 hashes are checked on import.
The raw 10x H5 is split into two binary transport parts to keep individual files
small. Notebook 02 rejoins and checks them automatically without changing any
counts. The reconstructed duplicate is excluded from Git.

## Setup

The exact recorded Python environment is in `docs/environment_recorded.json` and
`requirements.txt`. The analyses were run on Windows with Python 3.13.13.
Clustering can change with library/BLAS versions even when a seed is unchanged.
Use a separate environment, not your existing working environment:

```text
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

SpatialCorr 1.2.0 has an old, non-standard dependency specifier. It is deliberately
not in the requirements file. In the separate environment, install it without
resolving its old dependency constraints, then restore a current pip:

```text
python -m pip install "pip<24.1"
python -m pip install --no-deps spatialcorr==1.2.0
python -m pip install --upgrade pip
python -m pip check
```

The recorded setuptools version is below 81 because this SpatialCorr release
still imports `pkg_resources`. Keep that pin in the isolated environment.

The old SpatialCorr dependency metadata may still be reported by `pip check`.
Its import and native likelihood calculations are tested by `src/verify_release.py`.
The current machine's environment was tested; a completely clean network install
has not been performed for this release. Do not silently substitute a different
SpatialCorr version if the package index changes.

For SPARK-X, install R and its source-package toolchain (matching Rtools on Windows).
The original run used R 4.6.0, SPARK 1.1.1 at Git commit
`a8b4bf27b804604dfda53da42992f100b8e4e727`.
Notebook 03 installs that source only when `install_r=True` is explicitly chosen.
For the independent Conley check, install `conleyreg` (reference version 0.1.9)
and `units` in R. This release does not bundle anyone's R installation.

## Run order

Start Jupyter at the repository root and open the numbered files in `notebooks/`.
Their first setup cell locates `src/` automatically. Run cells from top to bottom:

1. `01_Simulation_original_figure.ipynb`: original five-group simulation,
   two penalties, three matrix representations, and the six-panel publication plot.
   The figure-producing default is 100 datasets of **2,000** observations.
   Controls remain at the end of its first code cell.
2. `02_Visium_prepare_and_select.ipynb`: rebuild the 200 candidates from raw counts
   and rerun the registered master, using the original result-producing bandwidth.
3. `03_Visium_clustering_and_gene_panels.ipynb`: recompute Moran/HVG/SPARK-X panels
   and cluster all panels using the same seed-0 expression/co-expression pipeline.
4. `04_Visium_pairwise_tests_and_Conley.ipynb`: actually perform 999 permutations
   for each pair in five 42-gene panels, apply BH within each panel, draw the
   heatmaps, and recompute the B2M/HLA-A spatial contrast and expression maps.
5. `05_MERFISH_M3_2.ipynb`: rebuild the fixed 1,200-cell boundary window and the
   136-gene candidate pool, refit at **25 um**, then cluster the top 23 genes.
6. `06_Visium_bandwidth_and_ridge.ipynb`: the response-letter staircase sweeps.
   Bandwidths 2, 3, 5, 7, 10; ridges 10^-10 through 10^-2.

The notebooks call `src/paper_reproduce.py`. They do not depend on your old workspace,
absolute local paths, another notebook's live kernel, or private helper locations.
If you skip a refit, downstream notebooks explicitly use the bundled reference
fit, not a newly computed fit. New successful fits are recorded in `outputs/*/latest.json`.
Do not run different sessions writing to the same output directory simultaneously.

`src/visium_42_gene_ablation_helpers.py` supports the comparison in Notebook 03:
SpaCoEx, Moran's I, HVG and SPARK-X select equally sized gene panels from the same
candidate pool, and the downstream clustering procedure is held fixed. It also
provides the shared Visium input loader and R/SPARK-X setup. "Ablation" is its
historical filename, not an additional dataset or a different SpaCoEx method.

Command-line alternatives:

```text
python src/verify_release.py
python src/paper_reproduce.py prepare-visium
python src/paper_reproduce.py fit-visium
python src/paper_reproduce.py panels --install-r
python src/paper_reproduce.py cluster-visium
python src/paper_reproduce.py br
python src/paper_reproduce.py conley
python src/paper_reproduce.py prepare-merfish
python src/paper_reproduce.py fit-merfish
python src/paper_reproduce.py cluster-merfish
python src/paper_reproduce.py sensitivity
```

For fast redraws only, without claiming new inference:

```text
python src/paper_reproduce.py br --reference
python src/paper_reproduce.py sensitivity --reference
```

The expensive steps are the permutation tests and 13 distinct sensitivity fits.
Do not reduce the permutation count for final inference. A small count is suitable
only for a smoke test and changes the attainable p-values.

## Figure/result map

| Paper item | Code | Inputs/reference |
|---|---|---|
| Figures 1-2, conceptual schematics | `docs/schematics/` | Not fitted data |
| Figure 3, simulation | Notebook 01 | Original simulator and plotting code |
| Figure 4 A-C, breast selection | Notebook 02 | `reference/visium/` |
| Figure 4 D, five differential-correlation panels | Notebook 04 | `reference/br/` |
| Figure 4 E, B2M/HLA-A and Conley results | Notebook 04 | `reference/conley/` |
| Figure 5 and Table 1 | Notebook 03 | `reference/visium_clustering/` |
| Figure 6, revised 25-um M3-2 analysis | Notebook 05 | `reference/merfish_25um/` |
| Response bandwidth/ridge staircases | Notebook 06 | `reference/sensitivity/` |

See `docs/DATA_SOURCES.md`, `docs/source_provenance.json`,
`docs/CHECKSUMS.sha256`, and `docs/VERIFICATION.json` for provenance and the
original checks actually performed. `docs/LAYOUT_VERIFICATION.json` records the
checks after folder reorganization. New checks write to `outputs/`.
No global-optimality claim is made for projected gradient ascent.

## Before public release

Choose a software license and add it to the repository.
No license has been invented or assigned to your work. Public input data remain
subject to their original providers' terms.

