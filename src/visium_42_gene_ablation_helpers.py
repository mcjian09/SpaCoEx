"""Matched-panel ablation; the registered SpaCoEx implementation is never edited."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


METHODS = ["SpaCoEx", "Moran's I", "HVG", "SPARK-X", "Random"]


def digest_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_study(home, run_dir, output_root, expected_candidates=200):
    home, run = Path(home).resolve(), Path(run_dir).resolve()
    loader = runpy.run_path(str(home / "spacoex_shared.py"))
    engine = loader["_engine"]()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {"dataset": "Visium Human Breast Cancer Block A Section 1",
                "preprocessing": "cp10k_log1p", "matrix": "logeuclidean",
                "selection": "gradient"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Expected {key}={value!r}, found {manifest.get(key)!r}.")
    current = {name: digest_file(Path(engine.__file__).with_name(name))
               for name in ("analysis.py", "frozen_math.py")}
    if current != manifest["method_sha256"]:
        raise ValueError("Saved run and current master differ. Refit this input with the current master first.")
    snapshot = next((p for p in (run.parent / "input.npz", run.parents[1] / "input.npz")
                     if p.is_file() and digest_file(p) == manifest["input_sha256"]), None)
    if snapshot is None:
        raise FileNotFoundError("Keep the saved request/input.npz with its runs folder.")
    x, xy, genes = engine.load_inputs(snapshot, manifest["preprocessing"])
    with np.load(snapshot, allow_pickle=False) as z:
        counts = np.array(z["X"], dtype=float)
        ids = pd.Index(z["expression_ids"].astype(str), name="barcode")
        totals = np.array(z["library_total"], dtype=float)
    if len(genes) != expected_candidates or len(set(genes)) != len(genes):
        raise ValueError(f"Expected {expected_candidates} distinct candidate genes.")
    if np.any(counts < 0) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("The master snapshot must contain raw integer counts.")
    spots = pd.read_csv(run / "spot_metadata.csv", index_col="barcode")
    if not spots.index.is_unique or set(spots.index) != set(ids):
        raise ValueError("Spot metadata and snapshot barcodes do not match.")
    spots = spots.loc[ids].copy()
    if spots.ground_truth.isna().any():
        raise ValueError("Missing fine-region annotations.")
    np.testing.assert_allclose(spots[["array_col", "array_row"]], xy)
    np.testing.assert_allclose(x, np.log1p(10000 * counts / totals[:, None]))
    with np.load(run / "association.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["genes"], genes)
        association = z["S"].copy()
    domains_path = home / "visium_internal_domain_grid.py"
    if not domains_path.is_file():
        raise FileNotFoundError(f"Existing downstream helper missing: {domains_path}")
    domains = runpy.run_path(str(domains_path))
    versions = {m: version(m) for m in ("numpy", "scipy", "pandas", "scikit-learn", "scanpy", "anndata")}
    signature = {"snapshot": digest_file(snapshot), "manifest": digest_file(run / "manifest.json"),
                 "spots": digest_file(run / "spot_metadata.csv"), "master": current,
                 "downstream_source": digest_file(domains_path),
                 "ablation_source": digest_file(__file__), "versions": versions}
    dest = Path(output_root).resolve() / ("study_" + digest_json(signature)[:12])
    dest.mkdir(parents=True, exist_ok=True)
    save_json(dest / "provenance.json", {**signature, "run_dir": str(run), "home": str(home),
              "candidate_policy": manifest["candidate_policy"],
              "scope": "Conditional gene selection within the same Moran-preselected 200 genes."})
    pd.DataFrame({"gene": genes}).to_csv(dest / "candidate_genes.csv", index=False)
    return dict(home=home, run=run, snapshot=snapshot, engine=engine, domains=domains,
                manifest=manifest, x=x, counts=counts, xy=xy, genes=genes, ids=ids,
                spots=spots, totals=totals, S=association, dest=dest, signature=signature)


def gradient_panel(study, size):
    run, genes = study["run"], study["genes"]
    w = pd.read_csv(run / "weights.csv", index_col="lambda", float_precision="round_trip")
    fits = pd.read_csv(run / "lambda_path.csv", index_col="lambda", float_precision="round_trip")
    if not w.index.equals(fits.index) or not w.columns.equals(pd.Index(genes)):
        raise ValueError("Saved weights and path are misaligned.")
    w = w.loc[fits.converged.eq(True)].sort_index()
    if w.empty or not np.isfinite(w.to_numpy()).all():
        raise ValueError("No finite converged gradient results.")
    active = w.to_numpy() > study["manifest"]["active_threshold"]
    last = np.where(active, w.index.to_numpy()[:, None], -np.inf).max(axis=0)
    table = pd.DataFrame({"gene": genes, "score": last})
    table["tied_rank"] = table.score.rank(method="min", ascending=False).astype(int)
    selected = table.loc[np.isfinite(last) & (table.tied_rank <= size)].sort_values(
        ["tied_rank", "gene"], ignore_index=True)
    if len(selected) != size:
        raise ValueError(f"Top {size} cuts a tied gradient group: keeping ties gives {len(selected)}. "
                         "Do not arbitrarily break this tie; use a complete group boundary.")
    # Alphabetical order within a tie is display order, not a stronger/weaker ranking.
    return selected, table


def moran_scores(x, pixels):
    tri = Delaunay(np.asarray(pixels, dtype=float))
    t = tri.simplices
    edges = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    W = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(x), len(x)))
    W = sparse.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W
    z = x - x.mean(axis=0)
    denominator = np.sum(z * z, axis=0)
    return np.divide(np.sum(z * (W @ z), axis=0), denominator,
                     out=np.zeros(x.shape[1]), where=denominator > 0)


def fixed_size_panel(table, size, ascending=False):
    if table.gene.duplicated().any() or not np.isfinite(table.score).all():
        raise ValueError("Ranking must have unique genes and finite scores.")
    ranked = table.sort_values(["score", "gene"], ascending=[ascending, True], ignore_index=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    boundary_tie = len(ranked) > size and ranked.score.iloc[size - 1] == ranked.score.iloc[size]
    return ranked.iloc[:size].copy(), ranked, bool(boundary_tie)


def python_panels(study, size=42, n_random=50, random_seed=1729):
    import anndata as ad
    import scanpy as sc
    if not 2 <= size < len(study["genes"]) or n_random < 1:
        raise ValueError("Need 2 <= panel size < candidate size, and at least one random panel.")
    genes, x = study["genes"], study["x"]
    panels, rankings, timing = {}, {}, []
    chosen, rankings["SpaCoEx"] = gradient_panel(study, size)
    panels["SpaCoEx"] = chosen.gene.tolist()
    timing.append(dict(method="SpaCoEx", selection_seconds=np.nan,
                       note="Saved master fit reused; original fitting time unavailable", cutoff_tie=False))
    start = time.perf_counter()
    scores = moran_scores(x, study["spots"][["pixel_col", "pixel_row"]])
    elapsed = time.perf_counter() - start
    recorded = pd.read_csv(study["run"] / "eligible_genes_moran_ranking.csv").set_index("gene")
    np.testing.assert_allclose(scores, recorded.loc[genes, "moran_I"], rtol=1e-5, atol=2e-6,
                               err_msg="Moran graph/preprocessing differs from the original screen")
    chosen, rankings["Moran's I"], tied = fixed_size_panel(pd.DataFrame({"gene": genes, "score": scores}), size)
    panels["Moran's I"] = chosen.gene.tolist()
    timing.append(dict(method="Moran's I", selection_seconds=elapsed,
                       note="Recomputed within shared candidate pool", cutoff_tie=tied))
    start = time.perf_counter()
    data = ad.AnnData(x.copy(), var=pd.DataFrame(index=pd.Index(genes)))
    data.uns["log1p"] = {"base": None}
    # Seurat dispersion HVGs expect log-normalized expression, not raw counts.
    sc.pp.highly_variable_genes(data, flavor="seurat", n_top_genes=size, n_bins=20)
    scores = data.var["dispersions_norm"].to_numpy(dtype=float)
    elapsed = time.perf_counter() - start
    chosen, rankings["HVG"], tied = fixed_size_panel(pd.DataFrame({"gene": genes, "score": scores}), size)
    panels["HVG"] = chosen.gene.tolist()
    timing.append(dict(method="HVG", selection_seconds=elapsed,
                       note="Scanpy Seurat normalized dispersion, 20 bins, shared 200 genes", cutoff_tie=tied))
    rng = np.random.default_rng(random_seed)
    used = set()
    for i in range(n_random):
        start = time.perf_counter()
        while True:
            idx = tuple(sorted(rng.choice(len(genes), size, replace=False).tolist()))
            if idx not in used:
                used.add(idx)
                break
        key = f"Random_{i+1:03d}"
        panels[key] = genes[list(idx)].tolist()
        timing.append(dict(method=key, selection_seconds=time.perf_counter() - start,
                           note="Uniform sample without replacement; independent overlapping panels allowed",
                           cutoff_tie=False))
    return panels, rankings, pd.DataFrame(timing)


def find_rscript(override=None):
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return str(p)
    if shutil.which("Rscript"):
        return shutil.which("Rscript")
    candidates = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError("Install R and set RSCRIPT to its Rscript.exe path in the controls cell.")


R_LIBRARY_SETUP = r'''
args <- commandArgs(trailingOnly=TRUE)
lib <- file.path(args[1], paste0("R-", getRversion()))
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
minor_version <- paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep=".")
user_home <- Sys.getenv("SPACOEX_R_USER_HOME")
existing_libraries <- c(
  file.path(user_home, "AppData", "Local", "SpaCoEx", "R-library", paste0("R-", minor_version)),
  file.path(user_home, "AppData", "Local", "R", "win-library", minor_version)
)
existing_libraries <- existing_libraries[dir.exists(existing_libraries)]
.libPaths(unique(c(lib, existing_libraries, .libPaths())))
'''


R_SETUP = R_LIBRARY_SETUP + r'''
cat(R.version.string, "\nLocal library:", lib, "\n")
if (!requireNamespace("SPARK", quietly=TRUE)) {
  if (args[2] != "TRUE") stop(paste(
    "SPARK is not installed for this R. Set INSTALL_R_PACKAGES=True in the notebook",
    "and rerun setup. Windows needs the Rtools version matching this R version."))
  options(repos=c(CRAN="https://cloud.r-project.org"), timeout=600)
  if (!requireNamespace("pkgbuild", quietly=TRUE)) install.packages("pkgbuild", lib=lib)
  if (!pkgbuild::has_build_tools(debug=TRUE)) stop(
    "Install matching Rtools from https://cran.r-project.org/bin/windows/Rtools/, restart the kernel, rerun setup.")
  if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", lib=lib)
  remotes::install_github("xzhoulab/SPARK", ref=args[3], lib=lib,
                          dependencies=NA, upgrade="never", build_vignettes=FALSE)
}
if (!exists("sparkx", asNamespace("SPARK"), inherits=FALSE)) stop("Installed SPARK does not provide sparkx().")
desc <- packageDescription("SPARK")
cat("SPARK:", as.character(packageVersion("SPARK")), "RemoteSha:", desc$RemoteSha, "\n")
cat("Loaded from:", find.package("SPARK"), "\n")
cat("Library search paths:\n"); print(.libPaths())
writeLines(capture.output(sessionInfo()), args[4])
'''


R_SPARKX = R_LIBRARY_SETUP + r'''
suppressPackageStartupMessages(library(SPARK))
suppressPackageStartupMessages(library(Matrix))
folder <- args[2]
counts <- as(readMM(file.path(folder, "counts_genes_by_spots.mtx")), "CsparseMatrix")
genes <- readLines(file.path(folder, "genes.txt"))
spots <- readLines(file.path(folder, "spots.txt"))
xy <- read.csv(file.path(folder, "coordinates.csv"), row.names=1, check.names=FALSE)
stopifnot(nrow(counts)==length(genes), ncol(counts)==length(spots), identical(rownames(xy), spots))
stopifnot(all(rowSums(counts)>0), all(colSums(counts)>0))
rownames(counts) <- genes
colnames(counts) <- spots
start <- proc.time()[3]
fit <- SPARK::sparkx(count_in=counts, locus_in=as.matrix(xy),
                     numCores=1, option="mixture", verbose=FALSE)
elapsed <- proc.time()[3] - start
tab <- fit$res_mtest
stopifnot(all(genes %in% rownames(tab)))
tab <- tab[genes, , drop=FALSE]
out <- data.frame(gene=genes, combinedPval=tab$combinedPval, adjustedPval=tab$adjustedPval)
stopifnot(all(is.finite(as.matrix(out[, -1]))), all(out$combinedPval>=0), all(out$combinedPval<=1))
write.csv(out, file.path(folder, "sparkx_scores.csv"), row.names=FALSE)
desc <- packageDescription("SPARK")
sha <- if (is.null(desc$RemoteSha)) "unavailable" else desc$RemoteSha
write.csv(data.frame(selection_seconds=as.numeric(elapsed),
                    SPARK_version=as.character(packageVersion("SPARK")), RemoteSha=sha,
                    R_version=R.version.string), file.path(folder, "sparkx_runtime.csv"), row.names=FALSE)
writeLines(capture.output(sessionInfo()), file.path(folder, "sparkx_sessionInfo.txt"))
'''


def run_r(source, path, rscript, arguments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    log = path.with_suffix(".log")
    env = os.environ.copy()
    env["SPACOEX_R_USER_HOME"] = str(Path.home())
    if os.name == "nt":
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
    with log.open("w", encoding="utf-8") as f:
        process = subprocess.Popen([rscript, "--vanilla", str(path), *map(str, arguments)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"R exited with code {code}. See {log}. No substitute SPARK-X scores were used.")


def setup_sparkx(output_root, r_library, rscript=None, install=False, ref="master"):
    exe = find_rscript(rscript)
    folder = Path(output_root).resolve() / "r_setup"
    folder.mkdir(parents=True, exist_ok=True)
    run_r(R_SETUP, folder / "setup_sparkx.R", exe,
          [Path(r_library).resolve(), "TRUE" if install else "FALSE", ref, folder / "sessionInfo.txt"])
    return exe


def sparkx_panel(study, size, r_library, rscript, force=False):
    folder = study["dest"] / "sparkx"
    folder.mkdir(exist_ok=True)
    signature = dict(input=study["signature"]["snapshot"], spots=study["signature"]["spots"],
                     source=hashlib.sha256(R_SPARKX.encode()).hexdigest(),
                     coords="pixel_col,pixel_row", counts="raw genes by spots", option="mixture")
    signature_path = folder / "input_signature.json"
    reuse = (not force and (folder / "sparkx_scores.csv").is_file()
             and signature_path.exists()
             and json.loads(signature_path.read_text()) == signature)
    if not reuse:
        # SPARK-X takes genes in rows, spots in columns. Do not log-transform its input.
        mmwrite(folder / "counts_genes_by_spots.mtx", sparse.coo_matrix(study["counts"].T))
        (folder / "genes.txt").write_text("\n".join(study["genes"]) + "\n", encoding="utf-8")
        (folder / "spots.txt").write_text("\n".join(study["ids"]) + "\n", encoding="utf-8")
        study["spots"][["pixel_col", "pixel_row"]].to_csv(folder / "coordinates.csv")
        run_r(R_SPARKX, folder / "run_sparkx.R", rscript, [Path(r_library).resolve(), folder])
        save_json(signature_path, signature)
    scores = pd.read_csv(folder / "sparkx_scores.csv")
    if set(scores.gene) != set(study["genes"]) or scores.gene.duplicated().any():
        raise ValueError("SPARK-X results do not cover exactly the candidate pool.")
    selected, ranked, tied = fixed_size_panel(scores.rename(columns={"combinedPval": "score"}),
                                              size, ascending=True)
    runtime = pd.read_csv(folder / "sparkx_runtime.csv").iloc[0]
    timing = dict(method="SPARK-X", selection_seconds=float(runtime.selection_seconds),
                  note="Official SPARK::sparkx, raw counts; cached original fitting time" if reuse else
                       "Official SPARK::sparkx, raw counts; timed R function only",
                  cutoff_tie=tied)
    print("SPARK-X version:", runtime.SPARK_version, "commit:", runtime.RemoteSha)
    print("SPARK-X selection is top-42 by combined P, not an FDR significance cutoff; adjustedPval is BY.")
    return selected.gene.tolist(), ranked, timing


def save_panels(study, panels, rankings, timing, size, n_random, random_seed):
    rows = []
    universe = set(study["genes"])
    for key, genes in panels.items():
        if len(genes) != size or len(set(genes)) != size or not set(genes) <= universe:
            raise ValueError(f"{key} is not a valid size-{size} panel from the shared pool.")
        for i, gene in enumerate(genes):
            rows.append(dict(panel=key, method="Random" if key.startswith("Random_") else key,
                             display_order=i+1, gene=gene))
    table = pd.DataFrame(rows)
    selection_id = digest_json({"panels": panels, "n_random": n_random, "random_seed": random_seed})[:12]
    dest = study["dest"] / ("selection_" + selection_id)
    dest.mkdir(exist_ok=True)
    table.to_csv(dest / "selected_genes.csv", index=False)
    timing.to_csv(dest / "selection_timing.csv", index=False)
    for key, ranking in rankings.items():
        name = key.replace("'", "").replace(" ", "_")
        ranking.to_csv(dest / f"{name}_scores.csv", index=False)
    selected = set(panels["SpaCoEx"])
    overlap = pd.DataFrame([dict(panel=key, overlap_with_SpaCoEx=len(selected & set(genes)),
                                jaccard_with_SpaCoEx=len(selected & set(genes)) / len(selected | set(genes)))
                            for key, genes in panels.items()])
    overlap.to_csv(dest / "gene_overlap.csv", index=False)
    save_json(dest / "selection_settings.json", dict(size=size, n_random=n_random, random_seed=random_seed,
              candidate_scope="same saved Moran-preselected 200 genes", tied_spacoex="include full tie group",
              other_score_ties="alphabetical deterministic tie-break; cutoff ties flagged",
              random="without replacement within each panel; overlap across methods/panels allowed"))
    return dest, table, overlap


def joint_features(study, genes, alpha, feature_seed=0):
    indices = pd.Index(study["genes"]).get_indexer(genes)
    if (indices < 0).any():
        raise ValueError("Panel contains a gene outside the common pool.")
    x = study["x"][:, indices]
    # These are the SAME functions used for the existing matched-panel breast maps.
    with threadpool_limits(limits=2):
        C = study["domains"]["_coexpression_pcs"](
            x, study["xy"], study["manifest"],
            study["engine"].math.transform_correlation_matrices, feature_seed)
        E = study["domains"]["_expression_pcs"](x, feature_seed)
        e, c = StandardScaler().fit_transform(E), StandardScaler().fit_transform(C)
        joined = np.column_stack((np.sqrt(1-alpha)*e, np.sqrt(alpha)*c))
        pcs = PCA(n_components=min(20, joined.shape[1]), random_state=feature_seed).fit_transform(joined)
    return pcs[:, :12]


def evaluate_panels(study, panels, selection_dir, *, alpha=0.5, cluster_seeds=(0, 1, 2, 3, 4),
                    n_clusters=20, feature_seed=0, force=False):
    if not 0 < alpha < 1:
        raise ValueError("This ablation freezes the joint pipeline: choose 0 < alpha < 1.")
    if not cluster_seeds or len(set(cluster_seeds)) != len(cluster_seeds):
        raise ValueError("Provide distinct clustering seeds.")
    settings = dict(alpha=float(alpha), n_clusters=int(n_clusters), n_init=50,
                    cluster_seeds=list(cluster_seeds), feature_seed=int(feature_seed),
                    expression="same helper: Scanpy scale max10 ->20 PCs",
                    coexpression="same helper: selected-panel correlations -> master log(+ridge I) -> weighted upper triangle -> standardize ->20 PCs",
                    joint="standardize PC blocks ->sqrt(1-alpha)/sqrt(alpha) ->20 PCs ->first12 ->KMeans",
                    PAS="10 nearest other spots in (array_row,array_col); >=6 mismatches",
                    bandwidth=study["manifest"]["bandwidth"], ridge=study["manifest"]["ridge"],
                    normalization="saved full-library CP10K then natural log1p; never renormalize on 42 genes",
                    annotations="fine ground_truth for evaluation only; 20 clusters fixed in advance",
                    scope="same 200-gene Moran-screened pool; descriptive, in-sample ablation")
    dest = Path(selection_dir) / ("downstream_" + digest_json(settings)[:12])
    dest.mkdir(exist_ok=True)
    save_json(dest / "settings.json", settings)
    truth = study["spots"].ground_truth.astype(str).to_numpy()
    if len(np.unique(truth)) != n_clusters:
        raise ValueError("Expected the original 20 fine annotated regions; inspect this dataset.")
    # Match the existing corrected PAS: kneighbors(X=None) already excludes self.
    neighbors = NearestNeighbors(n_neighbors=11).fit(
        study["spots"][["array_row", "array_col"]]).kneighbors(return_distance=False)[:, :10]
    all_rows = []
    for number, (key, genes) in enumerate(panels.items(), 1):
        method = "Random" if key.startswith("Random_") else key
        cache_key = digest_json(dict(genes=genes, signature=study["signature"], alpha=alpha,
                                    feature_seed=feature_seed))[:16]
        feature_folder = study["dest"] / "features"
        feature_folder.mkdir(exist_ok=True)
        cache = feature_folder / (cache_key + ".npz")
        feature_hit = cache.exists() and not force
        if feature_hit:
            with np.load(cache, allow_pickle=False) as z:
                features = z["features"]
                feature_seconds = float(z["feature_seconds"])
        else:
            start = time.perf_counter()
            features = joint_features(study, genes, alpha, feature_seed)
            feature_seconds = time.perf_counter() - start
            np.savez_compressed(cache, features=features, feature_seconds=feature_seconds)
        if features.shape != (len(truth), 12) or not np.isfinite(features).all():
            raise ValueError(f"Invalid downstream features for {key}.")
        panel_dir = dest / key.replace("'", "").replace(" ", "_")
        panel_dir.mkdir(exist_ok=True)
        metrics_file = panel_dir / "metrics.csv"
        reuse = metrics_file.exists() and not force
        if reuse:
            rows = pd.read_csv(metrics_file).to_dict("records")
        else:
            rows, label_table = [], pd.DataFrame(index=study["ids"])
            for seed in cluster_seeds:
                start = time.perf_counter()
                with threadpool_limits(limits=2):
                    labels = KMeans(n_clusters=n_clusters, n_init=50, random_state=seed).fit_predict(features)
                cluster_seconds = time.perf_counter() - start
                pas = np.mean((labels[neighbors] != labels[:, None]).sum(axis=1) >= 6)
                rows.append(dict(panel=key, method=method, seed=seed, n_genes=len(genes),
                                 ARI=adjusted_rand_score(truth, labels),
                                 NMI=normalized_mutual_info_score(truth, labels), PAS=pas,
                                 feature_seconds=feature_seconds, cluster_seconds=cluster_seconds))
                label_table[f"seed_{seed}"] = labels
            pd.DataFrame(rows).to_csv(metrics_file, index=False)
            label_table.to_csv(panel_dir / "clusters.csv.gz", compression="gzip")
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(dest / "metrics_by_panel_and_seed.csv", index=False)
        print(f"[{number}/{len(panels)}] {key}: ARI={np.mean([r['ARI'] for r in rows]):.3f}; "
              f"features {'cached' if feature_hit else 'computed'}, fits {'cached' if reuse else 'computed'}", flush=True)
    return pd.DataFrame(all_rows), dest


def summarize(metrics, dest):
    metrics = metrics.copy()
    summaries = []
    panel_means = metrics.groupby(["method", "panel"])[["ARI", "NMI", "PAS"]].mean().reset_index()
    panel_means.to_csv(Path(dest) / "metrics_panel_means.csv", index=False)
    for method in METHODS:
        d = metrics.loc[metrics.method.eq(method)]
        if d.empty:
            continue
        independent = panel_means.loc[panel_means.method.eq(method)] if method == "Random" else d
        row = dict(method=method, n_panels=d.panel.nunique(), seeds_per_panel=d.seed.nunique(),
                   SD_meaning="between random-panel means" if method == "Random" else "between clustering seeds")
        for metric in ("ARI", "NMI", "PAS"):
            row[metric + "_mean"] = independent[metric].mean()
            row[metric + "_SD"] = independent[metric].std(ddof=1) if len(independent) > 1 else 0.0
        summaries.append(row)
    table = pd.DataFrame(summaries)
    table.to_csv(Path(dest) / "summary.csv", index=False)
    return table, panel_means


def plot_metrics(metrics, dest):
    import matplotlib.pyplot as plt
    summary, panel_means = summarize(metrics, dest)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#777777"]
    for ax, metric in zip(axes, ("ARI", "NMI", "PAS")):
        for i, method in enumerate(METHODS):
            values = (panel_means.loc[panel_means.method.eq(method), metric] if method == "Random"
                      else metrics.loc[metrics.method.eq(method), metric]).to_numpy()
            if not len(values):
                continue
            jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(i + jitter, values, color=colors[i], s=23, alpha=.65)
            ax.plot([i-.2, i+.2], [values.mean()]*2, color="black", lw=2)
        ax.set_xticks(range(5), METHODS, rotation=30, ha="right")
        ax.set_ylabel(metric + (" (lower is better)" if metric == "PAS" else " (higher is better)"))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    for suffix in ("png", "pdf"):
        fig.savefig(Path(dest) / f"ablation_metrics.{suffix}", dpi=300, bbox_inches="tight")
    return fig, summary, panel_means


def plot_region_maps(study, dest, seed=0):
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment
    # Align cluster colors only for display, after unsupervised fitting.
    truth, names = pd.factorize(study["spots"].ground_truth.astype(str), sort=True)
    keys = ["SpaCoEx", "Moran's I", "HVG", "SPARK-X", "Random_001"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout="constrained")
    series = [("Manual annotation", truth)]
    for key in keys:
        path = Path(dest) / key.replace("'", "").replace(" ", "_") / "clusters.csv.gz"
        labels = pd.read_csv(path, index_col=0).loc[study["ids"], f"seed_{seed}"].to_numpy(int)
        counts = np.zeros((len(names), len(names)), int)
        np.add.at(counts, (labels, truth), 1)
        a, b = linear_sum_assignment(-counts)
        mapping = np.empty(len(names), int)
        mapping[a] = b
        series.append((key, mapping[labels]))
    pixels = study["spots"][["pixel_col", "pixel_row"]].to_numpy()
    for ax, (title, labels) in zip(axes.flat, series):
        ax.scatter(*pixels.T, c=labels, cmap=plt.get_cmap("tab20", len(names)),
                   vmin=-.5, vmax=len(names)-.5, s=7, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_axis_off()
    for suffix in ("png", "pdf"):
        fig.savefig(Path(dest) / f"ablation_regions_seed{seed}.{suffix}", dpi=300, bbox_inches="tight")
    return fig
