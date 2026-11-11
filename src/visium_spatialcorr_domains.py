"""Cluster Visium spots using a saved SpaCoEx panel and SpatialCorr correlations."""
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def resolve_run(run, workdir):
    if run is not None:
        folder = Path(run).expanduser().resolve()
        if (folder / "results" / "manifest.json").is_file():
            folder = folder / "results"
        if not (folder / "manifest.json").is_file():
            raise FileNotFoundError(f"No manifest.json in {folder}. Select a run or extracted results ZIP.")
        return folder
    matches = sorted(Path(workdir).glob("request_*/runs/*/manifest.json"))
    if len(matches) != 1:
        raise ValueError(f"Set RUN_DIR to one specific run. Found {len(matches)}: "
                         + ", ".join(str(f.parent) for f in matches))
    return matches[0].parent.resolve()


def choose_panel(run, manifest, genes, top_x, ordered_genes=None):
    if not isinstance(top_x, (int, np.integer)) or not 2 <= top_x <= len(genes):
        raise ValueError(f"TOP_X must be an integer from 2 to {len(genes)}.")
    if ordered_genes is not None:
        if isinstance(ordered_genes, str):
            raise ValueError("Pass ORDERED_GENES as a list of names, not a string.")
        order = pd.Index(ordered_genes)
        if not order.is_unique or not order.isin(genes).all() or len(order) < top_x:
            raise ValueError("ORDERED_GENES needs at least TOP_X unique, in-panel gene names.")
        return pd.DataFrame({"gene": order[:top_x], "rank": np.arange(1, top_x + 1)}), "user-supplied order"
    if manifest["selection"] == "backward":
        ranked = pd.read_csv(run / "backward_ranking.csv").sort_values("rank_last_removed_first")
        if set(ranked.gene) != set(genes) or len(ranked) != len(genes):
            raise ValueError("Backward ranking does not match the input genes.")
        return ranked.head(top_x).copy(), "backward, last removed first"
    if manifest["selection"] != "gradient":
        raise ValueError("Unknown SpaCoEx selection method.")
    weights = pd.read_csv(run / "weights.csv", index_col="lambda", float_precision="round_trip")
    fits = pd.read_csv(run / "lambda_path.csv", index_col="lambda", float_precision="round_trip")
    if (not fits.index.equals(weights.index) or not fits.index.is_unique
            or not weights.columns.equals(pd.Index(genes))):
        raise ValueError("Gradient fit and weight tables do not match the saved panel/grid.")
    w = weights.loc[fits.converged.eq(True)].sort_index()
    if w.empty or not np.isfinite(w.to_numpy()).all():
        raise ValueError("No usable converged gradient path.")
    active = w.to_numpy() > manifest["active_threshold"]
    last = np.where(active, w.index.to_numpy()[:, None], -np.inf).max(axis=0)
    table = pd.DataFrame({"gene": genes, "last_active_lambda": last,
                          "active_at_last_fit": active[-1]})
    table = table.loc[np.isfinite(last)].copy()
    if len(table) < top_x:
        raise ValueError(f"Only {len(table)} genes were active on the converged path.")
    table["rank"] = table.last_active_lambda.rank(ascending=False, method="min").astype(int)
    return (table.loc[table["rank"] <= top_x].sort_values(["rank", "gene"]).reset_index(drop=True),
            "largest tested converged lambda still active; include all cutoff ties")


def prepare_profiles(run, load_inputs, top_x, ordered_genes=None, contrib_thresh=10):
    from anndata import AnnData
    from spatialcorr import compute_kernel_matrix
    from spatialcorr.utils import compute_local_correlation

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != "Visium Human Breast Cancer Block A Section 1":
        raise ValueError("Choose a Block A Section 1 master run.")
    snap = next((p for p in (run.parent / "input.npz", run.parents[1] / "input.npz")
                 if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == manifest["input_sha256"]), None)
    if snap is None:
        raise ValueError("Missing input.npz matching the saved run. Keep the extracted archive together.")
    X, xy, genes = load_inputs(snap, manifest["preprocessing"])
    with np.load(snap, allow_pickle=False) as z:
        ids = pd.Index(z["expression_ids"].astype(str), name="barcode")
    spots = pd.read_csv(run / "spot_metadata.csv", index_col="barcode")
    if not spots.index.is_unique or set(spots.index) != set(ids):
        raise ValueError("Spot metadata barcodes do not match the saved expression matrix.")
    spots = spots.loc[ids, ["array_col", "array_row", "pixel_col", "pixel_row"]].copy()
    np.testing.assert_allclose(spots[["array_col", "array_row"]], xy)
    panel, rule = choose_panel(run, manifest, genes, top_x, ordered_genes)
    X = np.ascontiguousarray(X[:, pd.Index(genes).get_indexer(panel.gene)])
    h = manifest["bandwidth"]
    if not np.isfinite(contrib_thresh) or contrib_thresh < 0:
        raise ValueError("CONTRIB_THRESH must be finite and nonnegative.")
    K = compute_kernel_matrix(spots, bandwidth=h, x_col="array_col", y_col="array_row",
                              condition_on_region=False)
    mass = K.sum(axis=1)
    keep = np.flatnonzero(mass >= contrib_thresh)
    if len(keep) < 3:
        raise ValueError("Fewer than three spots pass SpatialCorr's kernel-weight filter.")

    # Verify the batched estimator against the installed package on the first actual pair.
    pair_ad = AnnData(X=X[:, :2], obs=pd.DataFrame(index=ids),
                      var=pd.DataFrame(index=panel.gene.iloc[:2].to_list()))
    with np.errstate(divide="ignore", invalid="ignore"):
        ref, ref_idx = compute_local_correlation(pair_ad, *panel.gene.iloc[:2],
                                                 kernel_matrix=K, contrib_thresh=contrib_thresh)
    np.testing.assert_array_equal(ref_idx, keep)
    if len(keep) != len(X):
        K, X = K[np.ix_(keep, keep)], X[keep]
    K /= K.sum(axis=1, keepdims=True)
    # SpatialCorr centers each source spot on its own local mean before smoothing products.
    # Batch the unique pairs rather than materializing spots x genes x genes covariance tensors.
    ii, jj = np.triu_indices(X.shape[1], 1)
    with threadpool_limits(limits=4, user_api="blas"):
        D = X - K @ X
        sd = np.sqrt(K @ (D * D))
        features = np.empty((len(X), len(ii)), dtype=np.float32)
        for a in range(0, len(ii), 64):
            b = slice(a, a + 64)
            cov = K @ (D[:, ii[b]] * D[:, jj[b]])
            denom = sd[:, ii[b]] * sd[:, jj[b]]
            corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
            features[:, b] = np.clip(corr, -1, 1)
    np.testing.assert_allclose(features[:, 0], ref, rtol=1e-5, atol=1e-7, equal_nan=True)
    # Never impute undefined correlations as zero or assign filtered spots a tissue label.
    finite = np.isfinite(features).all(axis=0)
    variable = np.zeros_like(finite)
    variable[finite] = np.ptp(features[:, finite], axis=0) > 1e-7
    if not variable.any():
        raise ValueError("No pairs have a finite, varying local correlation profile.")
    names = panel.gene.to_numpy()
    pairs = pd.DataFrame({"gene_a": names[ii], "gene_b": names[jj], "used": variable,
                          "status": np.select([~finite, ~variable], ["undefined", "constant"], default="used")})
    spots["kernel_weight_sum"] = mass
    info = dict(run=str(run), input_sha256=manifest["input_sha256"], preprocessing=manifest["preprocessing"],
                selection=manifest["selection"], selection_matrix=manifest["matrix"], ranking=rule,
                top_x=top_x, actual_genes=len(panel), bandwidth=h, contrib_thresh=contrib_thresh,
                spatialcorr_version=version("spatialcorr"))
    return dict(panel=panel, pairs=pairs, spots=spots, keep=keep, features=features[:, variable], info=info)


def cluster_profiles(data, n_clusters, n_pcs, seed, output_root):
    feats = data["features"]
    if not isinstance(n_clusters, int) or not 2 <= n_clusters < len(feats):
        raise ValueError("N_CLUSTERS must be an integer from 2 to the number of retained spots minus 1.")
    if not isinstance(n_pcs, int) or n_pcs < 1:
        raise ValueError("N_PCS must be a positive integer.")
    components = min(n_pcs, len(feats) - 1, feats.shape[1])
    with threadpool_limits(limits=4):
        Z = StandardScaler().fit_transform(feats)
        pcs = PCA(n_components=components, svd_solver="randomized", random_state=seed).fit_transform(Z)
        labels = KMeans(n_clusters=n_clusters, n_init=50, random_state=seed).fit_predict(pcs) + 1
    if len(np.unique(labels)) != n_clusters:
        raise ValueError("Fewer distinct clusters than requested. Reduce N_CLUSTERS.")
    spots = data["spots"].copy()
    spots["cluster"] = pd.array([pd.NA] * len(spots), dtype="Int64")
    spots.loc[spots.index.take(data["keep"]), "cluster"] = labels
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    for group, block in spots.groupby("cluster", dropna=False, sort=True):
        missing = pd.isna(group)
        color = "0.8" if missing else (plt.get_cmap("tab20")((int(group)-1) % 20)
                if n_clusters <= 20 else plt.get_cmap("turbo")((int(group)-1)/(n_clusters-1)))
        ax.scatter(block.pixel_col, block.pixel_row, s=9, linewidths=0, color=color,
                   label="Filtered" if missing else str(group))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(title="Spot cluster", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    info = dict(data["info"], n_clusters=n_clusters, n_pcs=components, seed=seed,
                kmeans_n_init=50, n_clustered=len(labels), n_filtered=len(spots)-len(labels),
                n_pairs_used=feats.shape[1], n_pairs_total=len(data["pairs"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = Path(output_root) / f"{stamp}_genes{info['actual_genes']}_clusters{n_clusters}"
    dest.mkdir(exist_ok=False, parents=True)
    data["panel"].to_csv(dest / "selected_genes.csv", index=False)
    data["pairs"].to_csv(dest / "gene_pairs.csv", index=False)
    spots.to_csv(dest / "spot_clusters.csv")
    (dest / "settings.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    fig.savefig(dest / "tissue_regions.png", dpi=240, bbox_inches="tight")
    return spots, fig, dest
