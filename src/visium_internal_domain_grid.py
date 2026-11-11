"""Manual, SpaCoEx, expression and five joint Visium domain maps from one master run."""
import json
from pathlib import Path
from runpy import run_path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HOME = Path(__file__).resolve().parent
ALPHAS = (0.2, 0.4, 0.5, 0.6, 0.8)


def _coexpression_pcs(x, xy, manifest, transform, seed):
    # Recompute each spot's matrix on the selected panel, before taking the matrix log.
    p = x.shape[1]
    i, j = np.triu_indices(p)
    K = cdist(xy, xy, "sqeuclidean")
    K /= -float(manifest["bandwidth"])**2
    np.exp(K, out=K)
    K /= K.sum(axis=1, keepdims=True)
    d = x - K @ x
    sd = np.sqrt(np.maximum(K @ (d*d), 1e-12))
    f = np.empty((len(x), len(i)))
    for lo in range(0, len(i), 64):
        sl = slice(lo, lo + 64)
        f[:, sl] = (K @ (d[:, i[sl]]*d[:, j[sl]])) / (sd[:, i[sl]]*sd[:, j[sl]])
    del K
    for lo in range(0, len(x), 64):
        v = f[lo:lo+64]
        r = np.empty((len(v), p, p))
        r[:, i, j] = r[:, j, i] = np.clip(v, -1, 1)
        r[:, np.arange(p), np.arange(p)] = 1
        m = transform(r, manifest["matrix"], manifest["ridge"], manifest["fisher_clip"])
        v[:] = m[:, i, j]
    f[:, i != j] *= np.sqrt(2)
    return PCA(n_components=min(20, len(x)-1, len(i)), random_state=seed).fit_transform(
        StandardScaler().fit_transform(f))


def _expression_pcs(x, seed):
    from anndata import AnnData
    import scanpy as sc
    # Scale a copy; both representations start from the same unscaled panel.
    e = AnnData(X=x.copy())
    sc.pp.scale(e, max_value=10)
    sc.tl.pca(e, n_comps=min(20, min(x.shape)-1), random_state=seed)
    return np.asarray(e.obsm["X_pca"])


def domain_agreement(domain_labels, obs, coord_cols=("array_row", "array_col")):
    """Score saved spot labels without refitting. PAS is a fraction, not a percentage."""
    if not domain_labels.index.is_unique or not obs.index.is_unique or not domain_labels.index.isin(obs.index).all():
        raise ValueError("Each labeled spot must match one coordinate row by barcode.")
    if domain_labels.isna().any().any():
        raise ValueError("Missing domain labels.")
    xy = obs.loc[domain_labels.index, list(coord_cols)].to_numpy(dtype=float)
    # Match the old notebook's coordinate order. X=None already omits each query spot.
    neighbors = NearestNeighbors(n_neighbors=11).fit(xy).kneighbors(return_distance=False)
    truth = domain_labels["manual"].to_numpy()
    alphas = {"manual": np.nan, "coexpression": 1., "expression": 0.,
              **{f"joint_{a:.1f}": a for a in ALPHAS}}
    rows = []
    for key, column in domain_labels.items():
        labels = column.to_numpy()
        mismatch = labels[neighbors] != labels[:, None]
        pas = np.mean(mismatch[:, :10].sum(axis=1) >= 6)
        legacy = np.mean(mismatch[:, 1:].sum(axis=1) >= 6)
        rows.append((key, alphas[key], adjusted_rand_score(truth, labels),
                     normalized_mutual_info_score(truth, labels), pas, legacy))
    return pd.DataFrame(rows, columns=["panel", "coexpression_alpha", "ARI", "NMI",
                                       "PAS", "PAS_old_notebook"])


def domain_grid(adata, run_dir, *, top_n=42, output_dir=None, seed=0):
    """Compare expression and co-expression using the same top-ranked SpaCoEx genes."""
    if top_n < 2:
        raise ValueError("Co-expression requires at least two selected genes.")
    run_dir = Path(run_dir)
    dest = Path(output_dir) if output_dir else run_dir / f"internal_domains_matched_top{top_n}"
    manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != "Visium Human Breast Cancer Block A Section 1" or manifest["preprocessing"] != "cp10k_log1p":
        raise ValueError("Expected the normalized Block A Section 1 master run.")
    engine = run_path(str(HOME/"spacoex_shared.py"))["_engine"]()
    snap = next((p for p in (run_dir.parent/"input.npz", run_dir.parents[1]/"input.npz")
                 if p.is_file() and engine.sha256(p) == manifest["input_sha256"]), None)
    if snap is None:
        raise FileNotFoundError("Cannot locate the saved run's matching input.npz.")
    x, xy, genes = engine.load_inputs(snap, manifest["preprocessing"])
    with np.load(snap, allow_pickle=False) as s:
        ids = pd.Index(s["expression_ids"].astype(str), name="barcode")
    if not adata.obs_names.is_unique or set(ids) != set(adata.obs_names):
        raise ValueError("The input and saved SpaCoEx run must cover the same barcodes.")
    adata = adata[ids]
    np.testing.assert_allclose(xy, adata.obs[["array_col", "array_row"]])
    choose = run_path(str(HOME/"visium_spatialcorr_domains.py"))["choose_panel"]
    panel, _ = choose(run_dir, manifest, genes, top_n)
    if len(panel) != top_n:
        raise ValueError(f"Rank cutoff splits a tie; use top_n={len(panel)} to keep the complete group.")
    idx = pd.Index(genes).get_indexer(panel.gene)
    if np.any(idx < 0) or not panel.gene.is_unique:
        raise ValueError("The selected panel must contain unique genes from the saved run.")
    x = x[:, idx]
    current = adata[:, panel.gene.tolist()].X
    np.testing.assert_allclose(current.toarray() if sparse.issparse(current) else current,
                               x, rtol=1e-5, atol=1e-6, err_msg="Panel expression scales differ.")
    annot = adata.obs["ground_truth"]
    return matched_domain_grid(x, xy, manifest, engine.math.transform_correlation_matrices,
                               panel, annot, dest, seed=seed)


def matched_domain_grid(x, xy, manifest, transform, panel, annot, dest, *, seed=0, show_key=False):
    """Fit the breast benchmark's seven representations on one labeled, matched panel."""
    dest = Path(dest)
    ids = annot.index
    if not ids.is_unique or x.shape != (len(ids), len(panel)) or xy.shape != (len(ids), 2):
        raise ValueError("Spot IDs, panel, expression and coordinates must align.")
    if annot.isna().any():
        raise ValueError("Missing manual annotation.")
    truth, regions = pd.factorize(annot.astype(str), sort=True)
    k = len(regions)
    results = pd.DataFrame({"manual": np.asarray(regions)[truth]}, index=ids)
    names = ["Manual annotation", "SpaCoEx co-expression", "Expression only"]
    keys = ["manual", "coexpression", "expression"]
    codes = [truth]
    def add_fit(key, PCs, dim):
        raw = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(PCs[:, :dim])
        results[key] = raw
        # Cluster IDs are arbitrary. Match their map colors only AFTER the fit.
        counts = np.zeros((k, k), dtype=int)
        np.add.at(counts, (raw, truth), 1)
        src, target = linear_sum_assignment(-counts)
        color_index = np.empty(k, dtype=int)
        color_index[src] = target
        codes.append(color_index[raw])

    with threadpool_limits(limits=2):
        C = _coexpression_pcs(x, xy, manifest, transform, seed)
        E = _expression_pcs(x, seed)
        # Retain the original pure-map conventions: all co-expression PCs, first 12 expression PCs.
        add_fit(keys[1], C, C.shape[1])
        add_fit(keys[2], E, 12)
        e, c = StandardScaler().fit_transform(E), StandardScaler().fit_transform(C)
        for alpha in ALPHAS:
            joint = np.column_stack((np.sqrt(1-alpha)*e, np.sqrt(alpha)*c))
            scores = PCA(n_components=min(20, joint.shape[1]), random_state=seed).fit_transform(joint)
            key = f"joint_{alpha:.1f}"
            add_fit(key, scores, 12)
            names.append(rf"Joint $\alpha={alpha:.1f}$")
            keys.append(key)

    metrics = domain_agreement(results, pd.DataFrame(xy, columns=["x", "y"], index=ids), ("y", "x"))
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.7), layout="constrained")
    colors = plt.get_cmap("tab20", k)
    panels = dict(zip(keys, zip(codes, names)))
    plot_order = ["manual", "coexpression", *(f"joint_{a:.1f}" for a in ALPHAS), "expression"]
    for ax, key in zip(axes.flat, plot_order):
        labels, title = panels[key]
        ax.scatter(*xy.T, c=labels, cmap=colors, vmin=-0.5, vmax=k-0.5,
                   marker="o", s=6, linewidths=0)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_axis_off()
    if show_key:
        from matplotlib.lines import Line2D
        axes.flat[0].legend(handles=[Line2D([], [], marker="o", ls="", color=colors(i),
                                            label=region, markersize=5)
                                     for i, region in enumerate(regions)],
                            loc="upper center", bbox_to_anchor=(.5, 0), ncol=3,
                            fontsize=8, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"spatial_domains_2x4.{ext}", dpi=300, bbox_inches="tight")
    results.to_csv(dest/"spot_clusters.csv")
    metrics.to_csv(dest/"agreement.csv", index=False)
    pd.DataFrame(dict(spot_color_index=np.arange(k), manual_region=regions)).to_csv(
        dest/"manual_color_key.csv", index=False)
    pd.DataFrame(dict(zip(keys,codes)),index=ids).to_csv(dest/"spot_color_indices.csv")
    panel.to_csv(dest/"spacoex_panel.csv", index=False)
    return fig, metrics, results
