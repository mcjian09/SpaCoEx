"""SpatialCorr spotwise correlation and Conley inference for strict tumor annotation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import spatialcorr


def hac_table(values, cancer, pixel_xy, cutoffs=(750, 1000, 1500, 2000, 2500, 3000)):
    """Intercept + binary contrast, radial Bartlett HAC, HC1, normal Wald inference."""
    z = np.asarray(cancer, dtype=bool)
    y = values.to_numpy(dtype=float)
    xy = np.asarray(pixel_xy, dtype=float)
    if len(xy) != len(y) or len(z) != len(y) or not np.isfinite(xy).all() or not np.isfinite(y).all():
        raise ValueError("Outcomes, groups and pixel coordinates must align and be finite.")
    n1, n0 = int(z.sum()), int((~z).sum())
    if min(n1, n0) == 0 or len(z) <= 2:
        raise ValueError("Both groups and positive residual degrees of freedom are required.")
    m1, m0 = y[z].mean(axis=0), y[~z].mean(axis=0)
    resid = y - np.where(z[:, None], m1, m0)
    # Coefficient influence for the difference in group means (OLS slope).
    s = resid * np.where(z, 1/n1, -1/n0)[:, None]
    tree = cKDTree(xy)
    a, b = tree.query_pairs(max(cutoffs), output_type="ndarray").T
    dist = np.linalg.norm(xy[a] - xy[b], axis=1)
    cross = s[a] * s[b]
    rows = []
    for cutoff in cutoffs:
        weights = np.maximum(1 - dist / cutoff, 0)
        variance = ((s*s).sum(axis=0) + 2*(weights @ cross)) * len(z)/(len(z)-2)
        if np.any(variance <= 0):
            raise ValueError(f"Nonpositive HAC variance at {cutoff} pixels.")
        se = np.sqrt(variance)
        delta = m1 - m0
        rows.append(pd.DataFrame(dict(outcome=values.columns, cutoff_px=cutoff,
                    cancer_n=n1, noncancer_n=n0, cancer_mean=m1, noncancer_mean=m0,
                    difference=delta, spatial_SE=se, lower95=delta-1.96*se,
                    upper95=delta+1.96*se, P=2*norm.sf(abs(delta/se)))))
    return pd.concat(rows, ignore_index=True), resid


def run_pair(adata, breast_h, out, gene_a="B2M", gene_b="HLA-A", contrib_thresh=10):
    """Pass the already CP10K/log1p AnnData; no second normalization or SpaCoEx fit."""
    obs = adata.obs
    genes = [gene_a, gene_b]
    if not adata.var_names.is_unique or not adata.obs_names.is_unique or not set(genes) <= set(adata.var_names):
        raise ValueError("Check gene names and uniqueness of gene/spot IDs.")
    family = obs["ground_truth"].astype(str).str.replace(r"_\d+$", "", regex=True)
    if not set(family) <= {"IDC", "DCIS/LCIS", "Tumor_edge", "Healthy"}:
        raise ValueError("Unrecognized manual annotation.")
    z = family.isin(["IDC", "DCIS/LCIS"]).to_numpy()
    if {"pixel_col", "pixel_row"} <= set(obs):
        xy = obs[["pixel_col", "pixel_row"]].to_numpy(dtype=float)
    else:
        xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    pair = adata[:, genes].copy()
    with threadpool_limits(limits=1):
        r, keep = spatialcorr.compute_local_correlation(
            pair, *genes, row_key="array_row", col_key="array_col", condition=None,
            bandwidth=breast_h, contrib_thresh=contrib_thresh)
    keep = np.asarray(keep, dtype=int)
    if len(keep) == 0 or not np.isfinite(r).all():
        raise ValueError("SpatialCorr returned no valid support or undefined correlations.")
    expr = pair.X.toarray() if sparse.issparse(pair.X) else np.asarray(pair.X)
    ix = pair.obs_names[keep]
    y = pd.DataFrame({f"{gene_a}/{gene_b} local r":r, gene_a:expr[keep,0],
                      gene_b:expr[keep,1]}, index=ix)
    xy, z = xy[keep], z[keep]
    table, u = hac_table(y, z, xy)
    folder = Path(out) / f"{gene_a}_{gene_b}_SpatialCorr_Conley_log1p"
    folder.mkdir(parents=True, exist_ok=True)
    y.assign(cancer=z, pixel_col=xy[:,0], pixel_row=xy[:,1]).to_csv(folder/"spot_outcomes.csv", index_label="spot")
    table.to_csv(folder/"cutoff_sensitivity.csv", index=False)
    # Empirical binned semivariograms of the three regression residuals.
    a,b = cKDTree(xy).query_pairs(6000, output_type="ndarray").T
    d = np.linalg.norm(xy[a]-xy[b], axis=1)
    bins = np.minimum((d/300).astype(int),19)
    cnt = np.bincount(bins, minlength=20)
    ok = cnt > 0
    lag = np.bincount(bins, weights=d, minlength=20)[ok]/cnt[ok]
    fig,axes = plt.subplots(1,3,figsize=(12,3.4),layout="constrained")
    for i,(ax,name) in enumerate(zip(axes,y)):
        gamma = np.bincount(bins, weights=.5*(u[a,i]-u[b,i])**2, minlength=20)[ok]/cnt[ok]
        ax.plot(lag,gamma,"o-",ms=3)
        ax.axvline(2000,color="0.4",ls="--")
        ax.set(title=name,xlabel="Distance (full-resolution pixels)",ylabel="Residual semivariance")
    fig.savefig(folder/"residual_semivariograms.png",dpi=180)
    plt.close(fig)
    return table.loc[table.cutoff_px.eq(2000)].reset_index(drop=True), table
