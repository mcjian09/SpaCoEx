"""Batched two-gene SpatialCorr BR tests for the saved Visium master ranking."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import spatialcorr.statistical_test as st
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits


def pairwise_br(top_n=16, n_shuffles=999, seed=1, *, genes=None, cancer_mode="expanded"):
    """Test a supplied gene list, or whole top-ranked groups if genes is None."""
    if not isinstance(n_shuffles, (int, np.integer)) or n_shuffles < 1:
        raise ValueError("n_shuffles must be a positive integer.")
    folder = Path(__file__).resolve().parent
    job = runpy.run_path(str(folder / "panel_pvalues.py"))
    adata, _ = job["prepare"](1)
    if cancer_mode == "strict":
        fine = adata.obs["ground_truth"].astype(str)
        adata.obs["region"] = np.where(
            fine.str.startswith(("IDC", "DCIS/LCIS")), "Cancer", "Non-cancer"
        )
    elif cancer_mode != "expanded":
        raise ValueError("cancer_mode must be 'strict' or 'expanded'.")
    if genes is None:
        ranks = pd.read_csv(folder.parents[1] / "reference/visium/panel_ranks.csv")
        ranks = ranks.sort_values(["master_gradient", "gene"])
        genes = ranks.loc[ranks.master_gradient <= top_n, "gene"].to_numpy()
        if len(genes) != top_n or top_n < 2:
            raise ValueError("Choose a top_n >= 2 ending at a complete rank group (e.g. 8 or 16).")
    else:
        genes = np.asarray(genes, dtype=str)
        if genes.ndim != 1 or len(genes) < 2 or len(np.unique(genes)) != len(genes):
            raise ValueError("Supply a one-dimensional list of at least two distinct genes.")
        missing = sorted(set(genes) - set(adata.var_names))
        if missing:
            raise ValueError(f"Genes absent from the 200-candidate data: {missing}")
    expr = adata[:, genes].X
    x = np.array(expr.toarray() if sparse.issparse(expr) else expr, dtype=float, copy=True)
    obs = adata.obs
    K = st.compute_kernel_matrix(obs, bandwidth=5.006316010800757, region_key="region",
                                 condition_on_region=True, y_col="array_row", x_col="array_col")
    keep = np.flatnonzero(K.sum(axis=1) >= 10)
    if cancer_mode == "expanded" and len(keep) != 3767:
        raise ValueError("Unexpected Visium spot filter.")
    ids = {key: np.flatnonzero(obs.region == key) for key in ("Cancer", "Non-cancer")}
    if any(np.isin(keep, idx).sum() < 3 for idx in ids.values()):
        raise ValueError("Fewer than three retained spots in a comparison group.")
    print(f"{cancer_mode} cancer definition: kept {len(keep)}/{len(obs)} spots.", flush=True)
    groups = []
    a, b = np.triu_indices(len(genes), k=1)
    region_r = {}
    for key, idx in ids.items():
        vals = x[idx]
        if not np.isfinite(vals).all() or np.any(vals.std(axis=0) == 0):
            raise ValueError(f"Nonfinite expression or constant gene in {key}.")
        region_r[key] = np.corrcoef(vals.T)[a, b]
        x[idx] = (vals - vals.mean(axis=0)) / vals.std(axis=0)
        kernel = K[np.ix_(idx, idx)]
        kernel /= kernel.sum(axis=1, keepdims=True)
        groups.append((idx, np.isin(idx, keep), kernel))

    pooled = np.corrcoef(x.T)
    r0 = pooled[a, b]
    den0 = 1 - r0*r0
    if np.any(den0 <= 0):
        raise ValueError("Singular pooled correlation for at least one pair.")

    def score(y):
        total = np.zeros(len(a))
        for idx, kept, kernel in groups:
            v = y[idx]
            d = v - kernel @ v
            z = d[kept] / np.sqrt((kernel @ np.square(d))[kept])
            h = z.T @ z
            r = np.corrcoef(v.T)[a, b]
            den = 1 - r*r
            # Expand the native Gaussian likelihood ratio using the inverse and
            # determinant of a 2 x 2 correlation matrix, for every pair at once.
            total += 0.5 * (
                len(z) * (np.log(den0) - np.log(den))
                + (1 / den0 - 1 / den) * (h[a, a] + h[b, b])
                + 2 * (r / den - r0 / den0) * h[a, b]
            )
        if not np.isfinite(total).all():
            raise ValueError("Undefined BR score (singular pair or zero local variance).")
        return total

    def native(y, j):
        cols = [a[j], b[j]]
        shared = np.broadcast_to(pooled[np.ix_(cols, cols)], (len(keep), 2, 2))
        return st.compute_llrts_between(obs.iloc[keep], y[:, cols].T, K, ids, shared, keep)[0]

    rng = np.random.RandomState(seed)
    hits = np.zeros(len(a), dtype=int)
    check = np.unique([0, len(a)//2, len(a)-1])
    with threadpool_limits(limits=2), np.errstate(invalid="raise", divide="raise"):
        observed = score(x)
        for j in check:
            np.testing.assert_allclose(observed[j], native(x, j), rtol=1e-9, atol=1e-7)
        for i in range(n_shuffles):
            y = rng.permutation(x)
            perm = score(y)
            if i == 0:
                for j in check:
                    np.testing.assert_allclose(perm[j], native(y, j), rtol=1e-9, atol=1e-7)
            # SpatialCorr counts strictly greater scores; use native arithmetic near ties.
            for j in np.flatnonzero(np.abs(perm - observed) < 1e-7):
                perm[j], observed[j] = native(y, j), native(x, j)
            hits += perm > observed
            if (i + 1) % 250 == 0:
                print(f"{i+1}/{n_shuffles} shuffles, all {len(a)} pairs", flush=True)
    p = (hits + 1) / (n_shuffles + 1)
    delta = region_r["Cancer"] - region_r["Non-cancer"]
    table = pd.DataFrame({
        "gene_a": genes[a], "gene_b": genes[b],
        "r_cancer": region_r["Cancer"], "r_non_cancer": region_r["Non-cancer"],
        "delta_r": delta, "abs_delta_r": np.abs(delta), "BR_score": observed,
        "n_shuffles": n_shuffles, "shuffles_above": hits, "BR_p": p,
        "BR_q_BH": multipletests(p, method="fdr_bh")[1],
        "cancer_mode": cancer_mode, "permutation_seed": seed,
        "n_cancer": len(ids["Cancer"]), "n_non_cancer": len(ids["Non-cancer"]),
        "n_kept_cancer": np.isin(keep, ids["Cancer"]).sum(),
        "n_kept_non_cancer": np.isin(keep, ids["Non-cancer"]).sum(),
    })
    return table.sort_values(["BR_q_BH", "BR_p", "abs_delta_r"], ascending=[True, True, False],
                             ignore_index=True)
