"""Version 1.1: independent per-run expression normalization and log1p controls."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import frozen_math as math


VERSION = "1.1"
RIDGE, FISHER_EPS, SPACINGS = 1e-6, 1e-7, 3.54
ACTIVE, TOL, MAX_ITER, RANDOM_STARTS, SEED = 1e-6, 1e-8, 10000, 10, 1


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_ids(z, key, n):
    ids = z[key]
    if ids.shape != (n,) or ids.dtype.kind != "U" or len(set(ids)) != n or any(not s.strip() for s in ids):
        raise ValueError(f"{key}: expected {n} unique, nonempty Unicode strings.")
    return ids


def resolve_preprocessing(config, normalization=None, log1p=None):
    policies = {("none", False): "as_provided", ("none", True): "log1p_supplied",
                ("cp10k", False): "cp10k", ("cp10k", True): "cp10k_log1p"}
    if "preprocessing" in config:
        if "normalization" in config or "log1p" in config:
            raise ValueError("Use either the legacy preprocessing name or normalization/log1p in a run file, not both.")
        old = {name: pair for pair, name in policies.items()}
        if config["preprocessing"] not in old:
            raise ValueError(f"Unknown preprocessing: {config['preprocessing']}")
        base, logged = old[config["preprocessing"]]
    else:
        base, logged = config.get("normalization", "none"), config.get("log1p", False)
    base = base if normalization is None else normalization
    logged = logged if log1p is None else log1p
    if base not in ("none", "cp10k") or type(logged) is not bool:
        raise ValueError("normalization must be 'none' or 'cp10k'; log1p must be True or False.")
    return base, logged, policies[base, logged]


def load_inputs(path, processing):
    with np.load(path, allow_pickle=False) as z:
        X, xy = np.asarray(z["X"], dtype=float), np.asarray(z["xy"], dtype=float)
        if X.ndim != 2 or min(X.shape) < 2 or xy.shape != (len(X), 2):
            raise ValueError("Expected X (cells x candidate genes) and xy (same cells x 2).")
        if not np.isfinite(X).all() or not np.isfinite(xy).all():
            raise ValueError("Expression and coordinates must be finite.")
        genes = text_ids(z, "genes", X.shape[1])
        ids = text_ids(z, "expression_ids", len(X))
        coords = text_ids(z, "coordinate_ids", len(xy))
        if set(ids) != set(coords):
            raise ValueError("Coordinate IDs and expression IDs do not match.")
        xy = xy[pd.Index(coords).get_indexer(ids)]
        if processing in ("cp10k", "cp10k_log1p"):
            if "library_total" not in z:
                raise ValueError("cp10k normalization requires full-library totals in the NPZ field 'library_total'.")
            total = np.asarray(z["library_total"], dtype=float)
            if (total.shape != (len(X),) or not np.isfinite(total).all() or (total <= 0).any()
                    or (X < 0).any() or (X != np.floor(X)).any()
                    or (total + 1e-8 < X.sum(axis=1)).any()):
                raise ValueError("Provide raw counts and full positive per-spot totals in expression-ID order.")
            X = 10000 * X / total[:, None]
            if processing == "cp10k_log1p":
                X = np.log1p(X)
        elif processing == "log1p_supplied":
            if (X < 0).any():
                raise ValueError("log1p_supplied expects nonnegative expression, not residuals.")
            X = np.log1p(X)
        elif processing != "as_provided":
            raise ValueError("Unknown expression preprocessing.")
    return X, xy, genes


def variation(X, xy, matrix, batch=12):
    n, p = X.shape
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    h = SPACINGS * spacing
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Median nearest-cell distance is zero or invalid; check coordinates.")

    def kernel(i):
        K = np.exp(-cdist(xy[i:i+batch], xy, "sqeuclidean") / h**2)
        return K / K.sum(axis=1, keepdims=True)

    residuals = np.empty_like(X)
    ess = np.empty(n)
    for i in range(0, n, batch):
        K = kernel(i)
        residuals[i:i+batch] = X[i:i+batch] - K @ X
        ess[i:i+batch] = 1 / (K**2).sum(axis=1)
    mean, ss = np.zeros((p, p)), np.zeros((p, p))
    for i in range(0, n, batch):
        K = kernel(i)
        raw = np.empty((len(K), p, p))
        for j, row in enumerate(K):
            keep = row > 0
            a = residuals[keep] * np.sqrt(row[keep, None])
            cov = a.T @ a
            sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
            raw[j] = np.clip(cov / np.outer(sd, sd), -1, 1)
            np.fill_diagonal(raw[j], 1)
        b = math.transform_correlation_matrices(raw, matrix, RIDGE, FISHER_EPS)
        bm = b.mean(axis=0)
        delta = bm - mean
        ss += ((b-bm)**2).sum(axis=0) + delta**2 * (i*len(b)/(i+len(b)))
        mean += delta * (len(b)/(i+len(b)))
    S = (ss + ss.T) * 0.5
    if not np.isfinite(S).all() or S.sum() <= 0:
        raise ValueError("No finite positive spatial association variation to select.")
    return S, h, ess


def select(S, genes, method, grid):
    ranking = None
    if method == "gradient":
        result = math.fit_L1L2_pg(
            S, lambda1_grid=grid, lambda2_grid=[0.], threshold=ACTIVE, seed=SEED,
            n_random=RANDOM_STARTS, tol=TOL, maxit=MAX_ITER,
        )["results"]
        W = np.stack(result["w"])
        summary = result[["lambda1", "objective", "converged", "termination",
                          "iterations", "best_start"]].rename(columns={"lambda1": "lambda"}).copy()
    elif method == "backward":
        back = math.greedy_backward(S, lam=0., zero_diag=False)
        sweep = math.approx_L1_path_backward(back, grid)
        W = np.zeros((len(grid), len(genes)))
        for i, inds in enumerate(sweep.selected):
            W[i, inds] = 1
        summary = sweep[["lambda", "optimal_objective"]].rename(columns={"optimal_objective": "objective"}).copy()
        ranking = back["removal_table"].iloc[::-1].reset_index(drop=True).copy()
        ranking.insert(0, "rank_last_removed_first", np.arange(1, len(genes)+1))
        ranking.insert(1, "gene", genes[ranking.pop("feature_index").to_numpy()])
        retention = back["path"][["k","unpenalized"]].sort_values("k").copy()
        retention["percent_retained"] = 100 * retention.pop("unpenalized") / S.sum()
    else:
        raise ValueError("Unknown selection method.")
    active = W > ACTIVE
    summary["active_genes"] = active.sum(axis=1)
    summary["total_weight"] = W.sum(axis=1)
    summary["percent_retained"] = 100 * np.sum((active @ S)*active, axis=1) / S.sum()
    if method == "gradient":
        retention = summary[["lambda", "active_genes", "percent_retained", "converged"]].rename(columns={"active_genes":"k"})
    return W, summary, retention, ranking


def analyze(config_path, *, matrix=None, selection=None, normalization=None, log1p=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"dataset", "input", "coordinate_units", "candidate_policy", "input_description"}
    missing = required - config.keys()
    extra = config.keys() - required - {"matrix", "selection", "preprocessing", "log1p", "normalization"}
    if missing or extra or any(not isinstance(config.get(k), str) or not config[k].strip() for k in required):
        raise ValueError(f"Invalid run configuration: missing={missing}, unknown={extra}; required fields must be nonempty strings.")
    matrix = config.get("matrix", "logeuclidean") if matrix is None else matrix
    method = config.get("selection", "gradient") if selection is None else selection
    norm, logged, processing = resolve_preprocessing(config, normalization, log1p)
    if matrix not in ("logeuclidean", "raw", "fisher_z") or method not in ("gradient", "backward"):
        raise ValueError("matrix: logeuclidean/raw/fisher_z; selection: gradient/backward.")
    source = (config_path.parent/config["input"]).resolve()
    fingerprint = sha256(source)
    X, xy, genes = load_inputs(source, processing)
    with threadpool_limits(limits=2):
        S, h, ess = variation(X, xy, matrix)
        grid = np.linspace(0, 2.02 * S.max(), 401)
        W, summary, retention, ranking = select(S, genes, method, grid)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = config_path.parent/"runs"/f"{stamp}_{fingerprint[:10]}_{matrix}_{method}"
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/"association.npz", S=S, genes=genes)
    pd.DataFrame(W, columns=genes, index=pd.Index(summary["lambda"], name="lambda")).to_csv(out/"weights.csv")
    summary.to_csv(out/"lambda_path.csv", index=False)
    retention.to_csv(out/"retention.csv", index=False)
    if ranking is not None:
        ranking.to_csv(out/"backward_ranking.csv", index=False)
    manifest = {**config, "release":VERSION, "input_sha256":fingerprint,
                "method_sha256":{f:sha256(Path(__file__).with_name(f)) for f in ("analysis.py","frozen_math.py")},
                "matrix":matrix,"selection":method,"normalization":norm,"log1p":logged,
                "preprocessing":processing,"cells":len(X),"features":len(genes),
                "spacings":SPACINGS,"bandwidth":h,"median_effective_neighbors":float(np.median(ess)),
                "kernel":"exp(-distance_squared / bandwidth_squared); self included; source-centered",
                "ridge":RIDGE if matrix=="logeuclidean" else None,"fisher_clip":FISHER_EPS if matrix=="fisher_z" else None,
                "zero_S_diagonal":False,"S_scaling":"sum of squared deviations, not normalized",
                "penalty":"lambda * k * (k - 1), k = sum(w)", "lambda_grid":"401 linear points, 0 to 2.02*max(S)",
                "requested_grid_points":len(grid),"evaluated_grid_points":len(summary),
                "terminal_lambda":float(summary["lambda"].iloc[-1]),"active_threshold":ACTIVE,
                "gradient_settings":dict(seed=SEED, random_starts=RANDOM_STARTS, max_iter=MAX_ITER,
                    tolerance=TOL, ascending=True, warm_start=True, stop="first converged fit with <=1 active") if method=="gradient" else None,
                "python":platform.python_version(),"versions":{m:version(m) for m in ("numpy","scipy","pandas","matplotlib","threadpoolctl")}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14,4), layout="constrained")
    if method == "gradient":
        r = retention.loc[retention.converged]
        axes[0].scatter(r.k, r.percent_retained, s=16)
    else:
        axes[0].plot(retention.k, retention.percent_retained, marker=".", lw=1)
    axes[0].set(xlabel="Number of genes", ylabel="Co-expression variation retained", ylim=(0,101))
    axes[0].yaxis.set_major_formatter(PercentFormatter(100))
    x = np.sqrt(summary["lambda"])
    axes[1].plot(x, W, lw=.8, alpha=.65, drawstyle="steps-post" if method=="backward" else "default")
    axes[1].set(xlabel=r"$\sqrt{\lambda}$", ylabel="Gene weight")
    axes[2].step(x, summary.active_genes, where="post", label="Active genes")
    axes[2].plot(x, summary.total_weight, "--", label="Total weight")
    axes[2].set(xlabel=r"$\sqrt{\lambda}$", ylabel="Count or total weight")
    axes[2].legend(frameon=False)
    fig.savefig(out/"paths.png", dpi=220)
    print(f"{config['dataset']} | {matrix} | {method} | normalization={norm}, log1p={logged} | "
          f"{len(X)} cells x {len(genes)} features\n{out}")
    return out, summary, ranking, fig
