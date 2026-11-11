"""Frozen numerical routines from the Animal 18 reference; use analysis.analyze."""
import warnings
import numpy as np
import pandas as pd

ZERO_DIAGONAL = False

def _symmetric_matrix(S):
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 1:
        raise ValueError("S must be a nonempty square matrix.")
    if not np.isfinite(S).all() or not np.allclose(S, S.T, rtol=1e-8, atol=1e-10):
        raise ValueError("S must be finite and symmetric.")
    return 0.5 * (S + S.T)

def _correlation_batch(matrices):
    matrices = np.array(matrices, dtype=float, copy=True)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or len(matrices) == 0:
        raise ValueError("Expected (settings, genes, genes) correlation matrices.")
    if not np.isfinite(matrices).all() or not np.allclose(matrices, matrices.swapaxes(-1, -2)):
        raise ValueError("Correlation matrices must be finite and symmetric.")
    if np.max(np.abs(matrices)) > 1 + 1e-10:
        raise ValueError("Correlation entries must lie in [-1, 1].")
    matrices = np.clip(0.5 * (matrices + matrices.swapaxes(-1, -2)), -1, 1)
    diagonal = np.arange(matrices.shape[1])
    matrices[:, diagonal, diagonal] = 1
    return matrices

def calc_S_log(cor_matrices, epsilon=0.0, matrix_axis=0, return_details=False):
    """S_ij = sum_s (log(C_s + epsilon I)_ij - mean_log_ij)^2.

    Diagonals are RETAINED, matching calc_S_log in R. Set matrix_axis=2
    when passing an array exported with the original R dimension order.
    epsilon is additive diagonal regularization, not log1p or eigenvalue clipping.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    matrices = np.moveaxis(np.asarray(cor_matrices, dtype=float), matrix_axis, 0)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or len(matrices) == 0:
        raise ValueError("Expected a nonempty batch of square matrices.")
    if not np.isfinite(matrices).all() or not np.allclose(
        matrices, matrices.swapaxes(-1, -2), rtol=1e-8, atol=1e-10
    ):
        raise ValueError("Correlation matrices must be finite and symmetric.")
    p = matrices.shape[1]
    regularized = 0.5 * (matrices + matrices.swapaxes(-1, -2)) + epsilon * np.eye(p)
    values, vectors = np.linalg.eigh(regularized)
    if np.any(values <= 0):
        raise ValueError(
            f"Matrix logarithm needs positive eigenvalues; minimum={values.min():.4g}. "
            "Increase epsilon explicitly. The default R simulation needs n > 15."
        )
    logs = (vectors * np.log(values)[:, None, :]) @ vectors.swapaxes(-1, -2)
    mean_L = logs.mean(axis=0)
    S = np.sum((logs - mean_L)**2, axis=0)
    S = 0.5 * (S + S.T)
    if return_details:
        return {"S": S, "log_cor_matrices": logs, "mean_L": mean_L}
    return S

def transform_correlation_matrices(
    matrices, matrix_type="logeuclidean", epsilon=0.0, fisher_clip_eps=1e-7,
):
    """Apply matrix log, elementwise Fisher Z, or no transformation.

    Raw self-correlations are one. Fisher Z self-correlations are explicitly
    zero BEFORE arctanh. epsilon is used ONLY by the matrix logarithm.
    """
    correlations = _correlation_batch(matrices)
    if matrix_type == "raw":
        return correlations
    if matrix_type == "fisher_z":
        if not np.isfinite(fisher_clip_eps) or not 0 < fisher_clip_eps < 1:
            raise ValueError("FISHER_CLIP_EPS must lie strictly between zero and one.")
        diagonal = np.arange(correlations.shape[-1])
        correlations[:, diagonal, diagonal] = 0
        np.clip(correlations, -1 + fisher_clip_eps, 1 - fisher_clip_eps, out=correlations)
        np.arctanh(correlations, out=correlations)
        return correlations
    if matrix_type == "logeuclidean":
        return calc_S_log(correlations, epsilon=epsilon, return_details=True)["log_cor_matrices"]
    raise ValueError("MATRIX_TYPE must be 'raw', 'fisher_z' or 'logeuclidean'.")

def greedy_backward(S, lam=0.0, zero_diag=None):
    """Delete the smallest current marginal contribution until none remain.

    lam changes reported objectives, not the deletion order. selected is the
    REVERSED removal order. subsets[k] contains the k surviving-longest genes.
    per_cell retains the original R column name but means variation / k**2,
    not variation per spatial cell or per independent observation.
    """
    S = _symmetric_matrix(S)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative.")
    if ZERO_DIAGONAL if zero_diag is None else zero_diag:
        np.fill_diagonal(S, 0)
    p = len(S)
    active = np.ones(p, dtype=bool)
    d = 2 * S.sum(axis=1) - np.diag(S)
    variation = float(S.sum())
    k_path, v_path, removed = [p], [variation], []
    removal_rows = []
    tied_steps = 0
    while active.any():
        candidates = np.flatnonzero(active)
        r = int(candidates[np.argmin(d[candidates])])
        loss = float(d[r])
        tied_steps += int(np.isclose(d[candidates], loss, rtol=1e-10, atol=1e-14).sum() > 1)
        removal_rows.append({"step": len(removed) + 1, "feature_index": r, "marginal_loss": loss})
        variation -= loss
        active[r] = False
        removed.append(r)
        remaining = np.flatnonzero(active)
        d[remaining] -= 2 * S[remaining, r]
        k_path.append(int(active.sum()))
        v_path.append(variation)
    # Remove roundoff at the mathematically empty endpoint only.
    v_path[-1] = 0.0
    selected = np.asarray(removed[::-1], dtype=int)
    v_path[-2] = float(S[selected[0], selected[0]])
    k = np.asarray(k_path)
    variation = np.asarray(v_path)
    per_cell = np.divide(variation, k**2, out=np.full_like(variation, np.nan), where=k > 0)#####################
    return {
        "removed": np.asarray(removed, dtype=int),
        "selected": selected,
        "subsets": [selected[:k].copy() for k in range(p + 1)],
        "path": pd.DataFrame({
            "k": k, "objective": variation - lam * k * (k - 1),
            "unpenalized": variation, "per_cell": per_cell,
        }),
        "removal_table": pd.DataFrame(removal_rows),
        "S_used": S, "tied_steps": tied_steps,
    }

def approx_L1_path_backward(backward, lambda_grid=None, legacy_extra_lambda=False):
    """Maximize V_k - lambda*k*(k-1) over ONLY the backward-generated subsets.

    This is not exhaustive global subset selection. Large lambda changes can
    skip intermediate sizes. Ties prefer the larger subset, matching R's
    descending-k path and ties.method='first'. legacy_extra_lambda=True
    reproduces the erroneous objective values in the original R function.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0, 50, 201)
    lambdas = np.atleast_1d(np.asarray(lambda_grid, dtype=float))
    if lambdas.ndim != 1 or len(lambdas) == 0 or not np.isfinite(lambdas).all() or np.any(lambdas < 0):
        raise ValueError("lambda_grid must contain finite nonnegative values.")
    path = backward["path"]
    k = path["k"].to_numpy(int)
    variation = path["unpenalized"].to_numpy(float)
    objectives = variation[:, None] - (k * (k - 1))[:, None] * lambdas[None, :]
    if legacy_extra_lambda:
        objectives -= lambdas[None, :]
    best_index = np.argmax(objectives, axis=0)
    optimal_k = k[best_index]
    return pd.DataFrame({
        "lambda": lambdas,
        "optimal_k": optimal_k,
        "optimal_objective": objectives[best_index, np.arange(len(lambdas))],
        "unpenalized": variation[best_index],
        "selected": [backward["subsets"][int(size)].copy() for size in optimal_k],
    })

def fit_L1L2_pg(
    S, lambda1_grid=None, lambda2_grid=None,
    n_lambda1=21, n_lambda2=11, n_random=10, w_init=None,
    backward=None, warm_start=True, maxit=2000, tol=1e-8,
    beta=0.5, armijo=1e-4, max_backtrack=50,
    threshold=1e-6, seed=None, verbose=False,
    random_starts=None,
):
    """Reference multistart/backtracking solver with a k*(k-1) first penalty.

    Maximize w.T @ S @ w - lambda1*sum(w)*(sum(w)-1) - lambda2/2*sum(w**2)
    for 0 <= w <= 1. Diagonals are NOT removed automatically.
    Best over starts is not a guarantee of global optimality.
    stationarity is the original unscaled proximal displacement; an additional
    projected_gradient_norm reports that displacement divided by its step.
    Optional random_starts has shape (number_of_grid_points, n_random, p)
    in the solver's sorted grid order. It supports identical R/Python starts.
    """
    S = _symmetric_matrix(S)
    p = len(S)
    if (
        int(n_random) != n_random or n_random < 0
        or int(maxit) != maxit or maxit < 1
        or int(max_backtrack) != max_backtrack or max_backtrack < 1
        or not np.isfinite(tol) or tol <= 0
        or not 0 < beta < 1 or not 0 < armijo < 0.5
        or not np.isfinite(threshold) or threshold < 0
    ):
        raise ValueError("Invalid solver settings.")
    rng = np.random.default_rng(seed)
    eigvals = np.linalg.eigvalsh(S)
    # Exploration endpoint, NOT a bound at which zero becomes optimal.
    lambda1_max = 2.02 * max(float(S.max()), 0.0) if S.max() > 0 else 1.0
    lambda2_max = float(2 * max(0, eigvals.max()))
    if lambda1_grid is None:
        if int(n_lambda1) != n_lambda1 or n_lambda1 < 1:
            raise ValueError("n_lambda1 must be a positive integer.")
        lambda1_grid = np.linspace(0, lambda1_max, int(n_lambda1))
    if lambda2_grid is None:
        if int(n_lambda2) != n_lambda2 or n_lambda2 < 1:
            raise ValueError("n_lambda2 must be a positive integer.")
        lambda2_grid = np.linspace(0, lambda2_max, int(n_lambda2))

    def validate_grid(grid):
        grid = np.atleast_1d(np.asarray(grid, dtype=float))
        if grid.ndim != 1 or len(grid) == 0 or not np.isfinite(grid).all() or np.any(grid < 0):
            raise ValueError("Penalty grids must contain finite nonnegative values.")
        return np.unique(grid)

    lambda1_grid = validate_grid(lambda1_grid)
    lambda2_grid = validate_grid(lambda2_grid)
    if random_starts is not None:
        random_starts = np.asarray(random_starts, dtype=float)
        expected_shape = (len(lambda1_grid) * len(lambda2_grid), int(n_random), p)
        if random_starts.shape != expected_shape or not np.isfinite(random_starts).all() or np.any((random_starts < 0) | (random_starts > 1)):
            raise ValueError(f"random_starts must be finite in [0, 1] with shape {expected_shape}.")
    if w_init is not None:
        w_init = np.asarray(w_init, dtype=float)
        if w_init.shape != (p,) or not np.isfinite(w_init).all():
            raise ValueError("w_init must be a finite length-p vector.")
    if backward is not None and len(backward["subsets"]) != p + 1:
        raise ValueError("backward must have p+1 subsets including k=0.")

    def make_starts(w_warm, grid_index):
        starts = {"zero": np.zeros(p), "one": np.ones(p)}
        if w_init is not None:
            starts["user"] = np.clip(w_init, 0, 1)
        if w_warm is not None:
            starts["warm"] = w_warm.copy()
        if backward is not None:
            sizes = np.unique(np.rint(np.linspace(1, p, min(5, p))).astype(int))
            for size in sizes:
                w0 = np.zeros(p)
                w0[backward["subsets"][int(size)]] = 1
                starts[f"backward_{size}"] = w0
        for j in range(1, int(n_random) + 1):
            if random_starts is not None:
                label = "continuous" if j % 2 else "binary"
                starts[f"{label}_{j}"] = random_starts[grid_index, j-1].copy()
                continue
            if j % 2:
                starts[f"continuous_{j}"] = rng.random(p)
            else:
                starts[f"binary_{j}"] = rng.binomial(1, rng.uniform(0.05, 0.95), p).astype(float)
        return starts

    def prox(w, g, step, lambda1):
        return np.clip(w + step * (g - lambda1 * (2 * w.sum() - 1)), 0, 1)

    def objective(w, Sw, lambda1, lambda2):
        return float(w @ Sw - lambda2 / 2 * (w @ w) - lambda1 * w.sum() * (w.sum() - 1))

    def solve_one(w, lambda1, lambda2, step_init):
        w = np.clip(w, 0, 1).astype(float, copy=True)
        Sw = S @ w
        obj = objective(w, Sw, lambda1, lambda2)
        converged, n_backtrack, iterations = False, 0, 0
        termination = "iteration_limit"
        step_ref = step_init / 2
        # Define this even when the first line search fails (undefined in R).
        stationarity = float(np.max(np.abs(prox(w, 2 * Sw - lambda2 * w, step_ref, lambda1) - w)))
        for iterations in range(1, int(maxit) + 1):
            g = 2 * Sw - lambda2 * w
            step, accepted = step_init, False
            for _ in range(int(max_backtrack)):
                w_new = prox(w, g, step, lambda1)
                d = w_new - w
                Sw_new = S @ w_new
                obj_new = objective(w_new, Sw_new, lambda1, lambda2)
                increase = armijo * (d @ d) / step
                numerical_tol = 1e-12 * max(1, abs(obj))
                if obj_new >= obj + increase - numerical_tol:
                    accepted = True
                    break
                step *= beta
                n_backtrack += 1
            if not accepted:
                termination = "line_search_failed"
                break
            change = float(np.max(np.abs(d)))
            w, Sw, obj = w_new, Sw_new, obj_new
            stationarity = float(np.max(np.abs(
                prox(w, 2 * Sw - lambda2 * w, step_ref, lambda1) - w
            )))
            if change < tol and stationarity < tol:
                converged = True
                termination = "converged"
                break
        total_weight = float(w.sum())
        unpen = float(w @ Sw)
        return {
            "w": w.copy(), "objective": obj, "unpenalized": unpen,
            "f": float(unpen - lambda2 / 2 * (w @ w)),
            "per_cell": unpen / total_weight**2 if total_weight > threshold else np.nan,
            "L1": total_weight, "L2": float(w @ w),
            "n_selected": int(np.sum(w > threshold)),
            "selected": np.flatnonzero(w > threshold),
            "iterations": iterations, "n_backtrack": n_backtrack,
            "stationarity": stationarity,
            "projected_gradient_norm": stationarity / step_ref,
            "converged": converged, "termination": termination,
        }

    fits, start_history, result_rows = [], [], []
    w_warm = None
    # R expand.grid varies lambda1 fastest and lambda2 slowest.
    for lambda2 in lambda2_grid:
        for lambda1 in lambda1_grid:
            L = float(np.max(np.abs(2 * eigvals - lambda2))) + 2 * lambda1 * p
            step_init = 2 / max(L, 1e-12)
            starts = make_starts(w_warm if warm_start else None, len(fits))
            candidates = [solve_one(w, lambda1, lambda2, step_init) for w in starts.values()]
            best_index = int(np.argmax([fit["objective"] for fit in candidates]))
            best = candidates[best_index]
            best["best_start"] = list(starts)[best_index]
            fits.append(best)
            start_history.append(pd.DataFrame({
                "start": list(starts),
                "objective": [fit["objective"] for fit in candidates],
                "converged": [fit["converged"] for fit in candidates],
                "termination": [fit["termination"] for fit in candidates],
            }))
            result_rows.append({"lambda1": lambda1, "lambda2": lambda2, **best})
            # Like R, carry warm starts across lambda2 row boundaries as well.
            w_warm = best["w"].copy()
            if best["converged"] and best["n_selected"] <= 1:
                break
            if verbose:
                print(
                    f"Fit {len(fits)}/{len(lambda1_grid)*len(lambda2_grid)} "
                    f"lambda1={lambda1:.4g} lambda2={lambda2:.4g} "
                    f"objective={best['objective']:.5g} selected={best['n_selected']}"
                )
    if any(not fit["converged"] for fit in fits):
        warnings.warn("Some best-start fits did not converge; inspect termination and diagnostics.")
    return {
        "results": pd.DataFrame(result_rows), "fits": fits,
        "start_history": start_history,
        "lambda1_grid": np.array([row["lambda1"] for row in result_rows]), "lambda2_grid": lambda2_grid,
        "lambda1_max": lambda1_max, "lambda2_max": lambda2_max,
    }
