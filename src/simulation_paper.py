import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _symmetric_matrix(S):
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 1:
        raise ValueError("S must be a nonempty square matrix.")
    if not np.isfinite(S).all() or not np.allclose(S, S.T, rtol=1e-8, atol=1e-10):
        raise ValueError("S must be finite and symmetric.")
    return 0.5 * (S + S.T)

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

def _covariance_to_correlation(covariance):
    covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2))
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    if not np.isfinite(covariance).all() or np.any(variance <= 0):
        raise ValueError("Gaussian pooling produced undefined variances; check constant genes.")
    sd = np.sqrt(variance)
    result = covariance / (sd[:, :, None] * sd[:, None, :])
    return _correlation_batch(result)

def gaussian_setting_correlations(X, population_correlations, bandwidth=5.0):
    """Gaussian-weighted Pearson correlations across replicated settings.

    K[s,t] = exp(-((s-t)/bandwidth)**2), including the focal setting.
    Every observation in setting t has weight K[s,t]/(n*sum_t K[s,t]).
    First/second moments avoid a (settings*n) squared kernel allocation.
    Population targets assume zero means and unit variances, as generated
    by simulate_five_groups. This uses focal weighted-mean centering.
    """
    X = np.asarray(X, dtype=float)
    population = _correlation_batch(population_correlations)
    if X.ndim != 3 or X.shape[0] != len(population) or X.shape[2] != population.shape[1]:
        raise ValueError("X and population matrices have incompatible dimensions.")
    if not np.isfinite(X).all() or X.shape[1] < 2:
        raise ValueError("Need finite data and at least two observations per setting.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Gaussian bandwidth must be positive and finite.")
    m, n, _ = X.shape
    distance = np.arange(m)[:, None] - np.arange(m)[None, :]
    kernel = np.exp(-(distance / float(bandwidth))**2)
    weights = kernel / kernel.sum(axis=1, keepdims=True)
    first = weights @ X.mean(axis=1)
    second_per_setting = np.einsum("sni,snj->sij", X, X, optimize=True) / n
    second = np.einsum("st,tij->sij", weights, second_per_setting, optimize=True)
    covariance = second - first[:, :, None] * first[:, None, :]
    correlations = _covariance_to_correlation(covariance)
    # E[X_t]=0 and Var(X_t)=1 in this simulation. Pool the population
    # covariance matrices too, to show the appropriate smoothed target.
    target_covariance = np.einsum("st,tij->sij", weights, population, optimize=True)
    target = _covariance_to_correlation(target_covariance)
    effective_settings = 1 / np.sum(weights**2, axis=1)
    return {
        "cor_matrices": correlations, "population_cor_matrices": target,
        "kernel": kernel, "kernel_weights": weights,
        "effective_settings": effective_settings,
        "effective_observations": n * effective_settings,
        "bandwidth": float(bandwidth),
    }

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

def association_variation(matrices):
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or not np.isfinite(matrices).all():
        raise ValueError("Need a finite batch of feature matrices.")
    centered = matrices - matrices.mean(axis=0)
    S = np.sum(centered**2, axis=0)
    return 0.5 * (S + S.T)

def apply_matrix_choices(
    simulation, correlation_estimator="sample", matrix_type="logeuclidean",
    gaussian_bandwidth=5.0, epsilon=0.0, fisher_clip_eps=1e-7,
):
    """Calculate selected estimator/transform without resimulating expression.

    Retains the input sample matrices so reapplying settings never smooths
    an already smoothed batch. S retains its diagonal; downstream switches
    determine whether it is used by gradient ascent.
    """
    if correlation_estimator not in ("sample", "gaussian"):
        raise ValueError("CORRELATION_ESTIMATOR must be 'sample' or 'gaussian'.")
    sample = simulation.get("sample_cor_matrices", simulation["cor_matrices"])
    population = simulation.get("original_population_cor_matrices", simulation["population_cor_matrices"])
    result = dict(simulation)
    result["sample_cor_matrices"] = sample
    result["original_population_cor_matrices"] = population
    if correlation_estimator == "gaussian":
        pooled = gaussian_setting_correlations(simulation["X"], population, gaussian_bandwidth)
        correlations = pooled["cor_matrices"]
        target = pooled["population_cor_matrices"]
        result.update({key: value for key, value in pooled.items() if key not in ("cor_matrices", "population_cor_matrices")})
    else:
        correlations = _correlation_batch(sample)
        target = _correlation_batch(population)
        result["effective_settings"] = np.ones(len(correlations))
        result["effective_observations"] = np.full(len(correlations), simulation["X"].shape[1])
        for key in ("kernel", "kernel_weights", "bandwidth"):
            result.pop(key, None)
    metric = transform_correlation_matrices(correlations, matrix_type, epsilon, fisher_clip_eps)
    population_metric = transform_correlation_matrices(target, matrix_type, epsilon, fisher_clip_eps)
    result.update({
        "cor_matrices": correlations, "population_cor_matrices": target,
        "metric_matrices": metric, "population_metric_matrices": population_metric,
        "S": association_variation(metric),
        "population_S": association_variation(population_metric),
        "correlation_estimator": correlation_estimator, "matrix_type": matrix_type,
        "population_S_raw": association_variation(target),
    })
    if matrix_type == "logeuclidean":
        result["log_cor_matrices"] = metric
        result["mean_L"] = metric.mean(axis=0)
        result["population_S_log"] = result["population_S"]
    else:
        for key in ("log_cor_matrices", "mean_L", "population_S_log"):
            result.pop(key, None)
    return result

def simulate_five_groups(
    n_sim=100, n=200, seed=123,
    strong_range=(0.0, 0.8), medium_range=(0.2, 0.6),
    weak_range=(0.35, 0.45), constant_rho=0.4,
):
    """Generate 50 genes: ten each of strong/medium/weak/constant/independent.

    Each setting contains n independent N(0, Sigma_s) observations. Every
    marginal variance is one. Within a block all off-diagonal correlations
    share rho_s; between blocks all population correlations are zero.
    The varying rhos follow linear schedules across ordered settings.
    These settings are not 2D tissue coordinates.
    """
    if int(n_sim) != n_sim or n_sim < 2 or int(n) != n or n < 2:
        raise ValueError("n_sim and n must be integers of at least 2.")
    n_sim, n = int(n_sim), int(n)

    def schedule(endpoints, name):
        endpoints = np.asarray(endpoints, dtype=float)
        if endpoints.shape != (2,) or not np.isfinite(endpoints).all():
            raise ValueError(f"{name} needs two finite endpoints.")
        return np.linspace(*endpoints, n_sim)

    if not np.isfinite(constant_rho):
        raise ValueError("constant_rho must be finite.")
    labels = ["Strongly varying", "Medium varying", "Weakly varying",
              "Constant correlation", "Independent"]
    prefixes = ["S", "M", "W", "C", "I"]
    schedules = np.column_stack([
        schedule(strong_range, "strong_range"),
        schedule(medium_range, "medium_range"),
        schedule(weak_range, "weak_range"),
        np.full(n_sim, constant_rho), np.zeros(n_sim),
    ])
    if np.any(schedules <= -1 / 9) or np.any(schedules >= 1):
        raise ValueError("Ten-gene equicorrelation blocks require -1/9 < rho < 1.")
    genes = np.array([f"{prefix}{j}" for prefix in prefixes for j in range(1, 11)])
    groups = np.repeat(labels, 10)
    population = np.zeros((n_sim, 50, 50))
    samples = np.empty((n_sim, n, 50))
    correlations = np.empty_like(population)
    rng = np.random.default_rng(seed)
    for s in range(n_sim):
        for block, rho in enumerate(schedules[s]):
            start = 10 * block
            covariance = np.full((10, 10), rho)
            np.fill_diagonal(covariance, 1)
            population[s, start:start + 10, start:start + 10] = covariance
        samples[s] = rng.standard_normal((n, 50)) @ np.linalg.cholesky(population[s]).T
        correlations[s] = np.corrcoef(samples[s], rowvar=False)
    return {
        "X": samples, "cor_matrices": correlations,
        "population_cor_matrices": population,
        "population_S_raw": np.sum((population - population.mean(axis=0))**2, axis=0),
        "gene_names": genes, "gene_group": groups,
        "gene_role": np.repeat("block member", 50),
        "group_labels": np.array(labels), "rho_by_group": schedules,
        "group_colors": dict(zip(labels, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"])),
        "seed": seed,
    }

"""Plot/export helpers embedded in the standalone simulation notebook."""
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter





from threadpoolctl import threadpool_limits
try:
    from numba import njit
except ImportError:
    def njit(fn):  # Optional acceleration, identical numerical rules.
        return fn

@njit
def ascend_one(S, lam, pair_cost, start, step0, tol, maxit):
    """Status: 0=stationary, 1=returned zero (k<1), 2=iteration limit, 3=line search failed."""
    w = start.copy()
    if w.sum() < 1.0:
        return np.zeros_like(w), 1, 0
    Sw = S @ w
    step = step0
    for it in range(1, maxit + 1):
        k = w.sum()
        g = 2.0 * Sw - lam * (2.0*k - 1.0 if pair_cost else 1.0)
        probe = np.clip(w + step0*g, 0.0, 1.0)
        if probe.sum() >= 1.0 and np.max(np.abs(probe - w)) <= tol:
            return w, 0, it
        for _ in range(60):
            trial = np.clip(w + step*g, 0.0, 1.0)
            crossed = trial.sum() < 1.0
            d = trial - w
            Sd = S @ d
            # Exact objective gain for these quadratic objectives.
            gain = g @ d + d @ Sd
            if pair_cost:
                gain -= lam * d.sum()**2
            if gain >= 1e-4 * (d @ d) / step:
                if crossed:
                    # This start ends here. Do not score or export the fractional trial.
                    return np.zeros_like(w), 1, it
                w = trial
                Sw = S @ w
                step = min(1.5 * step, 1e6 * step0)
                break
            step *= 0.5
        else:
            return w, 3, it
    return w, 2, maxit


def make_starts(S, seed, n_random):
    p = len(S)
    unit = np.eye(p)
    starts = [np.ones(p), np.zeros(p), unit[np.argmax(np.diag(S))]]
    # Deterministic two-gene candidate at k=1, useful even when the diagonal is zero.
    a, b = np.triu_indices(p, 1)
    best = np.argmax(S.diagonal()[a] + S.diagonal()[b] + 2*S[a, b])
    starts.append((unit[a[best]] + unit[b[best]]) / 2)
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        w = rng.random(p) if i % 2 == 0 else rng.binomial(1, rng.uniform(.05, .95), p).astype(float)
        starts.append(w)
    return starts


def gradient_path(S, penalty, grid):
    S = np.ascontiguousarray(_symmetric_matrix(S))
    if np.any(S < 0) or S.sum() <= 0:
        raise ValueError("The variation matrix must be nonnegative with positive sum.")
    if penalty not in ("k", "k_kminus1"):
        raise ValueError("Unrecognized penalty.")
    pair = penalty == "k_kminus1"
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or grid.min() < 0 or np.any(np.diff(grid) <= 0)):
        raise ValueError("Need an increasing finite, nonnegative lambda grid.")
    if N_RANDOM_STARTS < 0 or MAX_ITER < 1 or OPT_TOL <= 0 or not 0 <= ACTIVE_THRESHOLD < 1:
        raise ValueError("Invalid optimization controls.")
    starts = make_starts(S, SEED, N_RANDOM_STARTS)
    step0 = 1.0 / (2.0 * np.max(np.abs(np.linalg.eigvalsh(S))))
    rows, weights, warm = [], [], None
    for lam in grid:
        best_score = -np.inf
        candidates = starts if warm is None else [warm, *starts]
        for start in candidates:
            w, status, it = ascend_one(S, lam, pair, start, step0, OPT_TOL, MAX_ITER)
            k = w.sum()
            score = w @ S @ w - lam * (k*(k-1) if pair else k)
            if score > best_score:
                best_score, winner, reason, its = score, w, status, it
        warm = winner.copy()
        active = winner > ACTIVE_THRESHOLD
        percent = 100 * S[np.ix_(active, active)].sum() / S.sum()
        rows.append((lam, warm.sum(), active.sum(), percent, best_score,
                     ("stationary", "zero_cutoff", "iteration_limit", "line_search_failed")[reason], its))
        weights.append(warm)
    result = pd.DataFrame(rows, columns=["lambda", "total_weight", "n_active",
                          "variance_explained_pct", "objective", "stop", "iterations"])
    return {"summary": result, "weights": np.stack(weights)}

def build_export_paths(simulation, correlation_estimator, gaussian_bandwidth,
                       epsilon, fisher_clip_eps, zero_diag, n_lambda=501):
    """Compute each S once. Fits use the actual lambda, not its plotted square root."""
    if int(n_lambda) != n_lambda or n_lambda < 2:
        raise ValueError("n_lambda must be an integer >= 2.")
    results = {}
    with threadpool_limits(limits=1, user_api="blas"):
        for representation in ("logeuclidean", "raw", "fisher_z"):
            transformed = apply_matrix_choices(
                simulation, correlation_estimator=correlation_estimator,
                matrix_type=representation, gaussian_bandwidth=gaussian_bandwidth,
                epsilon=epsilon, fisher_clip_eps=fisher_clip_eps,
            )
            S = transformed["S"].copy()
            if zero_diag:
                np.fill_diagonal(S, 0.)
            for penalty in ("k", "k_kminus1"):
                # For nonnegative S and k >= 1 these endpoints give a negative
                # gradient on every positive coordinate. No removal order is used.
                upper = 2.02 * (S.sum(1).max() if penalty == "k" else S.max())
                grid = np.linspace(0., upper, int(n_lambda))
                fit = gradient_path(S, penalty, grid)
                df, W = fit["summary"], fit["weights"]
                results[penalty, representation] = dict(
                    S=S, weights=W, summary=df, **{"lambda": grid},
                    selected_k=df.n_active.to_numpy(), k=df.n_active.to_numpy(),
                    percent_retained=df.variance_explained_pct.to_numpy(),
                    total_weight=df.total_weight.to_numpy(),
                )
    return results


def make_variance_figure(result, figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.scatter(result["k"], result["percent_retained"], s=18, color="#0072B2", clip_on=False)
    ax.set_xlabel("number genes", fontsize=14)
    ax.set_ylabel("variance explained", fontsize=14)
    ax.set_xlim(0, len(result["S"]))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    return fig


def lambda_axis(lambdas, penalty):
    if penalty == "k_kminus1":
        return np.sqrt(lambdas), r"$\sqrt{\lambda}$"
    if penalty == "k":
        return lambdas, "lambda"
    raise ValueError("Unknown penalty for lambda axis.")


def make_weight_figure(result, simulation, figsize=(10, 5.5), x_jitter=.003, seed=123, *, penalty):
    if not np.isfinite(x_jitter) or x_jitter < 0:
        raise ValueError("x_jitter must be finite and nonnegative.")
    colors = simulation["group_colors"].copy()
    colors["Strongly varying"] = "#D55E00"
    colors["Medium varying"] = "#0072B2"
    x, xlabel = lambda_axis(result["lambda"], penalty)
    weights = result["weights"]
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-1, 1, weights.shape[1]) * x_jitter * np.ptp(x)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for index, group in enumerate(simulation["gene_group"]):
        display_x = np.clip(x + offsets[index], x[0], x[-1])
        display_x[0], display_x[-1] = x[0], x[-1]
        ax.step(display_x, weights[:, index], where="post", color=colors[group],
                linewidth=1.2, alpha=.65)
    handles = [Line2D([0], [0], color=colors[group], linewidth=2, label=group)
               for group in simulation["group_labels"]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=3, frameon=False, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("weight", fontsize=14)
    ax.set_xscale("linear")
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylim(-.03, 1.03)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    return fig


def make_total_weight_figure(result, figsize=(8, 4.5), *, penalty):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    selected_count = (result["weights"] > ACTIVE_THRESHOLD).sum(axis=1)
    np.testing.assert_array_equal(selected_count, result["selected_k"])
    x, xlabel = lambda_axis(result["lambda"], penalty)
    ax.step(x, selected_count, where="post", color="#0072B2", linewidth=2)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("number selected", fontsize=14)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, result["weights"].shape[1] + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    return fig


def export_eighteen_images(results, simulation, zip_path, image_format="png", dpi=300,
                         variance_figsize=(8, 4.5), weight_figsize=(10, 5.5), total_weight_figsize=(8, 4.5),
                         x_jitter=.003, jitter_seed=123, show_plots=False):
    """Write exactly 18 images with representation headers and no extra members."""
    image_format = image_format.lower()
    if image_format not in ("png", "svg"):
        raise ValueError("IMAGE_FORMAT must be 'png' or 'svg'.")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI must be positive.")
    archive_buffer = io.BytesIO()
    rows = []
    slugs = {"logeuclidean": "log_euclidean", "raw": "correlation", "fisher_z": "fisher_z"}
    headers = {"logeuclidean": "Log Euclidean", "raw": "Raw", "fisher_z": "Fisher Z"}
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for penalty in ("k", "k_kminus1"):
            for representation in ("logeuclidean", "raw", "fisher_z"):
                result = results[(penalty, representation)]
                for kind in ("variance_explained", "weight_vs_lambda", "number_selected_vs_lambda"):
                    if kind == "variance_explained":
                        fig = make_variance_figure(result, variance_figsize)
                    elif kind == "weight_vs_lambda":
                        fig = make_weight_figure(result, simulation, weight_figsize,
                                                 x_jitter, jitter_seed, penalty=penalty)
                    else:
                        fig = make_total_weight_figure(result, total_weight_figsize, penalty=penalty)
                    try:
                        fig.suptitle(headers[representation], fontsize=14)
                        image_buffer = io.BytesIO()
                        fig.savefig(image_buffer, format=image_format, dpi=dpi, facecolor="white")
                        filename = f"{len(rows)+1:02d}_penalty_{penalty}_{slugs[representation]}_{kind}.{image_format}"
                        archive.writestr(filename, image_buffer.getvalue())
                        rows.append({"file": filename, "penalty": penalty,
                                     "representation": representation, "plot": kind})
                        if show_plots:
                            plt.show()
                    finally:
                        plt.close(fig)
    manifest = pd.DataFrame(rows)
    with ZipFile(io.BytesIO(archive_buffer.getvalue())) as check:
        if len(check.namelist()) != 18 or len(set(check.namelist())) != 18 or check.testzip() is not None:
            raise RuntimeError("Archive verification failed.")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(archive_buffer.getvalue())
    return manifest

# -------------------- USER CONTROLS --------------------
# Simulation settings copied from the attached notebook.
N_SIM, N_OBSERVATIONS, SEED = 100, 2000, 123
STRONG_RANGE = (0, 0.8)
MEDIUM_RANGE = (0, 0.5)
WEAK_RANGE = (0, 0.3)
CONSTANT_RHO = 0.4

CORRELATION_ESTIMATOR = "sample"  # "sample" or "gaussian"
GAUSSIAN_BANDWIDTH = 5.0
EPSILON = 0.0
FISHER_CLIP_EPS = 1e-7
ZERO_DIAG_FOR_SELECTION = False

# Export settings: two penalties x three representations x three plots.
IMAGE_FORMAT = "png"  # "png" or "svg"; always exactly 18 images.
DPI = 300
VARIANCE_FIGSIZE = (8, 4.5)
WEIGHT_FIGSIZE = (10, 5.5)
TOTAL_WEIGHT_FIGSIZE = (8, 4.5)
X_JITTER = 0.003  # Display only, matching your original plot. Use 0 for exact x positions.
JITTER_SEED = 123
N_LAMBDA = 501  # Linear grid per penalty and representation.
SHOW_PLOTS = False
OUTPUT_DIR = Path.cwd()

N_RANDOM_STARTS = 10
MAX_ITER = 2000
OPT_TOL = 1e-8
ACTIVE_THRESHOLD = 1e-6
