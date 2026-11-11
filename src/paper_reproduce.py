"""Portable entry points for the November 11 paper; original numerical kernels unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay, cKDTree
from threadpoolctl import threadpool_limits

import spacoex_shared as shared
import visium_42_gene_ablation_helpers as ablation
import visium_internal_domain_grid as domains
from visium_spatialcorr_domains import choose_panel

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUTPUT = ROOT / "outputs"
REFERENCE = ROOT / "reference"
METHODS = ["SpaCoEx", "Moran's I", "HVG", "SPARK-X", "Random"]
SLUGS = {"SpaCoEx":"SpaCoEx", "Moran's I":"Morans_I", "HVG":"HVG", "SPARK-X":"SPARK-X", "Random":"Random"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def reference_run(kind):
    return REFERENCE / kind / "run"


def current_run(kind):
    pointer = OUTPUT / kind / "latest.json"
    return ROOT / json.loads(pointer.read_text())["run"] if pointer.exists() else reference_run(kind)


def manifest(run):
    return json.loads((Path(run) / "manifest.json").read_text())


def input_path(run):
    run = Path(run)
    expected = manifest(run)["input_sha256"]
    for p in (run.parent / "input.npz", run.parents[1] / "input.npz"):
        if p.is_file() and shared._engine().sha256(p) == expected:
            return p
    raise FileNotFoundError("Matching input.npz not found alongside this run.")


def study(run=None):
    return ablation.load_study(CODE, run or current_run("visium"), OUTPUT / "input_audit")


def prepare_visium():
    """Rebuild the candidate snapshot from the supplied original 10x counts."""
    import visium_input_functions as f
    data = ROOT / "data/visium"
    counts_path = data / f.SECTION1_SAMPLE_ID / "filtered_feature_bc_matrix.h5"
    provenance = json.loads((ROOT / "docs/source_provenance.json").read_text())
    prefix = counts_path.relative_to(ROOT).as_posix() + ".part"
    entries = sorted((r for r in provenance["copied_files"] if r["file"].startswith(prefix)), key=lambda r:r["file"])
    if len(entries) != 2:
        raise ValueError("Expected the two raw-count transport parts.")
    if not counts_path.exists():
        pieces = [(ROOT / r["file"]).read_bytes() for r in entries]
        if any(hashlib.sha256(piece).hexdigest() != r["sha256"] for piece, r in zip(pieces, entries)):
            raise ValueError("A raw-count transport part is damaged.")
        counts_path.write_bytes(b"".join(pieces))
    if hashlib.sha256(counts_path.read_bytes()).hexdigest() != entries[0]["original_sha256"]:
        raise ValueError("Raw Visium counts do not match the supplied source checksum.")
    a = f.load_visium_counts_and_coordinates(data / f.SECTION1_SAMPLE_ID)
    metadata = pd.read_csv(data / "BRCA1_metadata.tsv", sep="\t").set_index("ID", verify_integrity=True)
    a.obs = a.obs.join(metadata[["fine_annot_type"]], how="left")
    a = a[a.obs.fine_annot_type.notna()].copy()
    a.obs["ground_truth"] = a.obs.fine_annot_type.astype(str)
    assert a.n_obs == 3798 and a.obs.ground_truth.nunique() == 20
    detected = np.asarray((a.X > 0).sum(axis=0)).ravel()
    a = a[:, detected >= 3].copy()
    detected = detected[detected >= 3]
    totals = np.asarray(a.X.sum(axis=1)).ravel().astype(float)
    normalized = a.copy()
    normalized.X = (sparse.diags(10000 / totals) @ a.X).tocsr()
    normalized.X.data = np.log1p(normalized.X.data)
    eligible = (detected >= 1899) & np.array([not f.is_artifact_gene(g) for g in a.var_names])
    work = normalized[:, eligible]
    scores = pd.DataFrame(dict(gene=work.var_names.to_numpy(str), detected_spots=detected[eligible],
        detected_percent=100*detected[eligible]/len(a), moran_I=f.moran_scores(work)))
    scores = scores.sort_values(["moran_I", "gene"], ascending=[False, True], ignore_index=True)
    scores.insert(0, "moran_rank", np.arange(1, len(scores)+1))
    genes = scores.gene.head(200).to_numpy(str)
    payload = dict(X=a[:, genes].X.toarray().astype(float), xy=a.obs[["array_col", "array_row"]].to_numpy(float),
                   genes=genes, expression_ids=a.obs_names.to_numpy(str), coordinate_ids=a.obs_names.to_numpy(str),
                   library_total=totals)
    with np.load(REFERENCE / "visium/input.npz", allow_pickle=False) as ref:
        for key, value in payload.items():
            np.testing.assert_array_equal(value, ref[key], err_msg=f"Raw preparation differs: {key}")
    dest = OUTPUT / "prepared_visium"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "input.npz", **payload)
    scores.to_csv(dest / "eligible_genes_moran_ranking.csv", index=False)
    a[:, genes].copy().write_h5ad(dest / "raw_candidates.h5ad")
    print("Verified raw preparation: 3,798 spots, 200 candidates, full pre-screen library totals.")
    return a, normalized, payload


def _fit(kind, bandwidth, ridge=1e-6):
    """Run the registered master with an explicit bandwidth; never alter its files."""
    ref = REFERENCE / kind
    request = OUTPUT / kind / "request"
    request.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ref / "input.npz", request / "input.npz")
    shutil.copy2(ref / "run.json", request / "run.json")
    engine = shared._engine()
    m = manifest(ref / "run")
    x, xy, _ = engine.load_inputs(request / "input.npz", m["preprocessing"])
    spacing = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    engine.SPACINGS, engine.RIDGE = float(bandwidth / spacing), float(ridge)
    out, path, _, fig = engine.analyze(request / "run.json", matrix="logeuclidean", selection="gradient")
    plt.close(fig)
    np.testing.assert_allclose(manifest(out)["bandwidth"], bandwidth, rtol=1e-12)
    if not path.converged.all():
        raise RuntimeError(f"Some fits did not converge. Inspect {out}; not recording this as the current run.")
    if kind == "visium":
        for name in ("spot_metadata.csv", "eligible_genes_moran_ranking.csv"):
            shutil.copy2(ref / "run" / name, out / name)
    save_json(OUTPUT / kind / "latest.json", {"run":out.relative_to(ROOT).as_posix()})
    plot_paths(out, OUTPUT / kind)
    return out


def fit_visium():
    # Preserve the setting that generated Table 1; the sensitivity h=5 is exact.
    return _fit("visium", manifest(reference_run("visium"))["bandwidth"])


def plot_paths(run, dest):
    run, dest = Path(run), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = pd.read_csv(run / "lambda_path.csv")
    w = pd.read_csv(run / "weights.csv", index_col="lambda").to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    axes[0].plot(np.sqrt(path["lambda"]), w, lw=1, alpha=.65)
    axes[0].set(xlabel=r"$\sqrt{\lambda}$", ylabel="Gene weight", ylim=(-.02, 1.02))
    axes[1].step(np.sqrt(path["lambda"]), path.active_genes, where="post", lw=2)
    axes[1].set(xlabel=r"$\sqrt{\lambda}$", ylabel="Number selected")
    points = path.sort_values(["active_genes", "percent_retained"]).drop_duplicates(["active_genes", "percent_retained"])
    axes[2].plot(points.active_genes, points.percent_retained, "o-", ms=4)
    axes[2].set(xlabel="Number genes", ylabel="Variance explained", ylim=(0, 100))
    from matplotlib.ticker import PercentFormatter
    axes[2].yaxis.set_major_formatter(PercentFormatter(100))
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"selection_paths.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def reference_panels():
    selected = pd.read_csv(REFERENCE / "visium_clustering/selected_genes.csv")
    panels = {m:selected.loc[selected.method.eq(m)].sort_values("display_order").gene.tolist() for m in METHODS[:-1]}
    key = pd.read_csv(REFERENCE / "br/panel_key.csv")
    # Retain the original within-tie display order, not a new invented ranking.
    panels["SpaCoEx"], panels["Random"] = key.top_gene.tolist(), key.random_gene.tolist()
    assert all(len(v) == len(set(v)) == 42 for v in panels.values())
    return panels


def select_panels(run_sparkx=True, install_r=False, rscript=None):
    s = study()
    panels, ranks, _ = ablation.python_panels(s, size=42, n_random=1)
    selected = {m:panels[m] for m in METHODS[:3]}
    if run_sparkx:
        exe = ablation.setup_sparkx(OUTPUT, OUTPUT / "r_library", rscript, install_r,
                                   "a8b4bf27b804604dfda53da42992f100b8e4e727")
        selected["SPARK-X"], _, _ = ablation.sparkx_panel(s, 42, OUTPUT / "r_library", exe, force=True)
    else:
        scores = pd.read_csv(REFERENCE / "sparkx/sparkx_scores.csv")
        selected["SPARK-X"] = scores.sort_values(["combinedPval", "gene"]).gene.head(42).tolist()
        print("Using reference SPARK-X scores (not a fresh SPARK-X run).")
    ranks = pd.read_csv(REFERENCE / "visium/panel_ranks.csv").sort_values(["master_gradient", "gene"])
    pool = ranks.loc[ranks.master_gradient.eq(ranks.master_gradient.max()), "gene"].tolist()
    selected["Random"] = np.random.default_rng(42).choice(pool, 42, replace=False).tolist()
    reference = reference_panels()
    for method in METHODS:
        if set(selected[method]) != set(reference[method]):
            raise ValueError(f"{method} panel differs from the paper reference; inspect before continuing.")
    dest = OUTPUT / "panels"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reference).to_csv(dest / "selected_genes.csv", index=False)
    print("All five panels reproduce the reference gene sets.")
    return reference


def cluster_visium():
    s, panels = study(), reference_panels()
    all_scores = []
    for method in METHODS:
        print("Clustering", method, flush=True)
        # Use the original panel order for PCA, including within-tie alphabetic order.
        original = pd.read_csv(REFERENCE / "visium_clustering/selected_genes.csv")
        names = (original.loc[original.method.eq(method)].sort_values("display_order").gene.tolist()
                 if method != "Random" else panels[method])
        indices = pd.Index(s["genes"]).get_indexer(names)
        dest = OUTPUT / "visium_clustering" / SLUGS[method]
        panel = pd.DataFrame({"gene":names})
        fig, metrics, labels = domains.matched_domain_grid(s["x"][:, indices], s["xy"], s["manifest"],
            s["engine"].math.transform_correlation_matrices, panel, s["spots"].ground_truth, dest, seed=0)
        plt.close(fig)
        all_scores.append(metrics.assign(method=method))
    result = pd.concat(all_scores, ignore_index=True)
    result.to_csv(OUTPUT / "visium_clustering/all_methods.csv", index=False)
    print(result[["method", "coexpression_alpha", "ARI", "NMI", "PAS"]].to_string(index=False))
    return result


def plot_br(panels, results, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2), layout="constrained")
    for ax, method in zip(axes, METHODS):
        names, table = pd.Index(panels[method]), results[method]
        a, b = names.get_indexer(table.gene_a), names.get_indexer(table.gene_b)
        assert min(a.min(), b.min()) >= 0 and len(table) == 861
        values = np.full((42, 42), np.nan)
        values[a, b] = values[b, a] = -np.log10(table.BR_q_BH)
        image = ax.imshow(values, vmin=0, vmax=3, cmap=plt.get_cmap("Blues").with_extremes(bad="#e5e5e5"), interpolation="nearest")
        title = "Random 42 from the\nlowest-ranked group" if method == "Random" else f"{method} top 42"
        ax.set(title=title, xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(image, ax=axes.tolist(), shrink=.8, pad=.01)
    bar.set_label(r"$-\log_{10}$(BH-adjusted BR P-value)")
    bar.set_ticks([0, 1, -np.log10(.05), 2, 3], labels=["0", "1", "1.30", "2", "3"])
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"five_panel_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_br(n_permutations=999, reference=False):
    panels = reference_panels()
    dest = OUTPUT / "br" / ("reference_redraw" if reference else f"fresh_B{n_permutations}")
    dest.mkdir(parents=True, exist_ok=True)
    results = {}
    if not reference:
        test = runpy.run_path(str(CODE / "spatial_tests/pairwise_br.py"))["pairwise_br"]
    for method in METHODS:
        print("BR:", method, flush=True)
        results[method] = (pd.read_csv(REFERENCE / "br" / f"{SLUGS[method]}.csv") if reference else
                           test(genes=panels[method], n_shuffles=n_permutations, seed=1, cancer_mode="strict"))
        results[method].to_csv(dest / f"{SLUGS[method]}.csv", index=False)
    plot_br(panels, results, dest)
    summary = pd.DataFrame([dict(method=m, pairs=len(t), raw_significant=int((t.BR_p<.05).sum()),
        BH_significant=int((t.BR_q_BH<.05).sum())) for m, t in results.items()])
    summary.to_csv(dest / "significant_pairs.csv", index=False)
    return summary


def run_conley():
    import anndata as ad
    from scipy.stats import t as student_t
    original = runpy.run_path(str(CODE / "spatial_tests/breast_pair_spatialcorr_conley.py"))
    a = ad.read_h5ad(ROOT / "data/visium/section1_moran200.h5ad")
    h = manifest(reference_run("visium"))["bandwidth"]
    selected, table = original["run_pair"](a, h, OUTPUT / "conley")
    # Match conleyreg's HC0 and Student-t conventions, not the Python helper's HC1/normal p-values.
    n = selected.cancer_n + selected.noncancer_n
    selected["conleyreg_SE"] = selected.spatial_SE * np.sqrt((n-2)/n)
    selected["conleyreg_P"] = 2*student_t.sf(abs(selected.difference/selected.conleyreg_SE), n-2)
    selected.to_csv(OUTPUT / "conley/paper_conventions.csv", index=False)
    values = pd.read_csv(OUTPUT / "conley/B2M_HLA-A_SpatialCorr_Conley_log1p/spot_outcomes.csv", index_col="spot")
    raw = np.load(REFERENCE / "visium/input.npz", allow_pickle=False)
    indices = pd.Index(raw["expression_ids"].astype(str)).get_indexer(values.index)
    counts = raw["X"][indices][:, pd.Index(raw["genes"].astype(str)).get_indexer(["B2M", "HLA-A"])]
    xy = a.obs.loc[values.index, ["array_col", "array_row"]].to_numpy(float)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    axes[0].scatter(*xy.T, s=5, c=np.where(values.cancer, "#d62728", "#2878B5"))
    axes[0].set_title("Cancer / non-cancer")
    im = axes[1].scatter(*xy.T, s=5, c=values.iloc[:, 0], cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[1], shrink=.7)
    axes[1].set_title("B2M / HLA-A local correlation")
    for j, gene in enumerate(["B2M", "HLA-A"]):
        im = axes[j+2].scatter(*xy.T, s=5, c=counts[:, j], cmap="viridis")
        axes[j+2].set_title(f"{gene} raw UMI counts")
        fig.colorbar(im, ax=axes[j+2], shrink=.7)
    for ax in axes:
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / "conley" / f"B2M_HLA_A_maps.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selected


def run_conley_r(rscript=None):
    exe = ablation.find_rscript(rscript)
    folder = OUTPUT / "conley/B2M_HLA-A_SpatialCorr_Conley_log1p"
    if not (folder / "spot_outcomes.csv").exists():
        run_conley()
    subprocess.run([exe, "--vanilla", str(CODE / "spatial_tests/validate_conleyreg.R"), str(folder)], check=True)


def prepare_merfish():
    from merfish_window import crop_from_section
    with np.load(ROOT / "data/merfish/section_M3-2.npz", allow_pickle=False) as sec:
        crop = crop_from_section(sec)
    with np.load(ROOT / "data/merfish/roi_v1_file55130807.npz", allow_pickle=False) as ref:
        for key in ("ids", "source_row", "genes", "counts", "xy", "inside"):
            np.testing.assert_array_equal(crop[key], ref[key], err_msg=f"ROI differs: {key}")
    assert len(crop["ids"]) == 1200 and crop["inside"].sum() == 925
    names = pd.Index(crop["genes"].astype(str))
    control = names.str.match(r"(?i)^(?:blank|negctrl|negative.control|unused|intbc)")
    names, z = names[~control], crop["counts"][:, ~control]
    totals = z.sum(axis=1)
    x = np.log1p(10000*z/totals[:, None])
    ptr, cols = Delaunay(crop["xy"]).vertex_neighbor_vertices
    W = sparse.csr_matrix((np.ones(len(cols)), cols, ptr), shape=(1200, 1200))
    W = sparse.diags(1/np.asarray(W.sum(axis=1)).ravel()) @ W
    centered = x-x.mean(axis=0)
    den = (centered**2).sum(axis=0)
    moran = np.divide((centered*(W @ centered)).sum(axis=0), den, out=np.full_like(den, np.nan), where=den>0)
    scores = pd.DataFrame(dict(gene=names, detected=(z>0).mean(axis=0), moran_I=moran))
    scores = scores.loc[(scores.detected>=.05)&np.isfinite(scores.moran_I)].sort_values(["moran_I", "gene"], ascending=[False, True])
    genes = scores.gene.head(200).to_numpy(str)
    assert len(genes) == 136
    with np.load(REFERENCE / "merfish_25um/input.npz", allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference["genes"], genes)
        np.testing.assert_array_equal(reference["X"], z[:, names.get_indexer(genes)])
        np.testing.assert_array_equal(reference["library_total"], totals)
        np.testing.assert_array_equal(reference["xy"], crop["xy"])
    print("Verified M3-2 window: 1,200 cells, 925 inside / 275 outside; 136 candidates.")
    return crop


def fit_merfish():
    return _fit("merfish_25um", 25.0)


def cluster_merfish():
    run = current_run("merfish_25um")
    engine, m = shared._engine(), manifest(run)
    np.testing.assert_allclose(m["bandwidth"], 25.0)
    x, xy, genes = engine.load_inputs(input_path(run), m["preprocessing"])
    with np.load(input_path(run), allow_pickle=False) as raw:
        ids = pd.Index(raw["expression_ids"].astype(str), name="cell")
    reference = pd.read_csv(ROOT / "data/merfish/roi_cells.csv", index_col="cell").loc[ids]
    eligible, _ = choose_panel(run, m, genes, 23)
    eligible["moran_rank"] = pd.Index(genes).get_indexer(eligible.gene)+1
    eligible = eligible.sort_values(["rank", "moran_rank"], ignore_index=True)
    eligible["selected"] = eligible.index < 23
    panel = eligible.loc[eligible.selected].copy()
    dest = OUTPUT / "merfish_25um/domains"
    fig, metrics, _ = domains.matched_domain_grid(x[:, pd.Index(genes).get_indexer(panel.gene)], xy,
        m, engine.math.transform_correlation_matrices, panel, reference.reference, dest, seed=0)
    plt.close(fig)
    eligible.to_csv(dest / "panel_with_cutoff_ties.csv", index=False)
    return metrics


def sensitivity(mode="fresh"):
    from matplotlib.ticker import MaxNLocator
    if mode not in ("fresh", "reference"):
        raise ValueError("mode must be fresh or reference")
    dest = OUTPUT / "sensitivity" / mode
    dest.mkdir(parents=True, exist_ok=True)
    x, xy, genes = shared._engine().load_inputs(REFERENCE / "visium/input.npz", "cp10k_log1p")
    spacing = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    settings = [(h, 1e-6) for h in (2, 3, 5, 7, 10)] + [(5, float(10.**e)) for e in range(-10, -1)]
    paths = {}
    for h, ridge in dict.fromkeys(settings):
        stem = f"h_{h:g}_ridge_{ridge:.0e}"
        if mode == "reference":
            paths[h, ridge] = pd.read_csv(REFERENCE / "sensitivity" / f"{stem}.csv")
            continue
        engine = shared._engine()
        engine.SPACINGS, engine.RIDGE = h/spacing, ridge
        print(f"Sensitivity: h={h}, ridge={ridge:g}", flush=True)
        with threadpool_limits(limits=2):
            S, actual_h, _ = engine.variation(x, xy, "logeuclidean")
            w, path, _, _ = engine.select(S, genes, "gradient", np.linspace(0, 2.02*S.max(), 401))
        path.to_csv(dest / f"{stem}.csv", index=False)
        np.savez_compressed(dest / f"{stem}.npz", weights=w, S=S, lambdas=path["lambda"], genes=genes)
        if not path.converged.all():
            raise RuntimeError(f"Unconverged sensitivity fit: {stem}")
        paths[h, ridge] = path
    for name, choices in (("bandwidth", [(h, 1e-6, f"Bandwidth = {h}") for h in (2, 3, 5, 7, 10)]),
                          ("ridge", [(5, float(10.**e), rf"$\epsilon=10^{{{e}}}$") for e in range(-10, -1)])):
        with plt.rc_context({"font.size":21}):
            fig, axes = plt.subplots(int(np.ceil(len(choices)/3)), 3, figsize=(15, 4.5*np.ceil(len(choices)/3)), squeeze=False, layout="constrained")
            for ax, (h, r, label) in zip(axes.flat, choices):
                path = paths[h, r]
                ax.step(np.sqrt(path["lambda"]), path.active_genes, where="post", lw=2)
                ax.set(title=label, ylim=(0, 205), yticks=[0, 100, 200])
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(choices):]:
                ax.set_visible(False)
            fig.supxlabel(r"$\sqrt{\lambda}$")
            fig.supylabel("Number selected")
            for ext in ("png", "pdf"):
                fig.savefig(dest / f"{name}_paths.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=["prepare-visium", "fit-visium", "panels", "cluster-visium", "br", "conley", "prepare-merfish", "fit-merfish", "cluster-merfish", "sensitivity"])
    parser.add_argument("--reference", action="store_true", help="Redraw saved BR/sensitivity results; do not claim this is a new test")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--install-r", action="store_true")
    args = parser.parse_args()
    if args.analysis == "br":
        print(run_br(args.permutations, args.reference))
    elif args.analysis == "sensitivity":
        sensitivity("reference" if args.reference else "fresh")
    elif args.analysis == "panels":
        select_panels(run_sparkx=not args.reference, install_r=args.install_r)
    else:
        globals()[args.analysis.replace("-", "_")]()


if __name__ == "__main__":
    main()
