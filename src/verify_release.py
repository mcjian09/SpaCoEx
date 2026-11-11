"""Verify portability, inputs, reference numbers and small fresh computations."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import platform
import runpy
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import paper_reproduce as paper

ROOT = paper.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="Also rebuild raw inputs and run native-checked BR smoke tests")
    parser.add_argument("--record-environment", action="store_true")
    parser.add_argument("--check-fresh-outputs", action="store_true", help="Check previously executed downstream outputs, not just bundled references")
    args = parser.parse_args()
    start = time.perf_counter()
    checks = {}
    for file in ROOT.rglob("*.py"):
        if "outputs" not in file.relative_to(ROOT).parts:
            ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    for file in (ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(file.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert not cell["outputs"] and cell["execution_count"] is None
                ast.parse("".join(cell["source"]))
    checks["python_and_clean_notebook_syntax"] = True
    engine = paper.shared._engine()
    checks["master_source_checksums"] = True
    for name in ("visium", "merfish_25um"):
        run = paper.reference_run(name)
        m = paper.manifest(run)
        x, xy, genes = engine.load_inputs(paper.input_path(run), m["preprocessing"])
        weights = pd.read_csv(run / "weights.csv", index_col="lambda")
        fits = pd.read_csv(run / "lambda_path.csv", index_col="lambda")
        assert list(weights.columns) == list(genes) and fits.index.equals(weights.index)
        assert fits.converged.all() and np.isfinite(x).all() and np.isfinite(xy).all()
        np.testing.assert_array_equal(fits.active_genes, (weights.to_numpy()>m["active_threshold"]).sum(axis=1))
        with np.load(run / "association.npz", allow_pickle=False) as arrays:
            S = arrays["S"]
            active = weights.to_numpy()>m["active_threshold"]
            retained = 100*np.sum((active @ S)*active, axis=1)/S.sum()
            np.testing.assert_allclose(retained, fits.percent_retained, rtol=1e-12, atol=1e-10)
        checks[f"{name}_reference_alignment_and_retention"] = True
    expected = dict(zip(paper.METHODS, [372, 139, 160, 18, 0]))
    for name in paper.METHODS:
        t = pd.read_csv(paper.REFERENCE / "br" / f"{paper.SLUGS[name]}.csv")
        assert len(t) == 861
        assert (t.BR_q_BH<.05).sum() == expected[name]
        np.testing.assert_allclose(t.BR_p, (t.shuffles_above+1)/(t.n_shuffles+1), atol=1e-14)
        np.testing.assert_allclose(t.BR_q_BH, multipletests(t.BR_p, method="fdr_bh")[1], atol=1e-14)
    checks["BR_reference_counts_BH_and_permutation_arithmetic"] = expected
    metrics = pd.read_csv(paper.REFERENCE / "visium_clustering/results.csv")
    actual = metrics.loc[metrics.method.eq("SpaCoEx") & metrics.alpha.eq(.5)].iloc[0]
    np.testing.assert_allclose([actual.ARI, actual.NMI, actual.PAS], [.490, .660, .112], atol=.0005)
    checks["paper_Table1_alpha_half"] = True
    mp = paper.manifest(paper.reference_run("merfish_25um"))
    np.testing.assert_allclose(mp["bandwidth"], 25.0, rtol=1e-12)
    assert not (paper.REFERENCE / "merfish_paper").exists()
    checks["MERFISH_only_requested_25um_reference"] = True
    import simulation_paper as sim
    for pair in (False, True):
        w, status, _ = sim.ascend_one(np.eye(2), 0., pair, np.array([.3, .6]), .25, 1e-8, 100)
        assert status == 1 and not w.any()
    checks["simulation_strict_below_one_zero_cutoff"] = True
    if args.extended:
        paper.prepare_visium()
        checks["raw_Visium_candidate_snapshot_recreated"] = True
        paper.prepare_merfish()
        checks["MERFISH_ROI_and_candidates_recreated"] = True
        paper.select_panels(run_sparkx=False)
        checks["Moran_HVG_SpaCoEx_random_panels_recreated_SPARKX_reference_only"] = True
        test = runpy.run_path(str(paper.CODE / "spatial_tests/pairwise_br.py"))["pairwise_br"]
        smoke = test(genes=paper.reference_panels()["SpaCoEx"][:3], n_shuffles=3, seed=1, cancer_mode="strict")
        assert len(smoke) == 3
        checks["fresh_BR_three_pairs_three_permutations_native_checked"] = True
        paper.run_br(reference=True)
        paper.sensitivity(mode="reference")
        checks["reference_heatmap_and_sensitivity_figures_redrawn"] = True
    if args.check_fresh_outputs:
        fresh = pd.read_csv(paper.OUTPUT / "visium_clustering/all_methods.csv").rename(columns={"coexpression_alpha":"alpha"})
        reference = pd.read_csv(paper.REFERENCE / "visium_clustering/results.csv")
        key, fields = ["method", "alpha"], ["ARI", "NMI", "PAS"]
        fresh = fresh.loc[fresh.method.isin(paper.METHODS[:-1]) & fresh.alpha.notna()].set_index(key).sort_index()
        reference = reference.set_index(key).sort_index()
        np.testing.assert_allclose(fresh[fields], reference[fields], atol=1e-12, rtol=1e-12)
        checks["fresh_four_method_seven_alpha_clustering_matches_reference"] = True
        fresh = pd.read_csv(paper.OUTPUT / "merfish_25um/domains/agreement.csv").set_index("panel")
        reference = pd.read_csv(paper.REFERENCE / "merfish_25um/domains/agreement.csv").set_index("panel")
        np.testing.assert_allclose(fresh[fields], reference[fields], atol=1e-12, rtol=1e-12)
        checks["fresh_25um_MERFISH_clustering_matches_reference"] = True
        fresh = pd.read_csv(paper.OUTPUT / "conley/paper_conventions.csv").set_index("outcome")
        reference = pd.read_csv(paper.REFERENCE / "conley/validation_against_conleyreg.csv")
        reference = reference.loc[reference.cutoff_px.eq(2000)].set_index("outcome").loc[fresh.index]
        np.testing.assert_allclose(fresh[["difference", "conleyreg_SE", "conleyreg_P"]],
                                   reference[["r_difference", "r_package_SE", "r_package_P"]], rtol=1e-8, atol=1e-12)
        checks["fresh_Conley_matches_saved_independent_R_validation"] = True
        simulation = json.loads((paper.OUTPUT / "simulation/verification.json").read_text())
        assert len(simulation) == 6 and all(r["unresolved"] == 0 for r in simulation)
        checks["fresh_full_original_simulation_six_paths"] = simulation
    checks["complete_999_permutation_suite_rerun_in_this_verification"] = False
    checks["all_sensitivity_fits_rerun_in_this_verification"] = False
    checks["elapsed_seconds"] = time.perf_counter()-start
    paper.save_json(paper.OUTPUT / "release_verification.json", checks)
    if args.record_environment:
        names = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "scanpy", "anndata",
                 "spatialcorr", "statsmodels", "h5py", "threadpoolctl", "numba", "nbformat",
                 "nbclient", "requests", "seaborn", "ipykernel", "jupyterlab", "setuptools"]
        versions = {name:metadata.version(name) for name in names}
        paper.save_json(ROOT / "docs/environment_recorded.json", dict(python=platform.python_version(),
            platform=platform.system(), versions=versions))
        (ROOT / "requirements.txt").write_text(
            "# Reference environment; SpatialCorr is installed separately (see README).\n" +
            "\n".join(f"{k}=={v}" for k,v in versions.items() if k != "spatialcorr") + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
