"""One native SpatialCorr permutation P-value per eight-gene panel."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spatialcorr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FILE = ROOT.parents[1] / "data/visium/section1_moran200.h5ad"
DATA_HASH = "3f328e503b3081ece2f05f8d5c88ad4feaa0aa41de051f6b762deed21eeca8fe"
RESULTS = ROOT.parents[1] / "outputs/BR_panel_results"
CHOSEN = "COX6C FTL APOE COL1A1 IGFBP5 COL1A2 FN1 KRT19".split()
FIRST = "MUC1 MCCD1 PBX1 WFDC2 HK2 TYMP COMP LAPTM5".split()


def prepare(n):
    if n < 1 or hashlib.sha256(FILE.read_bytes()).hexdigest() != DATA_HASH:
        raise ValueError("Use the bundled 200-candidate data and request at least one random panel.")
    if version("spatialcorr") != "1.2.0":
        raise ValueError("Use SpatialCorr 1.2.0 to match the original tests.")
    a = ad.read_h5ad(FILE)
    pool = a.var_names.to_numpy()
    if a.shape != (3798, 200) or not a.var_names.is_unique:
        raise ValueError("Unexpected input dimensions.")
    rng = np.random.default_rng(42)
    panels = [CHOSEN] + [pool[np.sort(rng.choice(200, size=8, replace=False))].tolist() for _ in range(n)]
    if panels[1] != FIRST:
        raise ValueError("Panel order differs from the laptop.")
    fine = a.obs["ground_truth"].astype(str)
    a.obs["region"] = np.where(fine.str.startswith("Healthy_"), "Non-cancer", "Cancer")
    assert a.obs["region"].value_counts().to_dict() == {"Cancer":3313, "Non-cancer":485}
    return a, panels


def test(a, genes, K, max_perms=1000):
    np.random.seed(1)  # same permutation schedule as the two completed laptop tests
    with threadpool_limits(limits=1), np.errstate(invalid="raise", divide="raise", over="raise"):
        p, details = spatialcorr.run_test(
            a[:, genes].copy(), genes, bandwidth=5.006316010800757,
            cond_key="region", row_key="array_row", col_key="array_col", contrib_thresh=10,
            run_br=True, standardize_var=True, precomputed_kernel=K, n_procs=1,
            compute_spotwise_pvals=False, mc_pvals=False, max_perms=max_perms, verbose=0,
        )
    score = float(details["observed_log_likelihood_ratio"])
    nulls = np.asarray(details["permuted_log_likelihood_ratios"])
    if not (0 < p <= 1 and np.isfinite(score) and nulls.size and np.isfinite(nulls).all()):
        raise ValueError("SpatialCorr returned a nonfinite statistic or invalid P-value.")
    return score, float(p)


def row(i, genes, score, p, source):
    return dict(panel=i, panel_type="SpaCoEx" if i == 0 else "random",
                genes=", ".join(genes), BR_score=score, BR_p=p, source=source)


def get_row(i, genes):
    # These are the user's two completed full tests, not P-values inferred from scores.
    if i in (0, 1):
        score, p = [(200.112, .001), (-80.454, 1.)][i]
        return row(i, genes, score, p, "reported laptop test (score rounded)")
    path = RESULTS / f"panel_{i:04d}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if (record["genes"] != ", ".join(genes) or record["panel"] != i or
                not 0 < record["BR_p"] <= 1 or not np.isfinite(record["BR_score"])):
            raise ValueError(f"Invalid cached result: {path}")
        return record


def collect(panels):
    rows = [get_row(i, genes) for i, genes in enumerate(panels)]
    missing = [i for i, record in enumerate(rows) if record is None]
    if missing:
        raise SystemExit(f"Comparison not complete: missing {len(missing)} panel(s), first: {missing[:10]}.")
    table = pd.DataFrame(rows)
    chosen = table.loc[0, "BR_p"]
    random = table.loc[table.panel_type == "random", "BR_p"]
    table["p_minus_SpaCoEx_p"] = table["BR_p"] - chosen
    dest = RESULTS / f"individual_BR_pvalues_{len(random)}_random.csv"
    table.to_csv(dest, index=False)
    print(f"SpaCoEx P = {chosen:g}; {len(random)} random panels.")
    for label, count in [("smaller", (random < chosen).sum()), ("equal", (random == chosen).sum()),
                         ("larger", (random > chosen).sum()), ("at most SpaCoEx P", (random <= chosen).sum())]:
        print(f"Random P {label}: {count}/{len(random)}")
    print(dest)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=99)
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--only", type=int, metavar="PANEL_ID")
    action.add_argument("--collect", action="store_true")
    action.add_argument("--check", action="store_true")
    args = ap.parse_args()
    a, panels = prepare(args.n)
    if args.only is not None and not 0 <= args.only <= args.n:
        ap.error("--only must be in 0..--n")
    RESULTS.mkdir(exist_ok=True)
    if args.collect:
        collect(panels)
        return
    if args.check:
        print(f"{len(panels)} panels; {sum(get_row(i,g) is not None for i,g in enumerate(panels))} already have P-values.")
        print("Random panel 1:", ", ".join(panels[1]))
        return
    K = None
    for i in range(len(panels)) if args.only is None else [args.only]:
        genes = panels[i]
        if get_row(i, genes) is not None:
            continue
        if K is None:
            K = spatialcorr.compute_kernel_matrix(a.obs, bandwidth=5.006316010800757, region_key="region",
                     condition_on_region=True, y_col="array_row", x_col="array_col")
            if (K.sum(axis=1) >= 10).sum() != 3767:
                raise ValueError("Spot filter differs from the laptop test.")
        print(f"Testing random panel {i}/{args.n}: {', '.join(genes)}", flush=True)
        score, p = test(a, genes, K)
        dest = RESULTS / f"panel_{i:04d}.json"
        temp = dest.with_suffix(".tmp")
        temp.write_text(json.dumps(row(i, genes, score, p, "full SpatialCorr BR test"), allow_nan=False))
        temp.replace(dest)
        print(f"Panel {i}: P = {p:g} (saved)", flush=True)
    if args.only is None:
        collect(panels)


if __name__ == "__main__":
    main()
