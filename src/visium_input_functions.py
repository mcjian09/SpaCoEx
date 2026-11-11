from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay
SECTION1_SAMPLE_ID = "V1_Breast_Cancer_Block_A_Section_1"
def download_missing(url, destination):
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    print("Downloading", destination.name, flush=True)
    urlretrieve(url, temporary)
    temporary.replace(destination)

def load_visium_counts_and_coordinates(folder):
    """Read the same filtered 10x counts and spot positions without tissue images."""
    folder = Path(folder)
    count_file = folder / "filtered_feature_bc_matrix.h5"
    positions_file = folder / "spatial" / "tissue_positions_list.csv"
    base = ("https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/"
            f"{SECTION1_SAMPLE_ID}/{SECTION1_SAMPLE_ID}")
    download_missing(base + "_filtered_feature_bc_matrix.h5", count_file)
    if not positions_file.exists():
        archive = folder / "spatial.tar.gz"
        download_missing(base + "_spatial.tar.gz", archive)
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()
                       and Path(m.name).name == "tissue_positions_list.csv"]
            if len(members) != 1:
                raise ValueError("Expected one tissue_positions_list.csv in the 10x archive.")
            positions_file.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(members[0]) as source:
                positions_file.write_bytes(source.read())
    with h5py.File(count_file, "r") as handle:
        matrix = handle["matrix"]
        counts = sparse.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=tuple(matrix["shape"][:]),
        ).T.tocsr()
        features = matrix["features"]
        var = pd.DataFrame({"gene_ids": features["id"].asstr()[:],
                            "feature_types": features["feature_type"].asstr()[:]},
                           index=features["name"].asstr()[:])
        if "genome" in features:
            var["genome"] = features["genome"].asstr()[:]
        barcodes = matrix["barcodes"].asstr()[:]
    positions = pd.read_csv(positions_file, header=None, index_col=0,
                           names=["barcode", "in_tissue", "array_row", "array_col",
                                  "pixel_row", "pixel_col"])
    if not positions.index.is_unique or not pd.Index(barcodes).isin(positions.index).all():
        raise ValueError("10x barcodes and spatial coordinates do not align.")
    aligned = positions.loc[barcodes]
    result = ad.AnnData(counts, obs=aligned[["in_tissue", "array_row", "array_col"]].copy(), var=var)
    result.obsm["spatial"] = aligned[["pixel_col", "pixel_row"]].to_numpy(float)
    result = result[:, result.var["feature_types"].eq("Gene Expression")].copy()
    result.var_names_make_unique()
    return result

def is_artifact_gene(g):
    g = str(g).upper()
    return (
        g.startswith("MT-") or
        g.startswith("RPL") or
        g.startswith("RPS") or
        g.startswith("HB") or
        g.startswith("IGH") or
        g.startswith("IGK") or
        g.startswith("IGL") or
        g == "MALAT1"
    )

def delaunay_adjacency(coords):
    coords = np.asarray(coords, dtype=float)
    tri = Delaunay(coords)

    edges = set()
    for simplex in tri.simplices:
        for a, b in [(simplex[0], simplex[1]), (simplex[0], simplex[2]), (simplex[1], simplex[2])]:
            if a != b:
                edges.add((min(a, b), max(a, b)))

    rows, cols = [], []
    for a, b in edges:
        rows.extend([a, b])
        cols.extend([b, a])

    W = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(coords.shape[0], coords.shape[0])
    )
    W.setdiag(0)
    W.eliminate_zeros()
    return W

def row_standardize(W):
    degree = np.asarray(W.sum(axis=1)).ravel()
    inv_degree = np.divide(
        1.0, degree,
        out=np.zeros_like(degree, dtype=float),
        where=degree > 0
    )
    return sparse.diags(inv_degree) @ W

def moran_scores(ad, chunk_size=512):
    X_all = ad.X
    W = row_standardize(delaunay_adjacency(ad.obsm["spatial"]))
    n, p = ad.shape

    scores = np.zeros(p)

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        X_chunk = X_all[:, start:end]

        if sparse.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        else:
            X_chunk = np.asarray(X_chunk)

        Z = X_chunk.astype(float)
        Z -= Z.mean(axis=0, keepdims=True)

        denom = np.sum(Z ** 2, axis=0)
        WZ = W @ Z
        numer = np.sum(Z * WZ, axis=0)

        scores[start:end] = np.divide(
            numer, denom,
            out=np.zeros_like(numer, dtype=float),
            where=denom > 0
        )

    return scores
