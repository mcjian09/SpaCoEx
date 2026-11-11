import numpy as np
from scipy.spatial import cKDTree
SECTION='M3-2'
N_CELLS=1200
MIN_SIDE_FRACTION=0.20

def crop_from_section(sec):
    rows, xy, dist, inside = (sec[name] for name in ("source_row","xy","dist","inside"))
    need = int(np.ceil(MIN_SIDE_FRACTION * N_CELLS - 1e-10))
    if N_CELLS > len(rows) or min(inside.sum(), (~inside).sum()) < need or 2*need > N_CELLS:
        raise ValueError(f"{SECTION} has {len(rows):,} cells ({inside.sum():,} inside, "
                         f"{(~inside).sum():,} outside); requested {N_CELLS:,}, at least {need:,} each side.")
    tree = cKDTree(xy)
    for center in np.lexsort((rows, np.where(np.isfinite(dist), dist, np.inf))):
        if not np.isfinite(dist[center]): break
        distances, members = tree.query(xy[center], k=N_CELLS)
        n_inside = int(inside[members].sum())
        if min(n_inside, N_CELLS-n_inside) >= need:
            members = np.sort(members)
            return dict(ids=sec["ids"][members], source_row=rows[members], genes=sec["genes"],
                counts=sec["counts"][members].astype(float), xy=xy[members],
                inside=inside[members], center=xy[center],
                center_source_row=rows[center], radius=distances.max())
    raise ValueError(f"No {N_CELLS:,}-cell window in {SECTION} met the {MIN_SIDE_FRACTION:.0%} balance. "
                     "Try fewer cells or explicitly lower MIN_SIDE_FRACTION.")
