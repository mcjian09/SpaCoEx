"""Shared entry point for dataset notebooks. Loading this file does not run an analysis."""
import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path
from types import ModuleType as _ModuleType
from builtins import __import__ as _import
import builtins as _builtins


_REGISTRY = _Path(__file__).resolve().with_name("SpaCoEx_current.json")


def _engine():
    current = _json.loads(_REGISTRY.read_text(encoding="utf-8"))
    root = _REGISTRY.parent / current["directory"]
    modules = {}

    # Keep the legacy absolute import private: a dataset's own "frozen_math"
    # module, sys.path, globals and working directory must not affect a fit.
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return _import(name, globals, locals, fromlist, level)

    for name in ("frozen_math", "analysis"):
        path = root / (name + ".py")
        raw = path.read_bytes()
        if _hashlib.sha256(raw).hexdigest() != current["sha256"][path.name]:
            raise RuntimeError(f"{path} differs from the registered release. Check the shared installation.")
        module = _ModuleType(name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = {**vars(_builtins), "__import__": local_import}
        exec(compile(raw, str(path), "exec"), module.__dict__)
        modules[name] = module
    engine = modules["analysis"]
    if engine.VERSION != current["release"]:
        raise RuntimeError("Registered version does not match the shared method.")
    return engine


def run_spacoex(config_path, *, selection=None, matrix=None, normalization=None, log1p=None):
    """Explicit notebook options override this run's JSON; never rewrite that file."""
    overrides = dict(selection=selection, matrix=matrix, normalization=normalization, log1p=log1p)
    return _engine().analyze(config_path, **{k: v for k, v in overrides.items() if v is not None})


def run_spacoex_data(expression, coordinates, *, dataset, coordinate_units,
                     preprocessing=None, candidate_policy, input_description,
                     workdir="spacoex_runs", matrix=None, selection=None,
                     library_total=None, normalization=None, log1p=None):
    """Run from labeled DataFrames. Preserve the exact inputs alongside the results."""
    import tempfile
    import numpy as np
    import pandas as pd

    engine = _engine()
    config = dict(dataset=dataset, input="input.npz", coordinate_units=coordinate_units,
                  candidate_policy=candidate_policy, input_description=input_description)
    config.update({k: v for k, v in dict(matrix=matrix, selection=selection,
                  preprocessing=preprocessing, normalization=normalization, log1p=log1p).items()
                  if v is not None})
    norm, _, _ = engine.resolve_preprocessing(config)
    if not isinstance(expression, pd.DataFrame) or not isinstance(coordinates, pd.DataFrame):
        raise TypeError("expression and coordinates must be DataFrames indexed by cell/spot ID.")
    if list(coordinates.columns).count("x") != 1 or list(coordinates.columns).count("y") != 1:
        raise ValueError("coordinates needs x and y columns, each appearing once.")
    if (expression.index.hasnans or coordinates.index.hasnans or expression.columns.hasnans
            or not expression.index.is_unique or not coordinates.index.is_unique
            or not expression.columns.is_unique):
        raise ValueError("Cell IDs and gene names must be unique and nonmissing.")
    if not expression.index.difference(coordinates.index).empty or not coordinates.index.difference(expression.index).empty:
        raise ValueError("Expression and coordinate cell IDs do not match.")
    payload = dict(X=expression.to_numpy(dtype=float), genes=expression.columns.to_numpy(dtype=str),
                   expression_ids=expression.index.to_numpy(dtype=str),
                   coordinate_ids=coordinates.index.to_numpy(dtype=str),
                   xy=coordinates[["x","y"]].to_numpy(dtype=float))
    if norm == "cp10k":
        if not isinstance(library_total, pd.Series) or not library_total.index.is_unique:
            raise ValueError("cp10k requires a Series of full-library totals indexed by cell ID.")
        payload["library_total"] = library_total.loc[expression.index].to_numpy(dtype=float)
    elif library_total is not None:
        raise ValueError("library_total applies only to cp10k normalization.")

    root = _Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = _Path(tempfile.mkdtemp(prefix="request_", dir=root))
    np.savez_compressed(request/"input.npz", **payload)
    config_path = request/"run.json"
    config_path.write_text(_json.dumps(config, indent=2), encoding="utf-8")
    return engine.analyze(config_path)
