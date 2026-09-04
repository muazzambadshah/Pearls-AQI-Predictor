"""
Model Registry.

Stores every trained candidate with its metrics, the exact feature list it
expects, and a fingerprint of the data it was trained on, then promotes one to
production behind a stable path the dashboard and API can always load.

Promotion rule
--------------
The winner is chosen *within a training run* - the best test RMSE among models
that all saw identical data - and that winner is then promoted unconditionally.

The tempting alternative, "promote only if it beats the best RMSE ever
recorded", is wrong in a system that retrains daily on a growing window. Scores
from different runs are computed on different test sets, so they are not
comparable: an early model that got an easy fortnight can post a low RMSE and
then block every genuinely better model that follows. Comparing like with like
inside a run and always shipping that run's winner keeps the comparison valid
and the deployed model current.

Full run history is retained regardless, so the leaderboard can show how the
project has moved over time.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone

import joblib

from src import config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


def _load_registry() -> list[dict]:
    if not config.MODEL_REGISTRY_PATH.exists():
        return []
    try:
        with open(config.MODEL_REGISTRY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.warning("Registry index is corrupt; starting a fresh one")
        return []


def _save_registry(entries: list[dict]) -> None:
    config.MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, default=str)


def register(model,
             model_name: str,
             metrics: dict,
             feature_names: list[str],
             run_id: str,
             data_fingerprint: dict | None = None,
             horizon_metrics: list[dict] | None = None,
             extra: dict | None = None) -> str:
    """Persist one trained candidate and record it in the index."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.MODELS_DIR / f"{model_name}_{stamp}.pkl"

    joblib.dump({
        "model": model,
        "feature_names": list(feature_names),
        "model_name": model_name,
        "trained_at": stamp,
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
    }, path, compress=3)

    entry = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_name": model_name,
        "path": str(path),
        "metrics": metrics,
        "trained_at": stamp,
        "n_features": len(feature_names),
        "requires": model_requirements(model),
        "data_fingerprint": data_fingerprint or {},
        "horizon_metrics": horizon_metrics or [],
        **(extra or {}),
    }

    entries = _load_registry()
    entries.append(entry)
    _save_registry(entries)

    logger.info("Registered %-32s rmse=%.3f mae=%.3f r2=%.3f -> %s",
                model_name, metrics.get("rmse", float("nan")),
                metrics.get("mae", float("nan")), metrics.get("r2", float("nan")),
                path.name)
    return str(path)


def promote(run_id: str, model_name: str) -> dict:
    """
    Promote one model from `run_id` to production.

    Marks it in the index and copies its artefact to `models/best_model.pkl`, so
    consumers only ever need one stable path.
    """
    entries = _load_registry()
    match = next((e for e in entries
                  if e.get("run_id") == run_id and e.get("model_name") == model_name), None)
    if match is None:
        raise ValueError(f"No registry entry for run_id={run_id} model={model_name}")

    for entry in entries:
        entry["is_production"] = False
    match["is_production"] = True
    match["promoted_at"] = datetime.now(timezone.utc).isoformat()
    _save_registry(entries)

    shutil.copy(match["path"], config.BEST_MODEL_PATH)
    logger.info("PROMOTED %s (rmse=%.3f) -> %s",
                model_name, match["metrics"].get("rmse", float("nan")),
                config.BEST_MODEL_PATH.name)
    return match


def select_best(run_id: str, metric: str = "rmse", lower_is_better: bool = True) -> dict:
    """Best candidate within a single run - the only fair comparison available."""
    entries = [e for e in _load_registry() if e.get("run_id") == run_id]
    if not entries:
        raise ValueError(f"No registry entries for run_id={run_id}")

    def key(entry):
        value = entry["metrics"].get(metric)
        if value is None or value != value:  # None or NaN
            return float("inf") if lower_is_better else float("-inf")
        return value if lower_is_better else -value

    return min(entries, key=key)


def model_requirements(model) -> list[str]:
    """
    Third-party packages needed to *deserialise* this model.

    Recorded alongside the artefact because a pickle silently carries the import
    graph of whatever produced it. When the champion is a blend containing the
    PyTorch MLP, loading it imports torch - and an environment without torch
    fails at load time with a bare ModuleNotFoundError that says nothing about
    which model needed it or why.
    """
    needed: set[str] = set()

    def walk(obj):
        module = type(obj).__module__ or ""
        if module.startswith("torch"):
            needed.add("torch")
        if type(obj).__name__ == "TorchMLPRegressor":
            needed.add("torch")
        if module.startswith("lightgbm"):
            needed.add("lightgbm")
        for attr in ("base", "base_", "model", "members"):
            child = getattr(obj, attr, None)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                for item in child:
                    walk(item[1] if isinstance(item, tuple) and len(item) == 2 else item)
            else:
                walk(child)
        for step in getattr(getattr(obj, "named_steps", None), "values", lambda: [])():
            walk(step)
        for _, fitted in getattr(obj, "fitted_", []) or []:
            walk(fitted)

    try:
        walk(model)
    except Exception:  # noqa: BLE001 - best-effort metadata, never fatal
        pass
    return sorted(needed)


def load_best_model():
    """Return `(model, feature_names)` for the production model."""
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No production model found. Run the training pipeline first:\n"
            "    python -m src.train"
        )
    try:
        bundle = joblib.load(config.BEST_MODEL_PATH)
    except ModuleNotFoundError as exc:
        entry = production_entry() or {}
        needed = entry.get("requires") or []
        # `exc.name` is normally the missing package, but it is None for some
        # ways of raising this, so fall back to the recorded requirements.
        missing = exc.name or (", ".join(needed) if needed else "a required package")
        raise ModuleNotFoundError(
            f"Cannot load the production model - '{missing}' is not installed.\n"
            f"The promoted model is '{entry.get('model_name', 'unknown')}', which "
            f"depends on: {', '.join(needed) or missing}.\n"
            f"Install the full requirements:\n"
            f"    pip install -r requirements.txt\n"
            f"(PyTorch is required whenever the champion is - or contains - the "
            f"neural network; see the note in requirements.txt.)"
        ) from exc

    return bundle["model"], bundle["feature_names"]


def production_entry() -> dict | None:
    return next((e for e in _load_registry() if e.get("is_production")), None)


def leaderboard(limit: int | None = None) -> list[dict]:
    """All runs, best RMSE first."""
    entries = _load_registry()

    def rmse(entry):
        value = entry.get("metrics", {}).get("rmse")
        return float("inf") if value is None or value != value else value

    ranked = sorted(entries, key=rmse)
    return ranked[:limit] if limit else ranked


def latest_run_id() -> str | None:
    entries = _load_registry()
    return entries[-1].get("run_id") if entries else None


def prune(keep: int = 12) -> int:
    """
    Delete the oldest artefacts, keeping the production model and the most
    recent `keep` runs. A daily retrain would otherwise accumulate pickles
    without limit.
    """
    entries = _load_registry()
    if len(entries) <= keep:
        return 0

    protected = {e["path"] for e in entries if e.get("is_production")}
    stale = [e for e in entries[:-keep] if e["path"] not in protected]

    removed = 0
    for entry in stale:
        try:
            path = config.MODELS_DIR / entry["path"].split("\\")[-1].split("/")[-1]
            if path.exists():
                path.unlink()
            removed += 1
        except OSError as exc:  # noqa: PERF203
            logger.warning("Could not delete %s: %s", entry["path"], exc)

    stale_paths = {e["path"] for e in stale}
    _save_registry([e for e in entries if e["path"] not in stale_paths])
    logger.info("Pruned %d stale model artefacts", removed)
    return removed
