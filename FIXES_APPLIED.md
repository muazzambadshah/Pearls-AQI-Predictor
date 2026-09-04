# Fixes applied — 2026-09-04

## 1. `app/api.py` was gutted (critical) — and its real self was recovered
Only 106 lines survived in the copy I first extracted: one orphaned helper
function (`_forecast_is_current`), no imports, no FastAPI app, no routes. The
whole documented REST API (`/health`, `/current`, `/forecast`,
`/forecast/daily`, `/alerts`, `/explain`, `/model`, `/metrics`) was missing,
so `pytest` couldn't even collect `tests/test_api.py`
(`NameError: name 'pd' is not defined`).

My first pass rebuilt `app/api.py` from `app/flask_api.py` (its intact,
behaviour-identical Flask twin) as a stopgap. But the extracted rar also
contained `app/api.py.backup` — an 854-line file that turned out to be the
*real* original: full Pydantic response models, OpenAPI tags, a proper
`lifespan` handler, and — importantly — a **disk-fallback cache**
(`_load_disk_forecast()` reads `data/predictions.parquet` when a fresh
forecast can't be generated). That's a materially better design than my
Flask-derived stopgap, so **I discarded my rebuild and restored this original
instead**, applying only the NaN fix below on top of it.

If you have other `*.backup` files anywhere in the repo, check them before
assuming a broken file needs rebuilding from scratch — this one did.

## 2. `NaN` in blend models' `train_rows` crashed `/model` (confirmed bug)
`src/train.py::_search_blends()` scores blend models (`blend_top2`,
`blend_top3`) by averaging already-collected predictions — no refit, so no
`train_rows` was ever set on their per-fold rows. Concatenating those rows
with real per-model folds and averaging in `aggregate_folds()` produced
`train_rows: NaN` for every blend. Since `blend_top2` is your production
model, this NaN reached `/model`'s response and crashed JSON serialization:
`ValueError: Out of range float values are not JSON compliant: nan`.

**Fix (root cause):** blend folds now carry the true `train_rows` of their
member models at that fold's cutoff (member models share the same expanding-
window cutoff, so the number is well-defined) — threaded through via a new
`fold_train_rows` dict, the same pattern already used for `fold_cutoffs`.

**Fix (defense in depth):** added a `_json_safe()` helper to both `app/api.py`
(the restored original) and `app/flask_api.py` that recursively replaces
NaN/Inf with `None` before returning `/model`'s response. Other legitimate
NaNs can occur elsewhere (e.g. `rmse_std` over a single fold) — this stops any
of them from ever taking the endpoint down again.

**Note:** your existing `models/registry.json` still has the old NaN baked in
from a past training run. The fix only prevents it going forward; the next
`python -m src.train` run will produce a clean `train_rows` for blends. The
`_json_safe()` fix means `/model` is already resilient even without retraining
— confirmed: `/model` now returns 200 with the NaN silently mapped to `null`.

## 3. Config mismatch: `FEATURE_STORE_BACKEND=hopsworks` without the package
`.env` had `FEATURE_STORE_BACKEND=hopsworks`, but `hopsworks` is commented out
in `requirements.txt` (by design — it's optional). Locally this meant every
feature-store call crashed with `ModuleNotFoundError: No module named
'hopsworks'`, even though `data/raw_observations.parquet` already had real
data cached under the `local` backend.

**Fix:** switched `.env` back to `FEATURE_STORE_BACKEND=local` with a comment
explaining how to switch back once you `pip install hopsworks` and rotate
your key (see #4).

## 4. Security: live credentials were in the uploaded zip
`.env` is correctly listed in `.gitignore` (never would have hit a public
repo), but the zip you uploaded still contained it directly, with a real
Hopsworks API key and AQICN token in plain text.

**Action needed from you:** rotate both —
- Hopsworks: regenerate the API key for `Pearls_AQI_Predictor_khan` in your
  Hopsworks account settings.
- AQICN: request a fresh token at https://aqicn.org/api/ (the old one should
  be considered compromised).

Both were redacted to placeholders in the copy returned here.

---

## Final verification

Full suite, in this sandbox, **without torch installed** (deliberately — disk/
network constraints here):

```
57 passed in 19.02s
```

All of `tests/test_api.py` (11/11), `tests/test_leakage.py` (7/7 — the
data-leakage guardrail your README specifically calls out), `test_pipeline.py`
and `test_conftest.py` pass. The earlier 4 failures I reported mid-session
were against my *interim* Flask-derived rebuild, which lacked the real file's
disk-fallback cache; they're gone now that the real `api.py` is restored.

On your machine, with the full `requirements.txt` (torch included) and a live
network, you should see the same `57 passed`, plus live (non-disk-fallback)
forecasts. Run `pytest -q` to confirm.
