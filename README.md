# 🌫️ Pearls AQI Predictor

Three-day hourly **Air Quality Index** forecasting for Lahore, end to end: automated
data collection, a leak-free feature pipeline, walk-forward model selection, a REST
API, and an interactive dashboard — retraining itself daily on GitHub Actions.

Built for the 10Pearls Shine "Pearls AQI Predictor" brief.

---

## What it does

Every hour it pulls fresh weather and pollutant readings into a feature store.
Every day it rebuilds three years of history, backtests nine candidate models
plus their blends against three baselines, promotes whichever wins, and
regenerates its own technical report. At any moment you can ask it for the next
72 hours of AQI — with uncertainty bands, an explanation of what is driving the
number, and an alert if unhealthy air is coming.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

That is the whole setup. The repo ships a trained model and 32,001 hours of real
observations, so the dashboard has a live 3-day forecast the moment it starts —
no cold start. To rebuild everything from scratch instead:

```bash
python -m src.backfill        # ~32,000 real hourly observations, 2023 → today
python -m src.train           # backtest, select, promote
python -m src.report          # regenerate reports/report.md
```

No API keys. No accounts. Open-Meteo's archive and forecast endpoints are free
and unauthenticated, and the default feature store is a local parquet file.

> **PyTorch is a required dependency, not an optional one.** The current
> production model is `blend_top2` — an average of the PyTorch MLP and LightGBM
> — so unpickling `models/best_model.pkl` imports `torch`. `requirements.txt`
> pulls the CPU-only build (~200 MB, no GPU needed). The registry records each
> model's load-time dependencies under `requires`, and `load_best_model()` says
> plainly what is missing rather than surfacing a bare `ModuleNotFoundError`.

---

## Accuracy

Every figure below comes from **walk-forward backtesting** — train to a cutoff,
score the window that follows, roll forward, repeat — never from data the model
saw during training.

| Lead time | RMSE | MAE | R² | Correct EPA band | Skill vs persistence |
|---|---|---|---|---|---|
| Day 1 (1–24h) | **19.9** | 12.8 | **0.831** | 75.5% | **+20.5%** |
| Day 2 (25–48h) | **29.5** | 19.6 | **0.640** | 64.0% | **+29.5%** |
| Day 3 (49–72h) | **32.5** | 22.2 | **0.564** | 60.0% | **+31.1%** |

Overall: **RMSE 26.8**, **R² 0.652**, **+31.9% skill** against persistence, across
four expanding-window folds spanning June 2024 to August 2026.

Two things worth reading off that table. R² falls from 0.83 to 0.56 as the lead
time triples — that decay is the honest shape of the problem, and any single
averaged figure would hide it. But **skill against persistence moves the other
way**, from +20% to +31%: persistence degrades faster than the model does, so
the model's advantage is widest exactly where forecasting is hardest and a naive
guess is least useful.

**Why skill is the headline and not R².** Repeating the last observed AQI is a
genuinely strong forecast for a series this autocorrelated — persistence alone
scores R² 0.27 here, and much higher at short lead times. Skill measures the
fraction of that baseline's error actually removed, which is the number that says
whether any of this was worth building.

### Model comparison

| Model | RMSE | MAE | R² | Skill |
|---|---|---|---|---|
| **blend_top2** (deep MLP + LightGBM) | **26.84** | 18.19 | 0.652 | **+0.319** |
| blend_top3 | 26.89 | 18.10 | 0.651 | +0.317 |
| deep_mlp_anchored (PyTorch) | 28.15 | 19.06 | 0.614 | +0.290 |
| lightgbm | 28.54 | 19.49 | 0.615 | +0.268 |
| hist_gradient_boosting_anchored | 28.72 | 19.06 | 0.602 | +0.271 |
| random_forest | 29.10 | 19.83 | 0.601 | +0.252 |
| ridge | 29.50 | 21.15 | 0.574 | +0.246 |
| elastic_net | 30.27 | 21.46 | 0.558 | +0.224 |
| *baseline: persistence* | *38.96* | *26.47* | *0.273* | — |
| *baseline: climatology* | *39.15* | *29.44* | *0.305* | — |
| *baseline: seasonal naive* | *40.15* | *27.49* | *0.244* | — |

The PyTorch MLP is the best *single* model, ahead of every tree ensemble — and
the blend of it with LightGBM beats either alone, because a neural network and a
boosted tree make different mistakes and averaging cancels part of both.

---

## Architecture

```
Open-Meteo archive + forecast APIs
              │
              ▼
   src/data_sources.py ──────► src/feature_store.py   (raw hourly observations)
                                        │
                                        ▼
                          src/feature_engineering.py  (origin | future | horizon)
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                  src/train.py                    src/predict.py
             backtest → select → promote      direct 72h forecast
                        │                               │
                        ▼                               ▼
              src/model_registry.py            app/api.py + app/streamlit_app.py
```

| Module | Role |
|---|---|
| `src/data_sources.py` | Archive + forecast fetching, retry/backoff, synthetic generator for offline tests |
| `src/feature_store.py` | Raw-observation store; local parquet or Hopsworks behind one interface |
| `src/feature_engineering.py` | Direct multi-horizon features with a strict leakage boundary |
| `src/baselines.py` | Persistence, seasonal naive, climatology |
| `src/models.py` | Candidate zoo — linear, forests, boosting, anchored variants |
| `src/deep_model.py` | Optional PyTorch MLP |
| `src/backtest.py` | Expanding-window validation with a target-time embargo |
| `src/evaluate.py` | Regression, skill, and operational metrics; per-horizon curves |
| `src/train.py` | The daily pipeline: build → backtest → select → promote |
| `src/predict.py` | Inference against the real weather forecast |
| `src/explainability.py` | SHAP, feature and group level |
| `src/alerts.py` | Hazardous-air episode detection and webhooks |
| `src/report.py` | Regenerates `reports/report.md` from actual artefacts |

---

## The two decisions that matter

### 1. Direct multi-horizon, not recursive

A 72-hour forecast can be made by training a 1-step model and feeding its output
back in 72 times. Errors compound, and every exogenous input has to be guessed
forward — which usually means freezing the weather at its last observed value and
watching the forecast flat-line.

Instead, one model is trained to map `(state at t, horizon h) → aqi[t+h]`, with
`h` as an input feature. Every prediction is made in one shot, and the **real
weather forecast** for `t+h` goes in as a feature. Open-Meteo publishes hourly
temperature, wind, humidity and precipitation days ahead; that is legitimately
known information, and it is what lets the model anticipate a rain front clearing
the air instead of merely extrapolating the present.

### 2. A hard leakage boundary, with a test that enforces it

Every feature is either **ORIGIN** (observations at or before `t`), **FUTURE**
(weather and calendar at `t+h`, genuinely known ahead), or **HORIZON** (the lead
time). Pollutant and AQI measurements at `t+h` never appear — Open-Meteo derives
`us_aqi` from the pollutant concentrations, so a target-time PM2.5 column would
be the answer wearing a disguise.

This is not hypothetical. The previous version of this project computed
`aqi_change_rate = aqi.diff()` and fed it to the model alongside `aqi_lag_1h`:

```
aqi_lag_1h  +  aqi_change_rate  =  aqi[t]
```

The target was an exact sum of two input columns. Every model scored R² ≈ 1.00,
every metric was meaningless, and nothing in the code looked wrong — only the
impossible score gave it away.

`tests/test_leakage.py` makes that silent failure impossible. Its central test
perturbs AQI values *after* a cutoff and asserts the feature matrix for earlier
origins is bit-identical. Any feature that reads the target at or beyond
prediction time fails it, however the leak is spelled.

---

## Running it

### Pipelines

```bash
python -m src.backfill                 # full history from the archive APIs
python -m src.backfill --synthetic     # offline, no network
python -m src.feature_pipeline         # hourly refresh
python -m src.train                    # backtest + promote
python -m src.train --fast --folds 2   # quick run for CI
python -m src.train --deep             # include the PyTorch candidate
python -m src.report                   # regenerate reports/report.md
python -m src.alerts                   # check the forecast for hazardous air
```

### Apps

```bash
streamlit run app/streamlit_app.py     # dashboard  → localhost:8501
uvicorn app.api:app --port 8000        # REST API   → localhost:8000/docs
```

### Tests

```bash
pytest -q
```

The suite runs entirely against a deterministic synthetic generator, so it never
depends on a live API or on committed model artefacts.

---

## API

| Endpoint | Returns |
|---|---|
| `GET /health` | Readiness of the model and the feature store |
| `GET /current` | Latest observed AQI, category, pollutant breakdown |
| `GET /forecast?hours=72` | Hourly forecast with 80% uncertainty bands |
| `GET /forecast/daily` | Day-by-day min / mean / max plus health advice |
| `GET /alerts?threshold=150` | Hazardous-air episodes, grouped into events |
| `GET /explain` | SHAP importance, per feature and per driver group |
| `GET /model` | Production model card and training leaderboard |
| `GET /metrics` | Backtest accuracy by lead time |

```bash
curl localhost:8000/forecast/daily
curl "localhost:8000/alerts?threshold=180"
```

---

## Automation

| Workflow | Schedule | Does |
|---|---|---|
| `feature_pipeline.yml` | hourly | Fetch → store → refresh forecast → check alerts |
| `training_pipeline.yml` | daily 02:40 UTC | Full backfill → backtest → promote → regenerate report |
| `ci.yml` | every push / PR | Tests on 3.11 and 3.12, plus a full offline pipeline run |

The CI job asserts `rmse > 0` and `r2 < 0.999` after training. Those two lines
are a tripwire: a leak reintroduced anywhere in the feature code shows up as an
impossibly perfect score, and the build fails instead of quietly shipping a
model that has learned nothing.

**A note on scheduled runs.** GitHub disables cron workflows on repositories with
no activity for 60 days, and both data-writing workflows share a `concurrency`
group so they can never push at once.

---

## Configuration

Everything is environment-driven — see [`.env.example`](.env.example). Nothing
needs setting for the defaults to work.

| Variable | Default | Purpose |
|---|---|---|
| `CITY_NAME` / `CITY_LAT` / `CITY_LON` | Lahore | Target location |
| `HISTORY_START_DATE` | `2023-01-01` | Where `us_aqi` starts being populated |
| `FEATURE_STORE_BACKEND` | `local` | `local` (parquet) or `hopsworks` |
| `MAX_TRAIN_ROWS` | `250000` | Memory bound on the horizon-expanded dataset |
| `ENABLE_DEEP_MODEL` | `0` | Include the PyTorch candidate |
| `ALERT_AQI_THRESHOLD` | `150` | Alert trigger level |
| `ALERT_WEBHOOK_URL` | — | Slack-compatible incoming webhook |

Pointing it at another city is a three-line change to `.env` followed by
`python -m src.backfill && python -m src.train`. Nothing in the code is
Lahore-specific.

---

## Known limitations

Set out in full in [`reports/report.md`](reports/report.md), but the two that
matter most:

- **Perfect-prog training.** Target-hour weather features come from *observed*
  weather during training and from a real forecast at inference. Live day-3
  accuracy will therefore be somewhat worse than the backtest indicates, because
  the backtest never had to survive a wrong weather forecast.
- **The target is a model output, not a sensor.** Open-Meteo's `us_aqi` comes
  from CAMS reanalysis, not from Lahore ground stations. The system forecasts a
  high-quality model product and inherits whatever regional bias it carries.

---

## Brief coverage

| Requirement | Where |
|---|---|
| Feature pipeline (fetch → compute → store) | `src/feature_pipeline.py`, `src/feature_engineering.py`, `src/feature_store.py` |
| Time-based + derived features | `src/feature_engineering.py` — cyclical calendar, lags, rolling, backward deltas |
| Feature store (Hopsworks-compatible) | `src/feature_store.py` |
| Historical backfill | `src/backfill.py` — ~32,000 real hourly rows |
| Training pipeline, RMSE / MAE / R² | `src/train.py`, `src/evaluate.py` |
| Multiple model families | `src/models.py`, `src/deep_model.py` — linear, forests, boosting, MLP |
| Model registry | `src/model_registry.py` |
| Automated CI/CD (hourly + daily) | `.github/workflows/` |
| Web dashboard (Streamlit) | `app/streamlit_app.py` |
| API (FastAPI) | `app/api.py` |
| EDA | `src/eda.py`, dashboard *Explore* tab |
| SHAP explanations | `src/explainability.py`, dashboard *Explain* tab |
| Hazardous-AQI alerts | `src/alerts.py` |
| Detailed report | `reports/report.md` |
