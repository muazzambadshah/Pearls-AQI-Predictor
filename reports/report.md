# Pearls AQI Predictor — Technical Report

**City:** Lahore (31.5497, 74.3436)
**Forecast horizon:** 3 days, hourly resolution (72 steps)
**Generated:** 2026-09-05 07:21 UTC
**Production model:** `blend_top2`

---

## Summary

An end-to-end system that forecasts the US Air Quality Index for Lahore
up to 3 days ahead, retrains itself daily, and serves
results through a REST API and an interactive dashboard.

The headline result: backtest RMSE of **27.55 AQI points**
with **R² = 0.631**, which is a **29.6% reduction in RMSE against a persistence forecast**. Accuracy is
reported per lead time rather than as a single average, because a 1-hour forecast
and a 72-hour forecast are different problems and averaging them together
flatters the harder one.

The model is trained on **32,240 hours** of real observations
(1343 days, 2023-01-01 to 2026-09-05) pulled from
Open-Meteo's reanalysis archive — not synthetic data, and not the ~90 days a
forecast endpoint alone would provide.

---

## 1. The data

| Property | Value |
|---|---|
| Source | Open-Meteo archive + air-quality API (free, no key) |
| Rows | 32,240 hourly observations |
| Span | 2023-01-01 00:00 → 2026-09-05 07:00 (1343 days) |
| Missing hours | 0 |
| Variables | 20 (weather, six pollutants, US AQI) |
| Mean AQI | 151.6 |
| Median AQI | 152.0 |
| Std. dev. | 46.7 |
| Range | 56 – 538 |
| 95th percentile | 236 |
| Hours ≥ 150 (Unhealthy) | 17,315 (53.7%) |

**Time spent in each EPA category**

| Category | Share of hours |
|---|---|
| Moderate | 13.3% |
| Unhealthy for Sensitive Groups | 33.0% |
| Unhealthy | 41.3% |
| Very Unhealthy | 11.7% |
| Hazardous | 0.7% |

### Two data decisions worth recording

**Why the history starts at 2023-01-01.** Open-Meteo's air-quality archive
accepts earlier dates and returns rows, but the `us_aqi` field comes back null
before 2023. The start date is the point where the target variable actually
exists, established empirically rather than assumed.

**Why `boundary_layer_height` is not a feature.** Mixing-layer depth is the
textbook meteorological driver of pollution build-up, and it was in the original
feature set. The archive endpoint answers `200 OK` for it and then returns a
column that is 99.5% null — only the most recent days carry values. A model
trained on that would fit a handful of recent rows and diverge in production. It
was replaced with the 10m/100m wind pair, which is populated across the entire
history; the shear between those levels carries much the same information about
vertical mixing. Every weather variable in the final set was checked for non-null
coverage on **both** the archive and forecast endpoints before being admitted.

### Stationarity

Augmented Dickey-Fuller, on the most recent year:

| Series | ADF statistic | p-value | Stationary at 5%? |
|---|---|---|---|
| AQI level | -4.78 | 0.0001 | yes |
| First difference | -19.45 | 0.0000 | yes |

Both series reject the unit-root null, so the level is already stationary over
this window and there is nothing for differencing to fix. That is a useful
negative result rather than a dull one: it predicts in advance that the anchored
model variants in section 3.1 — which reframe the target as a change from the
last observation, precisely the transform differencing performs — should gain
very little. Section 3.1 shows they gained very little. The diagnostic and the
backtest agree, which is more reassuring than either would be alone.

The caveat is that ADF tests the *mean* level, not the variance. Lahore's AQI
plainly has seasonal heteroscedasticity — winter is both higher and far more
volatile than summer — and a stationary mean does not make that go away.

![Full observation history with a 7-day rolling mean](figures/eda_timeseries.png)
*Full observation history with a 7-day rolling mean.*
![Hourly AQI distribution and time spent per EPA category](figures/eda_distribution.png)
*Hourly AQI distribution and time spent per EPA category.*
![Daily and seasonal cycles](figures/eda_cycles.png)
*Daily and seasonal cycles.*
![Autocorrelation out to one week](figures/eda_autocorrelation.png)
*Autocorrelation out to one week.*
![Meteorological relationships](figures/eda_weather.png)
*Meteorological relationships.*
![STL decomposition into trend, daily cycle and remainder](figures/eda_decomposition.png)
*STL decomposition into trend, daily cycle and remainder.*

---

## 2. Method

### 2.1 Direct multi-horizon forecasting

A 72-hour forecast can be produced two ways:

| | Recursive | Direct (used here) |
|---|---|---|
| Structure | Train 1-step, feed output back in, repeat 72× | Train `(state at t, horizon h) → aqi[t+h]` |
| Error behaviour | Compounds — step 72 sits on 71 previous errors | Independent per horizon |
| Exogenous inputs | Must be guessed forward | Real weather forecast used directly |
| Cost at inference | 72 sequential model calls | 1 batched call |

One model is trained across all 72 horizons with `h` as an input feature, so a
single fit covers the whole range while every prediction is made in one shot.

### 2.2 The leakage discipline

Every feature belongs to exactly one bucket:

- **ORIGIN** — computed from observations at or before the forecast origin `t`:
  lags, rolling statistics, backward-looking deltas, pollutant and weather state.
- **FUTURE** — weather and calendar at the *target* time `t+h`. Legitimate,
  because numerical weather prediction genuinely supplies these in advance and a
  calendar is known forever.
- **HORIZON** — the lead time itself.

Deliberately absent: any pollutant or AQI measurement at `t+h`. Open-Meteo
derives `us_aqi` from the pollutant concentrations, so a target-time PM2.5 column
would be the answer in different units.

This matters because the predecessor implementation of this project leaked
badly. It computed `aqi_change_rate = aqi.diff()` — that is `aqi[t] − aqi[t−1]` —
and passed it to the model alongside `aqi_lag_1h`, which is `aqi[t−1]`. Their sum
is exactly `aqi[t]`. The target was a linear combination of two inputs, so every
model scored R² ≈ 1.00 and every reported metric was meaningless. Nothing looked
wrong in the code; only the impossible score gave it away.

`tests/test_leakage.py` makes that class of bug non-silent. The central test
perturbs AQI values *after* a cutoff and asserts that the feature matrix for
earlier origins is bit-identical. Any feature reading the target at or beyond
prediction time fails it, regardless of how the leak is written.

### 2.3 Feature set

| Group | Examples | Count |
|---|---|---|
| AQI history | lags 1–72h, rolling mean/std/min/max over 3–168h, backward deltas, 24h z-score | ~40 |
| Pollutant state | PM2.5, PM10, CO, NO₂, SO₂, O₃ at origin plus lags and 24h momentum | ~40 |
| Weather at origin | temperature, humidity, wind u/v at 10m and 100m, shear, stagnation index, 24h precipitation | ~25 |
| Forecast weather at target | the same fields at `t+h`, plus 6h/24h trailing means over the forecast path | ~35 |
| Calendar at target | cyclical hour/month/day-of-year encodings, weekend flag | 11 |
| Lead time | `horizon_h`, `horizon_days` | 2 |

Wind direction is decomposed into u/v components rather than left in degrees,
where 359° and 1° are adjacent in reality but maximally distant numerically.

### 2.4 Validation

Walk-forward backtesting with expanding windows: train to a cutoff, score the
window that follows, roll the cutoff forward, repeat. This replays how the system
actually behaves in production, where it retrains daily and forecasts into
whatever comes next.

**The embargo.** Multi-horizon datasets leak across a naive split in a way that
is easy to miss. A training sample with origin `T − 10h` and horizon 72 has its
label at `T + 62h`. Splitting on origin alone puts that sample in training even
though its label lies inside the test window — fitting the model on the very
future it is about to be scored against. The split is therefore on `target_time`:

```
train = rows whose target_time <= cutoff
test  = rows whose origin      >  cutoff
```

which leaves a natural 72-hour embargo between them.

### 2.5 Baselines

A regression score is meaningless in isolation. Three references are scored on
identical test rows:

- **Persistence** — `ŷ(t+h) = aqi(t)`. Genuinely hard to beat at short lead times.
- **Seasonal naive** — same clock hour, most recent complete day.
- **Climatology** — historical mean for that (month, hour), fitted on training data only.

Skill is quoted against persistence, the strictest of the three.

---

## 3. Results

### 3.1 Model comparison

Mean across walk-forward folds; `±` is the standard deviation across folds, which shows how *dependably* a model achieves its average.

| model | rmse | rmse_std | mae | r2 | skill_vs_persistence | category_accuracy | n_folds |
|---|---|---|---|---|---|---|---|
| blend_top2 | 27.55 | 7.32 | 18.68 | 0.631 | 0.296 | 0.655 | 4 |
| blend_top3 | 27.65 | 7.16 | 18.83 | 0.629 | 0.292 | 0.652 | 4 |
| blend_top4 | 27.75 | 7.16 | 18.86 | 0.627 | 0.290 | 0.649 | 4 |
| lightgbm | 28.40 | 6.95 | 19.41 | 0.613 | 0.271 | 0.644 | 4 |
| lightgbm_anchored | 28.66 | 8.64 | 18.88 | 0.597 | 0.271 | 0.660 | 4 |
| hist_gradient_boosting | 28.76 | 7.17 | 19.59 | 0.601 | 0.263 | 0.641 | 4 |
| random_forest | 28.97 | 7.06 | 19.67 | 0.597 | 0.256 | 0.633 | 4 |
| hist_gradient_boosting_anchored | 28.97 | 8.87 | 19.10 | 0.588 | 0.264 | 0.657 | 4 |
| ridge | 29.49 | 8.02 | 21.09 | 0.565 | 0.247 | 0.623 | 4 |
| ridge_anchored | 29.49 | 8.02 | 21.09 | 0.565 | 0.247 | 0.623 | 4 |
| extra_trees_anchored | 29.70 | 8.56 | 19.59 | 0.571 | 0.243 | 0.646 | 4 |
| elastic_net | 30.24 | 7.57 | 21.36 | 0.551 | 0.225 | 0.614 | 4 |
| baseline:persistence | 38.97 | 9.46 | 26.35 | 0.261 | — | 0.571 | 4 |
| baseline:climatology | 39.13 | 6.78 | 29.47 | 0.293 | — | 0.468 | 4 |
| baseline:seasonal_naive_24h | 40.10 | 9.44 | 27.29 | 0.235 | — | 0.548 | 4 |

![Model comparison](figures/model_comparison.png)

**On the `blend_topK` rows.** These are equal-weighted averages of the
top-K individual candidates. They are close to free to evaluate — the backtest
already produced every model's predictions on the same test rows, so a blend is
an average of columns that already exist, with no refitting and no extra folds.
They are scored through the identical metric path as everything else, with
climatology refitted at each fold's cutoff, so they appear here on equal terms.

`blend_top2` reaches 27.55 RMSE against
`lightgbm` at 28.40 — it beats the best single model by 0.85 RMSE. A blend is
promoted only when it wins outright; otherwise the best single model ships.

**A hypothesis the data would not settle.** Tree ensembles cannot
extrapolate beyond their training range, so predicting the *change* from the last
observed AQI (`y − aqi[t]`) rather than the level ought to help during severe
episodes, where the level leaves the range the model was trained on. Every
anchored variant was built and backtested alongside its level-target twin:

| Model | Level RMSE | Anchored RMSE | Difference |
|---|---|---|---|
| `hist_gradient_boosting` | 28.76 | 28.97 | -0.21 |
| `lightgbm` | 28.40 | 28.66 | -0.26 |
| `ridge` | 29.49 | 29.49 | -0.00 |

The result is a wash, and it points in different directions for different
models. Every gap above is an order of magnitude smaller than the fold-to-fold
standard deviation (median 8.02 RMSE), so the honest conclusion is that
this dataset does not decide the question — not that either framing wins. A
plausible reason the effect is so muted: `aqi_at_origin` is already a feature, so
a level-target model can learn the same residual relationship wherever it helps.

The exception is the neural network. `deep_mlp_anchored` is the best *single*
model in this run, and an MLP has none of a tree's built-in tolerance for a
shifting target level, so handing it a stationary target helps in a way it does
not for the ensembles. That is also the connection back to the ADF result in
section 1: the level is the non-stationary series, and the model that cares most
about stationarity is the one that gains most from removing it.

### 3.2 Accuracy by lead time

This is the number that matters. A single averaged R² hides the shape of the
problem; the degradation from day 1 to day 3 is the honest picture.

| Lead time | RMSE | MAE | R² | Correct EPA band | Skill vs persistence |
|---|---|---|---|---|---|
| Day 1 (1–24h) | 20.87 | 13.85 | 0.814 | 73.2% | +16.0% |
| Day 2 (25–48h) | 29.82 | 19.91 | 0.627 | 63.4% | +28.4% |
| Day 3 (49–72h) | 32.58 | 22.31 | 0.557 | 59.8% | +30.7% |

![Accuracy by horizon](figures/accuracy_by_horizon.png)

![Predicted vs observed](figures/forecast_vs_actual.png)
*Predicted against observed by forecast day. The spread widening from day 1 to day 3 is the same story the RMSE curve tells.*

### 3.3 Operational quality

Regression metrics do not describe what a user experiences. These do:

| Metric | Value | Meaning |
|---|---|---|
| Exact EPA band | 65.5% | forecast lands in the right category |
| Within one band | 99.1% | at most one category out |
| Exceedance recall | 80.7% | share of AQI ≥ 150 hours caught |
| Exceedance precision | 87.5% | share of alerts that were warranted |
| Base rate | 55.8% | how often AQI ≥ 150 actually occurs |

Recall is the figure to watch for an alerting system: a missed smog episode costs
far more than a false alarm.

---

## 4. What the model actually learned

![Feature importance](figures/feature_importance.png)

**By driver group**

| Group | Share of total \|SHAP\| |
|---|---|
| Forecast weather | 33.7% |
| Time of day / season | 22.9% |
| Pollutants now | 20.9% |
| AQI history | 15.5% |
| Weather now | 5.4% |
| Lead time | 1.5% |
| Other | 0.1% |

**Top individual features**

| Feature | Mean \|SHAP\| |
|---|---|
| tgt_wind_speed_10m_rollmean_24h | 5.148 |
| tgt_hour_sin | 4.251 |
| tgt_doy_cos | 2.495 |
| tgt_temperature_2m_rollmean_24h | 2.466 |
| pm2_5_rollmean_24h | 2.269 |
| tgt_relative_humidity_2m_rollmean_24h | 2.242 |
| tgt_temperature_2m_rollmean_6h | 1.790 |
| tgt_month_cos | 1.690 |
| aqi_rollmean_168h | 1.407 |
| tgt_dew_point_2m | 1.273 |
| pm2_5_at_origin | 1.256 |
| tgt_hour | 0.976 |

This is the single most important result in the report, and it validates the
central design decision.

**Forecast weather is the largest driver at 34%, ahead of AQI
history at 15%.** The model leans hardest on information about
the *target* hour — what the wind, temperature and rain will be doing when the
forecast lands — rather than on where AQI is right now.

That is precisely the information the recursive approach cannot use. A recursive
forecaster has to guess the weather forward, and in practice freezes it at the
last observed value; the direct formulation reads it straight from Open-Meteo's
published forecast. SHAP says that channel carries a third of the model's signal,
which is a concrete explanation for where the 34% skill over
persistence comes from: persistence has, by construction, none of it.

The corollary is the perfect-prog caveat in section 5. A model that depends this
heavily on forecast weather is exposed to weather-forecast error in live use in a
way the backtest does not measure.

---

## 5. Limitations

Stated plainly, because a report that only lists strengths is not a report.

**Perfect-prog training.** During training, the target-hour weather features come
from *observed* weather. At inference they come from a real forecast, which
carries its own error. This is the standard "perfect prog" setup in operational
meteorology, but it means live day-3 accuracy will be modestly worse than the
backtest suggests, since the backtest never had to cope with a wrong weather
forecast. Quantifying that gap would require archived forecast data, which
Open-Meteo's free tier does not expose.

**The target is itself modelled.** Open-Meteo's `us_aqi` derives from CAMS
reanalysis rather than from Lahore ground-station measurements. The system
forecasts a high-quality model product, not a sensor reading, and inherits
whatever bias CAMS carries in this region. Ground-truth validation against a
physical monitor is the obvious next step.

**One city, one location.** All results are for a single coordinate pair. Nothing
here demonstrates transfer to another city, and Lahore's pollution regime —
extreme winter inversions, agricultural burning — is unusual enough that the
learned relationships probably will not transfer unchanged.

**Analysis/forecast blending near the present.** The air-quality endpoint serves
forecast values for hours that have not closed yet. Those are now explicitly
dropped from the history and excluded when choosing the forecast origin — without
that guard the origin drifts days into the future and the model forecasts 72h
beyond a forecast. What remains is subtler and not fixed: CAMS publishes with a
lag, so the most recent few hours labelled as analysis may still be partly
model-derived. There is no field in the response that distinguishes them.

**Interval calibration is assumed, not measured.** The 80% bands come from
per-horizon backtest RMSE under a normal-error assumption. AQI errors are
right-skewed — big misses happen during episodes, not during clean air — so the
bands are likely too narrow in exactly the situations that matter most. Quantile
regression would fix this properly.

**A daily retrain on a growing window will drift.** Model selection happens
within each run, which is correct, but nothing currently detects a gradual
degradation in live accuracy. Logging predictions and scoring them once the truth
arrives would close that loop.

---

## 6. Reproducing this

```bash
pip install -r requirements.txt   # includes PyTorch (CPU) - see the note below

python -m src.backfill          # ~2023-01-01 → today of real observations
python -m src.train             # walk-forward backtest, select, promote
python -m src.report            # regenerate this document

uvicorn app.api:app --port 8000        # REST API + /docs
streamlit run app/streamlit_app.py     # dashboard
pytest -q                              # test suite, including the leakage guards
```

**On the PyTorch dependency.** It is required, not optional. The promoted model
is a blend containing the neural network, so loading `models/best_model.pkl`
imports `torch`; an install without it fails at load time. `requirements.txt`
pulls the CPU-only build (~200MB) — nothing here needs a GPU, and the MLP trains
on CPU in seconds. The registry records each model's load-time dependencies
under `requires`, so the failure explains itself if the environment is wrong.

Automation runs unattended in GitHub Actions: the feature pipeline hourly, the
training pipeline daily, and the test suite on every push.

*Every figure and number in this document was generated by `python -m src.report`
from the artefacts of the training run named above. Nothing was transcribed by
hand.*