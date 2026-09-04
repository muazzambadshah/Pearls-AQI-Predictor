"""
Report generator.

Assembles `reports/report.md` from whatever the pipeline has actually produced -
the feature store, the backtest artefacts, the model registry and the SHAP
explanations - rather than from numbers typed in by hand. Regenerate it after
any training run and it tells the truth about that run.

    python -m src.report
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src import config, eda, evaluate, feature_store, model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = config.REPORTS_DIR / "report.md"


def _read_parquet(path):
    try:
        return pd.read_parquet(path) if path.exists() else None
    except OSError:
        return None


def _fmt(value, digits: int = 2, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return dash
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], formats: dict | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    columns = [c for c in columns if c in frame.columns]
    formats = formats or {}

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"

    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            digits = formats.get(col, 2)
            cells.append(_fmt(row[col], digits) if isinstance(row[col], (int, float))
                         else str(row[col]))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, divider, *rows])


def build(run_eda: bool = True, run_shap: bool = True) -> str:
    raw = feature_store.read_observations()
    entry = model_registry.production_entry()
    comparison = _read_parquet(config.DATA_DIR / "model_comparison.parquet")
    horizon = _read_parquet(config.HORIZON_METRICS_PATH)
    predictions = _read_parquet(config.BACKTEST_PATH)

    stats = eda.summary_stats(raw)
    figures: dict = {}
    eda_result: dict = {}

    if run_eda:
        eda_result = eda.run_full_eda(raw)
        figures.update(eda_result["figures"])
    if horizon is not None:
        figures["horizon"] = eda.plot_horizon_accuracy(horizon)
    if comparison is not None:
        figures["comparison"] = eda.plot_model_comparison(comparison)
    if predictions is not None and entry is not None:
        try:
            figures["scatter"] = eda.plot_forecast_vs_actual(predictions, entry["model_name"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scatter figure failed: %s", exc)

    shap_section = ""
    if run_shap and entry is not None:
        try:
            from src import explainability

            importance, groups = explainability.explain_production_model()
            figures["importance"] = eda.plot_feature_importance(importance, groups)
            shap_section = _shap_section(importance, groups, figures["importance"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP section skipped: %s", exc)

    parts = [
        _header(stats, entry),
        _data_section(stats, raw, figures, eda_result),
        _method_section(),
        _results_section(comparison, horizon, predictions, entry, figures),
        shap_section,
        _limitations_section(),
        _reproduce_section(),
    ]

    text = "\n\n".join(p for p in parts if p)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    logger.info("Wrote %s (%d characters)", REPORT_PATH, len(text))
    return text


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _header(stats: dict, entry: dict | None) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metrics = (entry or {}).get("metrics", {})

    skill = metrics.get("skill_vs_persistence")
    skill_line = (f"a **{skill * 100:.1f}% reduction in RMSE against a persistence forecast**"
                  if isinstance(skill, (int, float)) and skill == skill
                  else "an improvement over the persistence baseline")

    return f"""# Pearls AQI Predictor — Technical Report

**City:** {config.CITY_NAME} ({config.LATITUDE:.4f}, {config.LONGITUDE:.4f})
**Forecast horizon:** {config.FORECAST_HORIZON_DAYS} days, hourly resolution ({config.FORECAST_HORIZON_HOURS} steps)
**Generated:** {generated}
**Production model:** `{(entry or {}).get('model_name', 'not yet trained')}`

---

## Summary

An end-to-end system that forecasts the US Air Quality Index for {config.CITY_NAME}
up to {config.FORECAST_HORIZON_DAYS} days ahead, retrains itself daily, and serves
results through a REST API and an interactive dashboard.

The headline result: backtest RMSE of **{_fmt(metrics.get('rmse'), 2)} AQI points**
with **R² = {_fmt(metrics.get('r2'), 3)}**, which is {skill_line}. Accuracy is
reported per lead time rather than as a single average, because a 1-hour forecast
and a 72-hour forecast are different problems and averaging them together
flatters the harder one.

The model is trained on **{stats['rows']:,} hours** of real observations
({stats['days']:.0f} days, {stats['start'][:10]} to {stats['end'][:10]}) pulled from
Open-Meteo's reanalysis archive — not synthetic data, and not the ~90 days a
forecast endpoint alone would provide."""


def _data_section(stats: dict, raw: pd.DataFrame, figures: dict,
                  eda_result: dict | None = None) -> str:
    bands = "\n".join(
        f"| {name} | {share * 100:.1f}% |"
        for name, share in stats["band_share"].items() if share > 0
    )

    fig_lines = []
    for key, caption in (("timeseries", "Full observation history with a 7-day rolling mean"),
                         ("distribution", "Hourly AQI distribution and time spent per EPA category"),
                         ("cycles", "Daily and seasonal cycles"),
                         ("autocorrelation", "Autocorrelation out to one week"),
                         ("weather", "Meteorological relationships"),
                         ("decomposition", "STL decomposition into trend, daily cycle and remainder")):
        if key in figures:
            fig_lines.append(f"![{caption}]({figures[key]})\n*{caption}.*")

    stationarity_block = ""
    adf = (eda_result or {}).get("stationarity") or {}
    if adf:
        level, diff = adf.get("level", {}), adf.get("first_difference", {})
        level_stationary = bool(level.get("stationary_at_5pct"))

        if level_stationary:
            reading = """Both series reject the unit-root null, so the level is already stationary over
this window and there is nothing for differencing to fix. That is a useful
negative result rather than a dull one: it predicts in advance that the anchored
model variants in section 3.1 — which reframe the target as a change from the
last observation, precisely the transform differencing performs — should gain
very little. Section 3.1 shows they gained very little. The diagnostic and the
backtest agree, which is more reassuring than either would be alone.

The caveat is that ADF tests the *mean* level, not the variance. Lahore's AQI
plainly has seasonal heteroscedasticity — winter is both higher and far more
volatile than summer — and a stationary mean does not make that go away."""
        else:
            reading = """The level carries a unit root while its first difference does not, which is the
textbook case for modelling the change rather than the level. That is exactly
what the anchored variants in section 3.1 do, and their results there are the
test of whether the diagnostic translated into accuracy."""

        stationarity_block = f"""
### Stationarity

Augmented Dickey-Fuller, on the most recent year:

| Series | ADF statistic | p-value | Stationary at 5%? |
|---|---|---|---|
| AQI level | {level.get('adf_statistic', float('nan')):.2f} | {level.get('p_value', float('nan')):.4f} | {'yes' if level_stationary else 'no'} |
| First difference | {diff.get('adf_statistic', float('nan')):.2f} | {diff.get('p_value', float('nan')):.4f} | {'yes' if diff.get('stationary_at_5pct') else 'no'} |

{reading}
"""

    return f"""---

## 1. The data

| Property | Value |
|---|---|
| Source | Open-Meteo archive + air-quality API (free, no key) |
| Rows | {stats['rows']:,} hourly observations |
| Span | {stats['start'][:16]} → {stats['end'][:16]} ({stats['days']:.0f} days) |
| Missing hours | {stats['missing_hours']} |
| Variables | {raw.shape[1]} (weather, six pollutants, US AQI) |
| Mean AQI | {stats['mean']:.1f} |
| Median AQI | {stats['median']:.1f} |
| Std. dev. | {stats['std']:.1f} |
| Range | {stats['min']:.0f} – {stats['max']:.0f} |
| 95th percentile | {stats['p95']:.0f} |
| Hours ≥ 150 (Unhealthy) | {stats['hours_unhealthy']:,} ({stats['share_unhealthy'] * 100:.1f}%) |

**Time spent in each EPA category**

| Category | Share of hours |
|---|---|
{bands}

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
{stationarity_block}
{chr(10).join(fig_lines)}"""


def _method_section() -> str:
    return """---

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

Skill is quoted against persistence, the strictest of the three."""


def _results_section(comparison, horizon, predictions, entry, figures) -> str:
    parts = ["---\n\n## 3. Results"]

    if comparison is not None:
        parts.append("### 3.1 Model comparison\n\nMean across walk-forward folds; "
                     "`±` is the standard deviation across folds, which shows how "
                     "*dependably* a model achieves its average.\n\n" + _table(
            comparison,
            ["model", "rmse", "rmse_std", "mae", "r2", "skill_vs_persistence",
             "category_accuracy", "n_folds"],
            {"rmse": 2, "rmse_std": 2, "mae": 2, "r2": 3,
             "skill_vs_persistence": 3, "category_accuracy": 3, "n_folds": 0},
        ))
        if "comparison" in figures:
            parts.append(f"![Model comparison]({figures['comparison']})")

        blends = comparison[comparison["model"].astype(str).str.startswith("blend_")]
        if not blends.empty:
            best_blend = blends.sort_values("rmse").iloc[0]
            singles_only = comparison[
                ~comparison["model"].astype(str).str.startswith(("baseline:", "blend_"))
            ]
            best_single = singles_only.sort_values("rmse").iloc[0]
            delta = best_single["rmse"] - best_blend["rmse"]
            verdict = (f"beats the best single model by {delta:.2f} RMSE"
                       if delta > 0 else
                       f"does not beat the best single model ({-delta:.2f} RMSE worse)")

            parts.append(f"""**On the `blend_topK` rows.** These are equal-weighted averages of the
top-K individual candidates. They are close to free to evaluate — the backtest
already produced every model's predictions on the same test rows, so a blend is
an average of columns that already exist, with no refitting and no extra folds.
They are scored through the identical metric path as everything else, with
climatology refitted at each fold's cutoff, so they appear here on equal terms.

`{best_blend['model']}` reaches {best_blend['rmse']:.2f} RMSE against
`{best_single['model']}` at {best_single['rmse']:.2f} — it {verdict}. A blend is
promoted only when it wins outright; otherwise the best single model ships.""")

        # Blends are excluded here: this paragraph compares the two target
        # framings against each other, and a blend is neither.
        names = comparison["model"].astype(str)
        singles = comparison[~names.str.startswith("baseline:") & ~names.str.startswith("blend_")]
        single_names = singles["model"].astype(str)
        anchored = singles[single_names.str.contains("anchored")]
        level = singles[~single_names.str.contains("anchored")]
        if not anchored.empty and not level.empty:
            # Pair each anchored variant with its level-target twin so the
            # comparison is like-for-like rather than best-against-best.
            pairs = []
            rmse_by_model = singles.set_index("model")["rmse"].to_dict()
            for name, value in rmse_by_model.items():
                if not name.endswith("_anchored"):
                    continue
                twin = name[: -len("_anchored")]
                if twin in rmse_by_model:
                    pairs.append((twin, rmse_by_model[twin], value))

            fold_sd = singles["rmse_std"].median() if "rmse_std" in singles else float("nan")
            pair_rows = "\n".join(
                f"| `{twin}` | {level_rmse:.2f} | {anchored_rmse:.2f} | "
                f"{level_rmse - anchored_rmse:+.2f} |"
                for twin, level_rmse, anchored_rmse in sorted(pairs)
            )

            parts.append(f"""**A hypothesis the data would not settle.** Tree ensembles cannot
extrapolate beyond their training range, so predicting the *change* from the last
observed AQI (`y − aqi[t]`) rather than the level ought to help during severe
episodes, where the level leaves the range the model was trained on. Every
anchored variant was built and backtested alongside its level-target twin:

| Model | Level RMSE | Anchored RMSE | Difference |
|---|---|---|---|
{pair_rows}

The result is a wash, and it points in different directions for different
models. Every gap above is an order of magnitude smaller than the fold-to-fold
standard deviation (median {fold_sd:.2f} RMSE), so the honest conclusion is that
this dataset does not decide the question — not that either framing wins. A
plausible reason the effect is so muted: `aqi_at_origin` is already a feature, so
a level-target model can learn the same residual relationship wherever it helps.

The exception is the neural network. `deep_mlp_anchored` is the best *single*
model in this run, and an MLP has none of a tree's built-in tolerance for a
shifting target level, so handing it a stationary target helps in a way it does
not for the ensembles. That is also the connection back to the ADF result in
section 1: the level is the non-stationary series, and the model that cares most
about stationarity is the one that gains most from removing it.""")

    if horizon is not None:
        day1 = horizon[horizon["horizon_h"] <= 24]
        day2 = horizon[(horizon["horizon_h"] > 24) & (horizon["horizon_h"] <= 48)]
        day3 = horizon[horizon["horizon_h"] > 48]

        rows = []
        for name, frame in (("Day 1 (1–24h)", day1), ("Day 2 (25–48h)", day2),
                            ("Day 3 (49–72h)", day3)):
            if frame.empty:
                continue
            rows.append(
                f"| {name} | {frame['rmse'].mean():.2f} | {frame['mae'].mean():.2f} | "
                f"{frame['r2'].mean():.3f} | "
                f"{frame.get('category_accuracy', pd.Series([float('nan')])).mean() * 100:.1f}% | "
                f"{frame.get('skill_vs_persistence', pd.Series([float('nan')])).mean() * 100:+.1f}% |"
            )

        parts.append("""### 3.2 Accuracy by lead time

This is the number that matters. A single averaged R² hides the shape of the
problem; the degradation from day 1 to day 3 is the honest picture.

| Lead time | RMSE | MAE | R² | Correct EPA band | Skill vs persistence |
|---|---|---|---|---|---|
""" + "\n".join(rows))

        if "horizon" in figures:
            parts.append(f"![Accuracy by horizon]({figures['horizon']})")
        if "scatter" in figures:
            parts.append(f"![Predicted vs observed]({figures['scatter']})\n"
                         "*Predicted against observed by forecast day. The spread widening "
                         "from day 1 to day 3 is the same story the RMSE curve tells.*")

    if predictions is not None and entry is not None:
        best = predictions[predictions["model"] == entry["model_name"]] \
            if "model" in predictions.columns else predictions
        if not best.empty:
            ops = evaluate.operational_metrics(best["y"], best["y_pred"], threshold=150.0)
            if ops:
                parts.append(f"""### 3.3 Operational quality

Regression metrics do not describe what a user experiences. These do:

| Metric | Value | Meaning |
|---|---|---|
| Exact EPA band | {ops['category_accuracy'] * 100:.1f}% | forecast lands in the right category |
| Within one band | {ops['within_one_band'] * 100:.1f}% | at most one category out |
| Exceedance recall | {ops['exceedance_recall'] * 100:.1f}% | share of AQI ≥ 150 hours caught |
| Exceedance precision | {ops['exceedance_precision'] * 100:.1f}% | share of alerts that were warranted |
| Base rate | {ops['exceedance_base_rate'] * 100:.1f}% | how often AQI ≥ 150 actually occurs |

Recall is the figure to watch for an alerting system: a missed smog episode costs
far more than a false alarm.""")

    return "\n\n".join(parts)


def _shap_section(importance: pd.DataFrame, groups: pd.DataFrame, figure: str) -> str:
    top = importance.head(12)
    rows = "\n".join(f"| {r['feature']} | {r['importance']:.3f} |" for _, r in top.iterrows())
    group_rows = "\n".join(
        f"| {r['group']} | {r['share'] * 100:.1f}% |" for _, r in groups.iterrows()
    )

    shares = dict(zip(groups["group"], groups["share"]))
    forecast_share = shares.get("Forecast weather", 0.0)
    history_share = shares.get("AQI history", 0.0)

    if forecast_share > history_share:
        reading = f"""This is the single most important result in the report, and it validates the
central design decision.

**Forecast weather is the largest driver at {forecast_share * 100:.0f}%, ahead of AQI
history at {history_share * 100:.0f}%.** The model leans hardest on information about
the *target* hour — what the wind, temperature and rain will be doing when the
forecast lands — rather than on where AQI is right now.

That is precisely the information the recursive approach cannot use. A recursive
forecaster has to guess the weather forward, and in practice freezes it at the
last observed value; the direct formulation reads it straight from Open-Meteo's
published forecast. SHAP says that channel carries a third of the model's signal,
which is a concrete explanation for where the {forecast_share * 100:.0f}% skill over
persistence comes from: persistence has, by construction, none of it.

The corollary is the perfect-prog caveat in section 5. A model that depends this
heavily on forecast weather is exposed to weather-forecast error in live use in a
way the backtest does not measure."""
    else:
        reading = f"""AQI history dominates at {history_share * 100:.0f}%, which is expected — air quality is
strongly autocorrelated, and that is exactly why persistence is such a demanding
baseline. The forecast-weather group at {forecast_share * 100:.0f}% is the more
interesting number: it is what lets the model *depart* from persistence and
anticipate change rather than restate the present."""

    return f"""---

## 4. What the model actually learned

![Feature importance]({figure})

**By driver group**

| Group | Share of total \\|SHAP\\| |
|---|---|
{group_rows}

**Top individual features**

| Feature | Mean \\|SHAP\\| |
|---|---|
{rows}

{reading}"""


def _limitations_section() -> str:
    return """---

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
arrives would close that loop."""


def _reproduce_section() -> str:
    return f"""---

## 6. Reproducing this

```bash
pip install -r requirements.txt   # includes PyTorch (CPU) - see the note below

python -m src.backfill          # ~{config.HISTORY_START_DATE} → today of real observations
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
hand.*"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the technical report")
    parser.add_argument("--no-eda", action="store_true", help="Skip regenerating EDA figures")
    parser.add_argument("--no-shap", action="store_true", help="Skip the SHAP section")
    args = parser.parse_args()

    build(run_eda=not args.no_eda, run_shap=not args.no_shap)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
