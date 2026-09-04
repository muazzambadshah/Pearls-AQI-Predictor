"""
Exploratory data analysis.

Produces the statistics and figures that go into `reports/report.md`, and that
justify several of the modelling choices made elsewhere - the strength of the
autocorrelation (which is why persistence is such a demanding baseline), the
diurnal and seasonal structure (which is why the calendar features are cyclical),
and the meteorological relationships (which is why the weather forecast is worth
conditioning on at all).

Figures are written as PNGs under `reports/figures/`.
"""
from __future__ import annotations

import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless - this runs in CI
import matplotlib.pyplot as plt  # noqa: E402

from src import config, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BAND_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7", "#7f1d1d"]


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return f"figures/{name}.png"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summary_stats(raw: pd.DataFrame) -> dict:
    target = raw[config.TARGET_COLUMN]
    bands = evaluate.aqi_band_index(target.to_numpy())
    band_share = {
        evaluate.AQI_BANDS[i][2]: float((bands == i).mean())
        for i in range(len(evaluate.AQI_BANDS))
    }
    return {
        "rows": int(len(raw)),
        "start": str(raw.index.min()),
        "end": str(raw.index.max()),
        "days": round((raw.index.max() - raw.index.min()).total_seconds() / 86400, 1),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "p95": float(target.quantile(0.95)),
        "hours_unhealthy": int((target >= 150).sum()),
        "share_unhealthy": float((target >= 150).mean()),
        "band_share": band_share,
        "missing_hours": int(
            len(pd.date_range(raw.index.min(), raw.index.max(), freq="h")) - len(raw)
        ),
    }


def autocorrelation(raw: pd.DataFrame, max_lag: int = 168) -> pd.Series:
    """
    AQI autocorrelation out to a week.

    The decay curve is the single most important thing to understand about this
    problem: it sets how much a persistence forecast already achieves, and
    therefore how much room a model has to add value.
    """
    target = raw[config.TARGET_COLUMN].astype(float)
    lags = range(1, max_lag + 1)
    return pd.Series({lag: target.autocorr(lag) for lag in lags}, name="autocorr")


def diurnal_profile(raw: pd.DataFrame) -> pd.DataFrame:
    target = config.TARGET_COLUMN
    grouped = raw.groupby(raw.index.hour)[target]
    return pd.DataFrame({
        "mean": grouped.mean(), "std": grouped.std(),
        "p25": grouped.quantile(0.25), "p75": grouped.quantile(0.75),
    })


def monthly_profile(raw: pd.DataFrame) -> pd.DataFrame:
    target = config.TARGET_COLUMN
    grouped = raw.groupby(raw.index.month)[target]
    return pd.DataFrame({
        "mean": grouped.mean(), "std": grouped.std(), "max": grouped.max(),
        "share_unhealthy": raw.groupby(raw.index.month)[target].apply(lambda s: (s >= 150).mean()),
    })


def stationarity(raw: pd.DataFrame, sample_hours: int = 24 * 365) -> dict:
    """
    Augmented Dickey-Fuller test on the AQI level and its first difference.

    Worth running because it decides how the problem should be framed. If the
    level is non-stationary but the difference is not, the natural target is the
    *change* in AQI rather than the level - which is the reasoning behind the
    anchored model variants in `src/models.py`. The result here is what those
    variants were built to exploit, and the backtest is what judged whether the
    exploitation actually paid off.
    """
    from statsmodels.tsa.stattools import adfuller

    series = raw[config.TARGET_COLUMN].astype(float).dropna().tail(sample_hours)

    def _test(values, label):
        stat, pvalue, _, nobs, crit, _ = adfuller(values, autolag="AIC")
        return {
            "series": label,
            "adf_statistic": float(stat),
            "p_value": float(pvalue),
            "n_obs": int(nobs),
            "critical_5pct": float(crit["5%"]),
            "stationary_at_5pct": bool(pvalue < 0.05),
        }

    return {
        "level": _test(series.to_numpy(), "AQI level"),
        "first_difference": _test(series.diff().dropna().to_numpy(), "AQI first difference"),
    }


def plot_decomposition(raw: pd.DataFrame, days: int = 60) -> str:
    """
    STL decomposition of the recent series into trend, daily cycle and remainder.

    The useful thing to read off it is how much of the variance the remainder
    holds. That residual is the part no calendar feature can explain, and it is
    the only part a weather-driven model has any chance of predicting.
    """
    from statsmodels.tsa.seasonal import STL

    series = raw[config.TARGET_COLUMN].astype(float).tail(days * 24)
    result = STL(series, period=24, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for ax, values, label, colour in (
        (axes[0], series, "Observed", "#1e293b"),
        (axes[1], result.trend, "Trend", "#f59e0b"),
        (axes[2], result.seasonal, "Daily cycle", "#2563eb"),
        (axes[3], result.resid, "Remainder", "#7c3aed"),
    ):
        ax.plot(series.index, values, lw=1.1, color=colour)
        ax.set_ylabel(label, fontsize=8.5)
    axes[3].axhline(0, color="#94a3b8", lw=1)

    total_var = float(np.var(series.dropna()))
    resid_var = float(np.var(result.resid.dropna()))
    share = resid_var / total_var if total_var else float("nan")
    axes[0].set_title(f"STL decomposition, last {days} days - "
                      f"remainder holds {share * 100:.0f}% of the variance")
    return _save(fig, "eda_decomposition")


def weather_correlations(raw: pd.DataFrame) -> pd.Series:
    numeric = raw.select_dtypes(include=[np.number])
    corr = numeric.corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    return corr.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_timeseries(raw: pd.DataFrame) -> str:
    target = raw[config.TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(11, 3.6))

    for i, (low, high, _) in enumerate(evaluate.AQI_BANDS):
        ax.axhspan(low, min(high, 520), color=BAND_COLORS[i], alpha=0.13, zorder=0)

    ax.plot(raw.index, target, lw=0.35, color="#1e293b", alpha=0.75, label="Hourly AQI")
    ax.plot(raw.index, target.rolling(24 * 7).mean(), lw=2, color="#f59e0b",
            label="7-day mean")
    ax.set_ylim(0, min(520, target.max() * 1.05))
    ax.set_ylabel("US AQI")
    ax.set_title(f"{config.CITY_NAME} AQI, {raw.index.min():%b %Y} to {raw.index.max():%b %Y}")
    ax.legend(loc="upper left", framealpha=0.9)
    return _save(fig, "eda_timeseries")


def plot_distribution(raw: pd.DataFrame) -> str:
    target = raw[config.TARGET_COLUMN]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))

    ax1.hist(target, bins=70, color="#2563eb", alpha=0.85)
    ax1.axvline(target.mean(), color="#ef4444", ls="--", lw=1.6,
                label=f"mean {target.mean():.0f}")
    ax1.axvline(150, color="#111827", ls=":", lw=1.6, label="unhealthy (150)")
    ax1.set_xlabel("US AQI")
    ax1.set_ylabel("Hours")
    ax1.set_title("Distribution of hourly AQI")
    ax1.legend()

    bands = evaluate.aqi_band_index(target.to_numpy())
    counts = pd.Series(bands).value_counts().sort_index()
    labels = [evaluate.AQI_BANDS[i][2].replace(" for Sensitive Groups", "\n(sensitive)")
              for i in counts.index]
    ax2.bar(range(len(counts)), counts.values / counts.sum() * 100,
            color=[BAND_COLORS[i] for i in counts.index])
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax2.set_ylabel("Share of hours (%)")
    ax2.set_title("Time spent in each EPA category")
    return _save(fig, "eda_distribution")


def plot_cycles(raw: pd.DataFrame) -> str:
    diurnal = diurnal_profile(raw)
    monthly = monthly_profile(raw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))

    ax1.fill_between(diurnal.index, diurnal["p25"], diurnal["p75"],
                     color="#2563eb", alpha=0.20, label="IQR")
    ax1.plot(diurnal.index, diurnal["mean"], lw=2.4, color="#2563eb", label="Mean")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("US AQI")
    ax1.set_title("Daily cycle")
    ax1.set_xticks(range(0, 24, 3))
    ax1.legend()

    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colours = [BAND_COLORS[int(evaluate.aqi_band_index([v])[0])] for v in monthly["mean"]]
    ax2.bar([names[m - 1] for m in monthly.index], monthly["mean"], color=colours)
    ax2.axhline(150, color="#111827", ls=":", lw=1.4)
    ax2.set_ylabel("Mean US AQI")
    ax2.set_title("Seasonal cycle")
    ax2.tick_params(axis="x", rotation=45)
    return _save(fig, "eda_cycles")


def plot_autocorrelation(raw: pd.DataFrame) -> str:
    acf = autocorrelation(raw, max_lag=168)
    fig, ax = plt.subplots(figsize=(11, 3.2))

    ax.plot(acf.index, acf.values, lw=2, color="#7c3aed")
    ax.axhline(0, color="#94a3b8", lw=1)
    for lag in (24, 48, 72):
        ax.axvline(lag, color="#f59e0b", ls=":", lw=1.2)
        ax.annotate(f"{lag}h\n{acf.loc[lag]:.2f}", (lag, acf.loc[lag]),
                    textcoords="offset points", xytext=(6, 10), fontsize=8,
                    color="#b45309")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("AQI autocorrelation - why persistence is a demanding baseline")
    return _save(fig, "eda_autocorrelation")


def plot_weather_relationships(raw: pd.DataFrame) -> str:
    corr = weather_correlations(raw)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    colours = ["#ef4444" if v < 0 else "#2563eb" for v in corr.values]
    ax1.barh(range(len(corr)), corr.values, color=colours)
    ax1.set_yticks(range(len(corr)))
    ax1.set_yticklabels(corr.index, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.axvline(0, color="#111827", lw=1)
    ax1.set_xlabel("Pearson correlation with AQI")
    ax1.set_title("What moves with air quality")

    if "wind_speed_10m" in raw.columns:
        wind_bins = pd.cut(raw["wind_speed_10m"], bins=[0, 4, 8, 12, 16, 20, 100])
        grouped = raw.groupby(wind_bins, observed=True)[config.TARGET_COLUMN].mean()
        ax2.bar(range(len(grouped)), grouped.values, color="#0ea5e9")
        ax2.set_xticks(range(len(grouped)))
        ax2.set_xticklabels([str(i) for i in grouped.index], rotation=35,
                            ha="right", fontsize=7.5)
        ax2.set_xlabel("Wind speed (km/h)")
        ax2.set_ylabel("Mean US AQI")
        ax2.set_title("Calm air traps pollution")
    return _save(fig, "eda_weather")


def plot_horizon_accuracy(horizon_metrics: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)

    ax1.plot(horizon_metrics["horizon_h"], horizon_metrics["rmse"],
             lw=2.4, color="#2563eb", label="Model")
    if "persistence_rmse" in horizon_metrics.columns:
        ax1.plot(horizon_metrics["horizon_h"], horizon_metrics["persistence_rmse"],
                 lw=2, ls="--", color="#94a3b8", label="Persistence")
    ax1.set_ylabel("RMSE (AQI)")
    ax1.set_title("Forecast error by lead time")
    ax1.legend()

    if "skill_vs_persistence" in horizon_metrics.columns:
        skill = horizon_metrics["skill_vs_persistence"] * 100
        ax2.fill_between(horizon_metrics["horizon_h"], 0, skill,
                         color="#16a34a", alpha=0.35)
        ax2.plot(horizon_metrics["horizon_h"], skill, lw=2, color="#16a34a")
        ax2.axhline(0, color="#ef4444", ls=":", lw=1.4)
    for day in (24, 48):
        ax2.axvline(day, color="#cbd5e1", lw=1)
    ax2.set_xlabel("Lead time (hours)")
    ax2.set_ylabel("Skill vs persistence (%)")
    ax2.set_title("Share of persistence error removed")
    return _save(fig, "accuracy_by_horizon")


def plot_model_comparison(comparison: pd.DataFrame) -> str:
    frame = comparison.sort_values("rmse", ascending=False)
    colours = ["#94a3b8" if str(m).startswith("baseline:") else "#2563eb"
               for m in frame["model"]]

    fig, ax = plt.subplots(figsize=(9, 0.36 * len(frame) + 1.4))
    bars = ax.barh(range(len(frame)), frame["rmse"], color=colours)
    if "rmse_std" in frame.columns:
        ax.errorbar(frame["rmse"], range(len(frame)),
                    xerr=frame["rmse_std"].fillna(0), fmt="none",
                    ecolor="#334155", capsize=3, lw=1)

    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["model"], fontsize=8)
    ax.set_xlabel("Backtest RMSE (lower is better)")
    ax.set_title("Model comparison - mean across walk-forward folds")

    for bar, value in zip(bars, frame["rmse"]):
        ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", va="center", fontsize=7.5)
    return _save(fig, "model_comparison")


def plot_forecast_vs_actual(predictions: pd.DataFrame, model_name: str) -> str:
    """Predicted against observed for the winning model, split by forecast day."""
    frame = predictions[predictions["model"] == model_name] \
        if "model" in predictions.columns else predictions

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, sharex=True)
    for ax, day in zip(axes, (1, 2, 3)):
        subset = frame[(frame["horizon_h"] > (day - 1) * 24) & (frame["horizon_h"] <= day * 24)]
        if subset.empty:
            continue
        sample = subset.sample(min(4000, len(subset)), random_state=0)
        ax.scatter(sample["y"], sample["y_pred"], s=2.5, alpha=0.20, color="#2563eb")

        lims = [0, max(sample["y"].max(), sample["y_pred"].max()) * 1.03]
        ax.plot(lims, lims, color="#ef4444", lw=1.4, ls="--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        metrics = evaluate.regression_metrics(subset["y"], subset["y_pred"])
        ax.set_title(f"Day {day}  ·  R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.1f}",
                     fontsize=9)
        ax.set_xlabel("Observed AQI")
    axes[0].set_ylabel("Predicted AQI")
    return _save(fig, "forecast_vs_actual")


def plot_feature_importance(importance: pd.DataFrame, groups: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    ax1.barh(range(len(groups)), groups["share"] * 100, color="#2563eb")
    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels(groups["group"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Share of total |SHAP| (%)")
    ax1.set_title("Drivers by group")

    top = importance.head(18)
    ax2.barh(range(len(top)), top["importance"], color="#7c3aed")
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top["feature"], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |SHAP|")
    ax2.set_title("Top individual features")
    return _save(fig, "feature_importance")


def run_full_eda(raw: pd.DataFrame) -> dict:
    """Generate every EDA figure and return the stats bundle plus figure paths."""
    logger.info("Running EDA on %d observations", len(raw))

    figures = {
        "timeseries": plot_timeseries(raw),
        "distribution": plot_distribution(raw),
        "cycles": plot_cycles(raw),
        "autocorrelation": plot_autocorrelation(raw),
        "weather": plot_weather_relationships(raw),
    }

    # STL and ADF are the slowest parts and the least essential, so a failure
    # here degrades the report rather than aborting it.
    stationarity_result = {}
    try:
        figures["decomposition"] = plot_decomposition(raw)
        stationarity_result = stationarity(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Decomposition/stationarity skipped: %s", exc)

    return {
        "stats": summary_stats(raw),
        "correlations": weather_correlations(raw),
        "autocorr": autocorrelation(raw, max_lag=168),
        "diurnal": diurnal_profile(raw),
        "monthly": monthly_profile(raw),
        "stationarity": stationarity_result,
        "figures": figures,
    }
