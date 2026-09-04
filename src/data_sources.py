"""
Raw data acquisition from external APIs.

Everything returns the same tidy shape - an hourly-indexed DataFrame of raw
observations - so the rest of the pipeline never has to care which provider or
endpoint the numbers came from.

Open-Meteo exposes three endpoints we care about, and they cover different
slices of the timeline:

  * archive-api.open-meteo.com/v1/archive      - reanalysis weather, but lags
                                                 real time by roughly 5 days.
  * air-quality-api.open-meteo.com/v1/air-quality
                                               - pollutants + `us_aqi`, covers
                                                 both history and a short
                                                 forecast in one endpoint.
  * api.open-meteo.com/v1/forecast             - weather for the recent past
                                                 (`past_days`) *and* the future
                                                 (`forecast_days`).

`fetch_history` stitches the first two together for bulk backfill and then tops
up the archive's trailing gap from the forecast endpoint, so the resulting table
runs right up to the current hour with no hole in the middle.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AQICN_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/"

# Weather variables. Every one of these was verified to return non-null data on
# BOTH the archive and the forecast endpoint - which is a stricter test than it
# sounds. `boundary_layer_height` and the pressure-level temperatures
# (`temperature_925hPa`, `temperature_850hPa`) all answer 200 on the archive but
# hand back a column of nulls, so a model trained with them would fit on the
# handful of recent rows that have them and then diverge in production. The
# 100m wind pair replaces boundary-layer height as the dispersion signal: shear
# between 10m and 100m carries much the same information about vertical mixing
# and is genuinely present across the whole history.
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "vapour_pressure_deficit",
]

# Both endpoints serve the identical set, which keeps train and serve aligned.
ARCHIVE_WEATHER_VARS = list(WEATHER_VARS)

AIR_QUALITY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

# Columns that describe the atmosphere and are genuinely available ahead of time
# from a weather forecast. These are the only "future" inputs the model may use.
FORECASTABLE_WEATHER = list(WEATHER_VARS)

# Pollutant columns - only ever known up to the forecast origin.
POLLUTANT_COLUMNS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                     "sulphur_dioxide", "ozone"]

_SESSION = requests.Session()
_MAX_RETRIES = 7
_BACKOFF_CAP = 60


def _get_json(url: str, params: dict, timeout: int = 120) -> dict:
    """
    GET with bounded exponential backoff.

    Retries generously rather than minimally: a bulk backfill issues dozens of
    requests, so a single transient drop anywhere in the sequence would
    otherwise abort an operation that takes minutes to restart. Backoff is
    capped so a genuine outage still fails in reasonable time.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)

            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, _BACKOFF_CAP)
                logger.warning("Rate limited by %s; sleeping %ss", url, wait)
                time.sleep(wait)
                continue

            # A 4xx is a deterministic complaint about the request itself -
            # a bad date range, an unknown variable. Retrying cannot change the
            # answer, and doing so buries the API's actual explanation under a
            # generic "failed after N attempts". Surface it immediately.
            if 400 <= resp.status_code < 500:
                raise RuntimeError(
                    f"{url} rejected the request ({resp.status_code}): {resp.text[:300]}"
                )

            resp.raise_for_status()
            return resp.json()

        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry transport/5xx errors only
            last_exc = exc
            wait = min(2 ** attempt, _BACKOFF_CAP)
            logger.warning("Request to %s failed (%s); retry %d/%d in %ss",
                           url, exc, attempt + 1, _MAX_RETRIES, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} after {_MAX_RETRIES} attempts") from last_exc


def archive_end_cap() -> date:
    """
    Latest date the archive endpoints will serve.

    Reanalysis is only published once a day has closed, so the archive rejects
    any `end_date` of today with a 400. Everything from that cap up to the
    current hour is filled from the forecast endpoint's `past_days` window
    instead - see `fetch_history`.
    """
    return date.today() - timedelta(days=1)


def _hourly_frame(payload: dict, rename: dict | None = None) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "timestamp", **(rename or {})})
    df = df.set_index("timestamp").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def _chunk_ranges(start: date, end: date, days: int = 180):
    """Split a long date range into API-friendly chunks."""
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


# ---------------------------------------------------------------------------
# Historical (archive) fetches
# ---------------------------------------------------------------------------
def fetch_air_quality_range(start: date, end: date) -> pd.DataFrame:
    end = min(end, archive_end_cap())
    frames = []
    for chunk_start, chunk_end in _chunk_ranges(start, end):
        logger.info("Air quality %s -> %s", chunk_start, chunk_end)
        payload = _get_json(AIR_QUALITY_URL, {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(AIR_QUALITY_VARS),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": config.TIMEZONE,
        })
        frames.append(_hourly_frame(payload, rename={"us_aqi": config.TARGET_COLUMN}))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def fetch_weather_range(start: date, end: date) -> pd.DataFrame:
    end = min(end, archive_end_cap())
    frames = []
    for chunk_start, chunk_end in _chunk_ranges(start, end):
        logger.info("Weather archive %s -> %s", chunk_start, chunk_end)
        payload = _get_json(WEATHER_ARCHIVE_URL, {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(ARCHIVE_WEATHER_VARS),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": config.TIMEZONE,
        })
        frames.append(_hourly_frame(payload))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Recent + future fetches
# ---------------------------------------------------------------------------
def fetch_weather_forecast(past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    """Recent-past and future weather. `past_days` covers the archive's lag."""
    payload = _get_json(WEATHER_FORECAST_URL, {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": min(past_days, 92),
        "forecast_days": min(forecast_days, 16),
        "timezone": config.TIMEZONE,
    })
    return _hourly_frame(payload)


def fetch_air_quality_forecast(past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    payload = _get_json(AIR_QUALITY_URL, {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(AIR_QUALITY_VARS),
        "past_days": min(past_days, 92),
        "forecast_days": min(forecast_days, 7),
        "timezone": config.TIMEZONE,
    })
    return _hourly_frame(payload, rename={"us_aqi": config.TARGET_COLUMN})


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def fetch_history(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """
    Full historical record of weather + pollutants + AQI, hourly, from
    `start_date` to now. Archive endpoints supply the bulk; the forecast
    endpoint's `past_days` window patches the ~5-day archive lag so the series
    is continuous up to the present hour.
    """
    start = pd.Timestamp(start_date or config.HISTORY_START_DATE).date()
    end = pd.Timestamp(end_date).date() if end_date else date.today()

    aq = fetch_air_quality_range(start, end)
    wx = fetch_weather_range(start, end)

    # The archive stops at yesterday, so both series are topped up from the
    # forecast endpoints' `past_days` window. Without this the training set
    # would silently end several days before "now", and the most recent - most
    # relevant - observations would never reach the model.
    def _top_up(frame: pd.DataFrame, fetcher, label: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        gap_days = (pd.Timestamp(end) - frame.index.max().normalize()).days
        if gap_days <= 0:
            return frame
        logger.info("%s archive lags %d day(s); topping up from forecast endpoint",
                    label, gap_days)
        recent = fetcher(past_days=min(gap_days + 5, 92), forecast_days=1)
        merged = pd.concat([frame, recent])
        return merged[~merged.index.duplicated(keep="last")].sort_index()

    wx = _top_up(wx, fetch_weather_forecast, "Weather")
    aq = _top_up(aq, fetch_air_quality_forecast, "Air quality")

    if aq.empty or wx.empty:
        raise RuntimeError("Received no data from Open-Meteo - check connectivity/parameters")

    df = wx.join(aq, how="inner").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # The top-up endpoints return the remainder of the current day, which is
    # forecast rather than observation. Those rows must not enter the store as
    # history: they would become training labels the model is asked to reproduce,
    # and the live persistence anchor would be a forecast of itself.
    now_hour = pd.Timestamp.now().floor("h")
    future_rows = int((df.index > now_hour).sum())
    if future_rows:
        logger.info("Dropping %d forecast row(s) beyond %s from the history",
                    future_rows, now_hour)
        df = df[df.index <= now_hour]

    df.index.name = "timestamp"
    return df


def fetch_recent_and_forecast(past_days: int = 14, forecast_days: int | None = None):
    """
    Returns `(observed, future_weather)`.

    `observed` is the recent hourly record (weather + pollutants + AQI) up to the
    latest hour that actually has an AQI reading. `future_weather` holds ONLY
    forecastable weather columns for hours after that - these are the legitimate
    known-ahead inputs the model is allowed to condition on.
    """
    forecast_days = forecast_days or config.FORECAST_HORIZON_DAYS
    wx = fetch_weather_forecast(past_days=past_days, forecast_days=forecast_days + 1)
    aq = fetch_air_quality_forecast(past_days=past_days, forecast_days=min(forecast_days + 1, 7))

    combined = wx.join(aq, how="left").sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    target = config.TARGET_COLUMN
    if target not in combined.columns or combined[target].notna().sum() == 0:
        raise RuntimeError("No AQI observations returned by the air-quality API")

    # The air-quality endpoint returns AQI for future hours too - those are CAMS
    # forecasts. Taking "last non-null AQI" as the origin would therefore place
    # the origin days ahead of now and then forecast 72h beyond *that*. The
    # origin has to be the last hour that has both actually happened and carries
    # a reading.
    now_hour = pd.Timestamp.now().floor("h")
    has_reading = combined[target].notna() & (combined.index <= now_hour)
    if not has_reading.any():
        raise RuntimeError(
            f"No AQI reading at or before the current hour ({now_hour}) - "
            "the air-quality API returned only forecast rows."
        )
    origin = combined.index[has_reading].max()

    observed = combined.loc[:origin].copy()
    future_weather = combined.loc[combined.index > origin, FORECASTABLE_WEATHER].copy()
    return observed, future_weather


def fetch_aqicn_current() -> pd.DataFrame:
    """Current reading from AQICN. Requires a free token; returns a single row."""
    if not config.AQICN_TOKEN:
        raise RuntimeError(
            "AQICN_TOKEN is not set. Get a free token at https://aqicn.org/api/ "
            "or keep DATA_SOURCE=open_meteo."
        )
    url = AQICN_URL.format(lat=config.LATITUDE, lon=config.LONGITUDE)
    payload = _get_json(url, {"token": config.AQICN_TOKEN})
    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN API error: {payload}")

    data = payload["data"]
    iaqi = data.get("iaqi", {})

    def _v(key):
        return iaqi.get(key, {}).get("v")

    row = {
        "timestamp": pd.to_datetime(data["time"]["s"]),
        config.TARGET_COLUMN: data.get("aqi"),
        "pm2_5": _v("pm25"), "pm10": _v("pm10"), "ozone": _v("o3"),
        "nitrogen_dioxide": _v("no2"), "sulphur_dioxide": _v("so2"),
        "carbon_monoxide": _v("co"), "temperature_2m": _v("t"),
        "relative_humidity_2m": _v("h"), "surface_pressure": _v("p"),
        "wind_speed_10m": _v("w"),
    }
    return pd.DataFrame([row]).set_index("timestamp")


# US EPA PM2.5 -> AQI breakpoints (2024 revision), expressed as contiguous
# concentration edges and their AQI anchors.
#
# The published table lists ranges as 0.0-9.0, 9.1-35.4, 35.5-55.4 and so on,
# because EPA truncates reported concentrations to one decimal place. Encoding
# those literally leaves gaps: a continuous value of 9.05 belongs to no range and
# maps to NaN. Using shared edges instead keeps the function total and monotonic,
# and reproduces the table exactly at every published breakpoint.
_PM25_EDGES = np.array([0.0, 9.0, 35.4, 55.4, 125.4, 225.4, 500.4])
_AQI_EDGES = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0])


def us_aqi_from_pm25(pm25):
    """Piecewise-linear US EPA AQI from a PM2.5 concentration (ug/m3)."""
    pm = np.asarray(pm25, dtype=float)
    # np.interp clamps outside the range, which is the behaviour we want at both
    # ends: negative concentrations read as 0 and anything above 500.4 pins at 500.
    return np.interp(pm, _PM25_EDGES, _AQI_EDGES)


def generate_synthetic_data(days: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic dataset with realistic diurnal/seasonal structure and
    genuine physical drivers (ventilation, rain washout, seasonal baseline).
    Used by the offline test suite so CI never depends on a live API.
    """
    periods = days * 24
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=periods, freq="h")
    rng = np.random.default_rng(seed)

    hour = idx.hour.values.astype(float)
    doy = idx.dayofyear.values.astype(float)

    temperature = (22 + 11 * np.sin(2 * np.pi * (hour - 7) / 24)
                   + 9 * np.sin(2 * np.pi * (doy - 100) / 365)
                   + rng.normal(0, 1.4, periods))
    humidity = np.clip(58 - 0.7 * (temperature - 22) + rng.normal(0, 6, periods), 8, 100)
    wind_speed = np.abs(6 + 3 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 2.2, periods))
    gusts = wind_speed * 1.7 + np.abs(rng.normal(0, 1.5, periods))
    pressure = 1013 + 6 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 2.5, periods)
    precipitation = np.clip(rng.exponential(0.25, periods) - 0.32, 0, None)
    cloud = np.clip(35 + 220 * precipitation + rng.normal(0, 18, periods), 0, 100)

    # Mixing depth drives the pollutant dynamics below but is deliberately NOT
    # emitted as a column - Open-Meteo's archive does not carry it either. It
    # stays a latent variable the model has to infer from wind shear, which is
    # exactly the situation the real feature set is in.
    mixing_depth = np.clip(350 + 700 * np.sin(np.pi * np.clip(hour - 6, 0, 12) / 12)
                           + 60 * wind_speed + rng.normal(0, 90, periods), 60, 2600)

    # Wind aloft runs faster than at the surface, and more so when the surface
    # layer is shallow and decoupled - which is what makes shear informative.
    shear_ratio = 1.25 + 0.5 * (800.0 / np.clip(mixing_depth, 100, None))
    wind_speed_100m = wind_speed * np.clip(shear_ratio, 1.0, 3.2) \
        + np.abs(rng.normal(0, 0.8, periods))
    wind_direction_10m = rng.uniform(0, 360, periods)
    # Ekman veer: direction rotates slightly clockwise with height.
    wind_direction_100m = (wind_direction_10m + rng.normal(12, 6, periods)) % 360

    # Vapour pressure deficit from saturation vapour pressure and humidity (kPa).
    svp = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    vpd = np.clip(svp * (1.0 - humidity / 100.0), 0, None)

    # PM2.5 accumulates when the boundary layer is shallow and wind is calm, and
    # washes out with rain - the same mechanism the real series shows.
    base = 45 + 55 * np.sin(2 * np.pi * (doy - 330) / 365)
    pm2_5 = np.zeros(periods)
    level = float(base[0])
    for i in range(periods):
        ventilation = (wind_speed[i] / 6.0) * (mixing_depth[i] / 800.0)
        level = (level + 0.14 * (base[i] - level) - 5.5 * (ventilation - 1.0)
                 - 14 * precipitation[i] + 6.5 * np.sin(2 * np.pi * (hour[i] - 8) / 24)
                 + rng.normal(0, 4.0))
        level = float(np.clip(level, 3, 480))
        pm2_5[i] = level

    pm10 = np.clip(pm2_5 * 1.75 + rng.normal(0, 9, periods), 4, 700)
    co = np.clip(0.35 + 0.011 * pm2_5 + rng.normal(0, 0.08, periods), 0.05, 6)
    no2 = np.clip(14 + 0.28 * pm2_5 - 0.6 * wind_speed + rng.normal(0, 5, periods), 1, 180)
    so2 = np.clip(6 + 0.045 * pm2_5 + rng.normal(0, 2, periods), 0.5, 70)
    ozone = np.clip(32 + 0.55 * temperature - 0.18 * pm2_5 + rng.normal(0, 8, periods), 1, 200)
    aqi = np.clip(us_aqi_from_pm25(pm2_5) + rng.normal(0, 3, periods), 5, 500)

    df = pd.DataFrame({
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "dew_point_2m": temperature - (100 - humidity) / 5.0,
        "apparent_temperature": temperature + 0.3 * (humidity - 50) / 10.0,
        "precipitation": precipitation,
        "surface_pressure": pressure,
        "cloud_cover": cloud,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_direction_10m,
        "wind_gusts_10m": gusts,
        "wind_speed_100m": wind_speed_100m,
        "wind_direction_100m": wind_direction_100m,
        "vapour_pressure_deficit": vpd,
        "pm10": pm10,
        "pm2_5": pm2_5,
        "carbon_monoxide": co,
        "nitrogen_dioxide": no2,
        "sulphur_dioxide": so2,
        "ozone": ozone,
        config.TARGET_COLUMN: aqi,
    }, index=idx)
    df.index.name = "timestamp"
    return df


def fetch_raw_data(past_days: int = 14, forecast_days: int | None = None) -> pd.DataFrame:
    """Provider-agnostic entry point used by the hourly feature pipeline."""
    if config.DATA_SOURCE == "aqicn":
        try:
            return fetch_aqicn_current()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AQICN fetch failed (%s); falling back to Open-Meteo", exc)
    observed, _ = fetch_recent_and_forecast(past_days=past_days, forecast_days=forecast_days)
    return observed
