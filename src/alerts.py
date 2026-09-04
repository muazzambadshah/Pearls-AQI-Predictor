"""
Hazardous-AQI alerting.

Turns the forecast into something actionable: scan the horizon for hours above a
threshold, group consecutive hours into episodes, and describe each one in the
terms a person actually needs - when it starts, how long it lasts, how bad it
gets, and what to do about it.

Grouping matters. Ninety separate "AQI is high at 14:00, AQI is high at 15:00"
notifications for one smog event is noise that trains people to ignore alerts.
One "unhealthy air expected Thursday 09:00-21:00, peaking at 186" is a warning
someone can act on.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config
from src.predict import aqi_category, health_advice

logger = logging.getLogger(__name__)

# Severity tiers, most severe first, so the first match wins.
SEVERITY_TIERS = [
    (300, "emergency"),
    (200, "critical"),
    (150, "warning"),
    (100, "advisory"),
]


def severity_for(aqi: float) -> str:
    for threshold, level in SEVERITY_TIERS:
        if aqi >= threshold:
            return level
    return "none"


def find_episodes(pred_df: pd.DataFrame,
                  threshold: float | None = None,
                  min_duration_hours: int = 2,
                  value_column: str = "predicted_aqi") -> list[dict]:
    """
    Group forecast hours above `threshold` into contiguous episodes.

    `min_duration_hours` suppresses single-hour blips, which are usually model
    noise around the threshold rather than a real air-quality event.
    """
    threshold = threshold if threshold is not None else config.ALERT_AQI_THRESHOLD
    if pred_df.empty or value_column not in pred_df.columns:
        return []

    series = pred_df[value_column].sort_index()
    above = series >= threshold
    if not above.any():
        return []

    # Each maximal run of True values gets its own group id.
    group_id = (above != above.shift()).cumsum()

    episodes = []
    for _, group in series.groupby(group_id):
        if not above.loc[group.index].iloc[0]:
            continue
        if len(group) < min_duration_hours:
            continue

        peak_value = float(group.max())
        peak_time = group.idxmax()
        label, colour = aqi_category(peak_value)

        episodes.append({
            "start": str(group.index.min()),
            "end": str(group.index.max()),
            "duration_hours": int(len(group)),
            "peak_aqi": round(peak_value, 1),
            "peak_time": str(peak_time),
            "mean_aqi": round(float(group.mean()), 1),
            "category": label,
            "color": colour,
            "severity": severity_for(peak_value),
            "advice": health_advice(peak_value),
            "threshold": float(threshold),
        })

    return episodes


def build_alert(pred_df: pd.DataFrame, threshold: float | None = None) -> dict:
    """Full alert payload for the API, dashboard, and webhook."""
    threshold = threshold if threshold is not None else config.ALERT_AQI_THRESHOLD
    episodes = find_episodes(pred_df, threshold=threshold)

    if not episodes:
        return {
            "alert": False,
            "threshold": float(threshold),
            "city": config.CITY_NAME,
            "message": f"No AQI above {threshold:.0f} expected in the next "
                       f"{config.FORECAST_HORIZON_DAYS} days.",
            "episodes": [],
        }

    worst = max(episodes, key=lambda e: e["peak_aqi"])
    start = pd.Timestamp(worst["start"])

    return {
        "alert": True,
        "threshold": float(threshold),
        "city": config.CITY_NAME,
        "severity": worst["severity"],
        "episode_count": len(episodes),
        "message": (
            f"{worst['category']} air expected in {config.CITY_NAME} from "
            f"{start.strftime('%a %d %b %H:%M')}, peaking at AQI "
            f"{worst['peak_aqi']:.0f} over {worst['duration_hours']}h."
        ),
        "advice": worst["advice"],
        "episodes": episodes,
    }


def send_webhook(payload: dict, url: str | None = None) -> bool:
    """
    POST the alert to a webhook (Slack-compatible).

    Returns True on success. Never raises: an alerting failure must not take
    down the pipeline that produced the forecast.
    """
    url = url or config.ALERT_WEBHOOK_URL
    if not url:
        logger.debug("No ALERT_WEBHOOK_URL configured - skipping webhook")
        return False

    try:
        import requests

        body = {"text": f"[{payload.get('severity', 'info').upper()}] {payload['message']}\n"
                        f"{payload.get('advice', '')}"}
        resp = requests.post(url, json=body, timeout=15)
        resp.raise_for_status()
        logger.info("Alert webhook delivered")
        return True
    except Exception as exc:  # noqa: BLE001 - alerting is best-effort
        logger.warning("Alert webhook failed: %s", exc)
        return False


def check_and_notify(pred_df: pd.DataFrame | None = None,
                     threshold: float | None = None,
                     notify: bool = True) -> dict:
    """Evaluate the current forecast and dispatch a webhook if warranted."""
    if pred_df is None:
        from src import predict

        pred_df = predict.load_cached_forecast()
        if pred_df is None:
            pred_df = predict.forecast()

    payload = build_alert(pred_df, threshold=threshold)
    if payload["alert"]:
        logger.warning("ALERT: %s", payload["message"])
        if notify:
            send_webhook(payload)
    else:
        logger.info(payload["message"])
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import json

    print(json.dumps(check_and_notify(), indent=2))


if __name__ == "__main__":
    main()
