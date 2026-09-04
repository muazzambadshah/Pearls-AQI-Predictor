"""
Feature Pipeline - runs hourly.

Fetches the newest observations, upserts them into the feature store, and
verifies that engineered features can still be built on top of the result. That
last check is the point of the job: a pipeline that silently stores unusable
rows is worse than one that fails loudly, because the breakage only surfaces a
day later when training produces a nonsense model.

Scheduled by .github/workflows/feature_pipeline.yml.

Usage
-----
    python -m src.feature_pipeline
    python -m src.feature_pipeline --synthetic   # offline smoke test
"""
from __future__ import annotations

import argparse
import logging

from src import config, data_sources, feature_engineering, feature_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(synthetic: bool = False, past_days: int = 14) -> dict:
    logger.info("Feature pipeline: %s (%.4f, %.4f)",
                config.CITY_NAME, config.LATITUDE, config.LONGITUDE)

    if synthetic:
        logger.info("SYNTHETIC mode - no network calls")
        observed = data_sources.generate_synthetic_data(days=past_days)
    else:
        observed, future_weather = data_sources.fetch_recent_and_forecast(
            past_days=past_days, forecast_days=config.FORECAST_HORIZON_DAYS
        )
        logger.info("Weather forecast available for %d future hours", len(future_weather))

    target = config.TARGET_COLUMN
    observed = observed[observed[target].notna()]
    if observed.empty:
        raise RuntimeError("Fetched no rows carrying an AQI value")

    logger.info("Fetched %d observed rows: %s -> %s",
                len(observed), observed.index.min(), observed.index.max())

    store = feature_store.get_feature_store()
    total_rows = store.write_observations(observed)

    # Sanity check: features must still be derivable from what is now stored.
    history = store.read_observations()
    origin_features = feature_engineering.build_origin_features(history).dropna()
    if origin_features.empty:
        raise RuntimeError(
            "Stored observations cannot produce a complete feature row - "
            "the history is too short or has too many gaps."
        )

    latest_aqi = float(history[target].iloc[-1])
    summary = {
        "rows_fetched": int(len(observed)),
        "rows_in_store": int(total_rows),
        "latest_observation": str(history.index.max()),
        "latest_aqi": latest_aqi,
        "usable_feature_rows": int(len(origin_features)),
    }
    logger.info("Feature pipeline complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly AQI feature pipeline")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--past-days", type=int, default=14)
    args = parser.parse_args()
    run(synthetic=args.synthetic, past_days=args.past_days)


if __name__ == "__main__":
    main()
