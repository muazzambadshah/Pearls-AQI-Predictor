"""
Historical backfill.

Pulls the full observation record from Open-Meteo's archive endpoints into the
feature store, giving the training pipeline a real multi-year dataset instead of
whatever the hourly job has happened to accumulate.

The window starts at 2023-01-01 because that is where Open-Meteo's air-quality
archive begins reporting a populated `us_aqi` field - earlier dates return rows
with the field present but null.

Usage
-----
    python -m src.backfill                       # 2023-01-01 -> today
    python -m src.backfill --start 2024-01-01
    python -m src.backfill --synthetic --days 180   # offline, no network
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src import config, data_sources, feature_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(start: str | None = None,
        end: str | None = None,
        synthetic: bool = False,
        synthetic_days: int = 400) -> pd.DataFrame:
    if synthetic:
        logger.info("Generating %d days of synthetic observations (offline mode)", synthetic_days)
        raw = data_sources.generate_synthetic_data(days=synthetic_days)
    else:
        start = start or config.HISTORY_START_DATE
        logger.info("Backfilling %s from %s to %s", config.CITY_NAME, start, end or "today")
        raw = data_sources.fetch_history(start_date=start, end_date=end)

    target = config.TARGET_COLUMN
    before = len(raw)
    raw = raw[raw[target].notna()]
    if before != len(raw):
        logger.info("Dropped %d rows with no AQI reading", before - len(raw))

    if raw.empty:
        raise RuntimeError("Backfill produced no usable rows")

    store = feature_store.get_feature_store()
    total = store.write_observations(raw)

    span_days = (raw.index.max() - raw.index.min()).total_seconds() / 86400
    logger.info("Backfill complete: %d new rows, %d in store, %.0f days of history",
                len(raw), total, span_days)
    _report_gaps(raw)
    return raw


def _report_gaps(raw: pd.DataFrame) -> None:
    """
    Flag holes in the hourly grid.

    Gaps are not fatal - the feature builder drops origins whose lag windows are
    incomplete - but a large gap quietly shrinks the usable training set, so it
    is worth surfacing rather than discovering later as a mystery row count.
    """
    expected = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    missing = len(expected) - len(raw.index.unique())
    if missing > 0:
        pct = 100 * missing / len(expected)
        logger.warning("%d of %d hourly slots missing (%.2f%%)", missing, len(expected), pct)
    else:
        logger.info("Hourly grid is complete - no gaps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical AQI observations")
    parser.add_argument("--start", default=None, help="ISO start date (default: 2023-01-01)")
    parser.add_argument("--end", default=None, help="ISO end date (default: today)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate offline synthetic data instead of calling the API")
    parser.add_argument("--days", type=int, default=400,
                        help="Days of synthetic data when --synthetic is set")
    args = parser.parse_args()
    run(start=args.start, end=args.end, synthetic=args.synthetic, synthetic_days=args.days)


if __name__ == "__main__":
    main()
