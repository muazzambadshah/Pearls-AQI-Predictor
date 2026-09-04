"""
Feature Store abstraction.

Design note
-----------
The store holds **raw hourly observations**, not engineered features. That is a
deliberate reversal of the obvious approach, and it matters:

  * Feature logic changes constantly during development. If the store held
    engineered columns, every tweak to a lag window would require re-fetching
    three years of history from the API.
  * Serving needs the raw series anyway, to recompute lags over a window that
    straddles the boundary between stored history and a live fetch.

So raw observations are the single source of truth, and `materialise_features`
derives the supervised table on demand. The engineered view is cached to parquet
purely as an optimisation - deleting it is always safe.

Backends
--------
`local` (default) writes parquet to disk and needs no accounts at all.
`hopsworks` talks to a real Hopsworks Feature Store. Both expose the same
`write_observations` / `read_observations` pair, so nothing downstream changes.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class LocalFeatureStore:
    """Parquet-backed store. Zero setup, works offline, good enough for one city."""

    def __init__(self, path=None):
        self.path = path or config.RAW_OBSERVATIONS_PATH

    def write_observations(self, df: pd.DataFrame) -> int:
        """Upsert hourly rows by timestamp. Returns the total row count held."""
        df = df.sort_index()
        df.index.name = "timestamp"

        if self.path.exists():
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, df])
            # `keep="last"` makes the newest fetch win, so a revised reanalysis
            # value quietly replaces the earlier real-time estimate.
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = df

        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path)
        logger.info("Feature store: wrote %d rows, %d total, span %s -> %s",
                    len(df), len(combined), combined.index.min(), combined.index.max())
        return len(combined)

    def read_observations(self, start=None, end=None) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"No observations at {self.path}. Run the backfill first:\n"
                f"    python -m src.backfill"
            )
        df = pd.read_parquet(self.path).sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def exists(self) -> bool:
        return self.path.exists()


class HopsworksFeatureStore:
    """
    Hopsworks-backed store, for the managed-feature-store deployment path.

    Requires `pip install hopsworks` plus HOPSWORKS_API_KEY / HOPSWORKS_PROJECT.
    Kept behind a lazy import so `hopsworks` never becomes a hard dependency.
    """

    FEATURE_GROUP_NAME = "aqi_raw_observations"
    FEATURE_GROUP_VERSION = 1

    def __init__(self):
        import hopsworks

        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY or None,
            project=config.HOPSWORKS_PROJECT or None,
        )
        self.fs = project.get_feature_store()

    def _feature_group(self):
        return self.fs.get_or_create_feature_group(
            name=self.FEATURE_GROUP_NAME,
            version=self.FEATURE_GROUP_VERSION,
            description="Hourly weather + pollutant observations for AQI forecasting",
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=False,
            time_travel_format="HUDI",
        )

    def write_observations(self, df: pd.DataFrame) -> int:
        frame = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
        fg = self._feature_group()
        fg.insert(frame, write_options={"wait_for_job": True})
        logger.info("Hopsworks: inserted %d rows into '%s'", len(frame), self.FEATURE_GROUP_NAME)
        return len(frame)

    def read_observations(self, start=None, end=None) -> pd.DataFrame:
        fg = self.fs.get_feature_group(self.FEATURE_GROUP_NAME,
                                       version=self.FEATURE_GROUP_VERSION)
        df = fg.read(read_options={"use_hive": True})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def exists(self) -> bool:
        try:
            self.fs.get_feature_group(self.FEATURE_GROUP_NAME,
                                      version=self.FEATURE_GROUP_VERSION)
            return True
        except Exception:  # noqa: BLE001 - absence is not an error here
            return False


_STORE_CACHE: dict = {}


def get_feature_store():
    """
    Factory - returns the configured backend, cached per backend.

    The cache matters specifically for Hopsworks: constructing
    `HopsworksFeatureStore` calls `hopsworks.login()`, a real network round
    trip. Without caching, every dashboard refresh, every `/health` check and
    every forecast call would each open a fresh login - the dashboard here
    polls every 60s, so that's a login roughly once a minute, all day, for no
    reason. The local backend is cheap either way, but is cached too so both
    paths behave the same.
    """
    backend = config.FEATURE_STORE_BACKEND
    if backend not in _STORE_CACHE:
        _STORE_CACHE[backend] = (
            HopsworksFeatureStore() if backend == "hopsworks" else LocalFeatureStore()
        )
    return _STORE_CACHE[backend]


def read_observations(start=None, end=None) -> pd.DataFrame:
    """Convenience wrapper used across the pipelines."""
    return get_feature_store().read_observations(start=start, end=end)


def describe() -> dict:
    """Summary of what the store currently holds - surfaced by the API and UI."""
    store = get_feature_store()
    if not store.exists():
        return {"available": False, "rows": 0}
    df = store.read_observations()
    target = config.TARGET_COLUMN
    return {
        "available": True,
        "backend": config.FEATURE_STORE_BACKEND,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "aqi_coverage": float(df[target].notna().mean()) if target in df else 0.0,
    }