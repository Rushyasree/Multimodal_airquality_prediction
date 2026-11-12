"""
src/data/resilient_ingestion.py

Phase 17.3: Resilient Data Ingestion Manager.
Provides:
1. Bounded retries with exponential backoff and jitter for transient API failures.
2. Station-level concurrency locking to prevent duplicate concurrent ingestion.
3. Upstream rate limiting and request pacing.
4. Response caching with explicit cache-age tracking.
5. Preservation of last-known valid physical observations during outages (marked STALE).
6. Zero generation of fabricated or fake live observations.
7. Absolute secret redaction in logging and error handling.
"""

from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timezone, timedelta
import threading
import time
import random
import logging
import re
import requests

from src.services.operational_health_service import get_operational_health_service

logger = logging.getLogger("VayuDrishti.ResilientIngestion")

SECRET_PATTERN = re.compile(r"([0-9a-fA-F]{32,64}|[a-zA-Z0-9_\-]{30,})")


def sanitize_text(text: str) -> str:
    """Masks secret API tokens or credentials."""
    if not isinstance(text, str):
        return str(text)
    return SECRET_PATTERN.sub("[REDACTED_CREDENTIAL]", text)


class ResilientDataIngestionManager:
    """Manages resilient network queries, rate-limiting, and cached physical observations."""

    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(ResilientDataIngestionManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
        request_timeout_seconds: float = 8.0,
        rate_limit_delay_seconds: float = 0.5,
        max_cache_age_seconds: float = 1800.0,  # 30 minutes cache TTL
    ):
        if getattr(self, "_initialized", False):
            return

        self.max_retries = int(max_retries)
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.rate_limit_delay_seconds = float(rate_limit_delay_seconds)
        self.max_cache_age_seconds = float(max_cache_age_seconds)

        self._station_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._last_request_time: float = 0.0

        # Memory cache: station_id -> { "data": ..., "cached_at_utc": datetime, "is_stale": bool }
        self._cached_observations: Dict[str, Dict[str, Any]] = {}
        # Preserved last-valid observation store for outage resilience
        self._preserved_last_valid: Dict[str, Dict[str, Any]] = {}

        self.health_service = get_operational_health_service()
        self._initialized = True

    def _get_station_lock(self, station_id: str) -> threading.Lock:
        clean_sid = str(station_id).strip().upper()
        with self._global_lock:
            if clean_sid not in self._station_locks:
                self._station_locks[clean_sid] = threading.Lock()
            return self._station_locks[clean_sid]

    def _apply_rate_limit(self) -> None:
        """Paces requests to avoid exceeding upstream OpenAQ / Open-Meteo rate limits."""
        with self._global_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.rate_limit_delay_seconds:
                sleep_time = self.rate_limit_delay_seconds - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def execute_with_retry(
        self,
        target_fn: Callable[[], Any],
        service_name: str = "OpenAQ_v3",
        max_attempts: Optional[int] = None,
    ) -> Any:
        """Executes a network or API call with bounded exponential backoff and jitter."""
        attempts = max_attempts or self.max_retries
        last_exception = None

        for attempt in range(1, attempts + 1):
            self._apply_rate_limit()
            start_t = time.perf_counter()
            try:
                result = target_fn()
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                self.health_service.record_api_call(service_name, success=True, latency_ms=latency_ms)
                return result
            except (requests.RequestException, TimeoutError, ConnectionError, IOError) as exc:
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                err_clean = sanitize_text(str(exc))
                last_exception = exc
                self.health_service.record_api_call(service_name, success=False, latency_ms=latency_ms, error_msg=err_clean)

                if attempt < attempts:
                    # Exponential backoff with random jitter: base * 2^(attempt-1) + jitter
                    backoff = self.base_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"Transient failure in {service_name} (attempt {attempt}/{attempts}): "
                        f"{err_clean}. Retrying in {backoff:.2f}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"All {attempts} attempts failed for {service_name}: {err_clean}."
                    )
            except Exception as unrecoverable_exc:
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                err_clean = sanitize_text(str(unrecoverable_exc))
                self.health_service.record_api_call(service_name, success=False, latency_ms=latency_ms, error_msg=err_clean)
                raise unrecoverable_exc

        raise RuntimeError(f"{service_name} failed after {attempts} attempts: {sanitize_text(str(last_exception))}")

    def get_station_observation(
        self,
        station_id: str,
        fetcher_fn: Callable[[str], Dict[str, Any]],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Retrieves station observation with concurrency isolation, caching, and outage fallback.

        Never returns fake or fabricated readings.
        """
        clean_sid = str(station_id).strip().upper()
        station_lock = self._get_station_lock(clean_sid)

        with station_lock:
            now_utc = datetime.now(timezone.utc)

            # 1. Check valid cached observation
            if not force_refresh and clean_sid in self._cached_observations:
                cached_entry = self._cached_observations[clean_sid]
                cache_age = (now_utc - cached_entry["cached_at_utc"]).total_seconds()

                if cache_age <= self.max_cache_age_seconds:
                    res = dict(cached_entry["data"])
                    res["is_cached"] = True
                    res["cache_age_seconds"] = round(cache_age, 1)
                    return res

            # 2. Ingest via execute_with_retry
            try:
                fresh_data = self.execute_with_retry(
                    lambda: fetcher_fn(clean_sid),
                    service_name="OpenAQ_v3",
                )

                # Store in cache and preserved store
                self._cached_observations[clean_sid] = {
                    "data": fresh_data,
                    "cached_at_utc": now_utc,
                }
                self._preserved_last_valid[clean_sid] = {
                    "data": fresh_data,
                    "preserved_at_utc": now_utc,
                }
                self.health_service.update_cache_timestamp()

                res = dict(fresh_data)
                res["is_cached"] = False
                res["cache_age_seconds"] = 0.0
                return res

            except Exception as e:
                err_msg = sanitize_text(str(e))
                logger.warning(f"Live retrieval failed for {clean_sid}: {err_msg}")

                # 3. Fallback to preserved last-known valid observation during temporary outage
                if clean_sid in self._preserved_last_valid:
                    preserved = self._preserved_last_valid[clean_sid]
                    pres_age = (now_utc - preserved["preserved_at_utc"]).total_seconds()
                    res = dict(preserved["data"])
                    res["is_cached"] = True
                    res["is_outage_preserved"] = True
                    res["cache_age_seconds"] = round(pres_age, 1)
                    res["freshness_status"] = "STALE (Preserved During Outage)"
                    res["outage_notice"] = f"Upstream source temporary outage: {err_msg}. Showing last valid observation."
                    return res

                # 4. If no previous observation exists, fail gracefully with UNAVAILABLE
                return {
                    "result_type": "Observation",
                    "station_id": clean_sid,
                    "status": "unavailable",
                    "freshness_status": "UNAVAILABLE",
                    "pm25": None,
                    "pm10": None,
                    "is_live": False,
                    "is_cached": False,
                    "rejection_reason": f"Live telemetry unavailable and no valid cached readings exist ({err_msg}).",
                }

    def clear_cache(self) -> None:
        """Clears memory cache (for testing and resetting)."""
        with self._global_lock:
            self._cached_observations.clear()
            self._preserved_last_valid.clear()


_INGESTION_MANAGER: Optional[ResilientDataIngestionManager] = None


def get_resilient_ingestion_manager() -> ResilientDataIngestionManager:
    """Returns singleton ResilientDataIngestionManager."""
    global _INGESTION_MANAGER
    if _INGESTION_MANAGER is None:
        _INGESTION_MANAGER = ResilientDataIngestionManager()
    return _INGESTION_MANAGER
