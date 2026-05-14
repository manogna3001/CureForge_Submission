import atexit
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.src.utils.settings import get_settings


def _labels_key(labels: dict | None) -> str:
    if not labels:
        return ""
    return json.dumps(labels, sort_keys=True)


def _labels_from_key(key: str) -> dict:
    if not key:
        return {}
    return json.loads(key)


class MetricsCoordinator:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._lock = threading.Lock()
                instance._counters: dict[str, dict[str, float]] = {}
                instance._gauges: dict[str, dict[str, float]] = {}
                instance._histograms: dict[str, dict[str, dict]] = {}
                instance._call_count = 0
                instance._flush_every = 100
                atexit.register(instance.flush)
                cls._instance = instance
        return cls._instance

    def _metrics_path(self) -> Path:
        return Path(".cache") / "metrics" / "metrics.json"

    def increment(self, metric_name: str, labels: dict = None, value: float = 1) -> None:
        key = _labels_key(labels)
        with self._lock:
            bucket = self._counters.setdefault(metric_name, {})
            bucket[key] = bucket.get(key, 0) + value
            self._maybe_flush()

    def gauge(self, metric_name: str, labels: dict = None, value: float = 0) -> None:
        key = _labels_key(labels)
        with self._lock:
            self._gauges.setdefault(metric_name, {})[key] = value
            self._maybe_flush()

    def histogram(self, metric_name: str, labels: dict = None, value: float = 0) -> None:
        key = _labels_key(labels)
        with self._lock:
            bucket = self._histograms.setdefault(metric_name, {})
            if key not in bucket:
                bucket[key] = {"sum": 0.0, "count": 0}
            bucket[key]["sum"] += value
            bucket[key]["count"] += 1
            self._maybe_flush()

    def get_value(self, metric_name: str, labels: dict = None):
        key = _labels_key(labels)
        with self._lock:
            if metric_name in self._counters:
                return self._counters[metric_name].get(key)
            if metric_name in self._gauges:
                return self._gauges[metric_name].get(key)
            if metric_name in self._histograms:
                return self._histograms[metric_name].get(key)
        return None

    def _maybe_flush(self) -> None:
        # called while holding self._lock
        self._call_count += 1
        if self._call_count % self._flush_every == 0:
            self._write()

    def _update_system_metrics(self) -> None:
        # called while holding self._lock
        cache_path = Path(get_settings().cache_prefix)
        if cache_path.exists():
            total = sum(
                f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
            )
            self._gauges.setdefault("cache_disk_usage_bytes", {})[""] = float(total)

    def _write(self) -> None:
        # called while holding self._lock
        try:
            self._update_system_metrics()
        except Exception:
            pass

        path = self._metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        metrics_out: dict[str, list] = {}

        for name, samples in self._counters.items():
            metrics_out[name] = [
                {"labels": _labels_from_key(k), "value": v}
                for k, v in samples.items()
            ]

        for name, samples in self._gauges.items():
            metrics_out[name] = [
                {"labels": _labels_from_key(k), "value": v}
                for k, v in samples.items()
            ]

        for name, samples in self._histograms.items():
            metrics_out[name] = [
                {"labels": _labels_from_key(k), "sum": v["sum"], "count": v["count"]}
                for k, v in samples.items()
            ]

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics_out,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def flush(self) -> None:
        try:
            with self._lock:
                self._write()
        except Exception:
            pass


def get_metrics() -> MetricsCoordinator:
    return MetricsCoordinator()
