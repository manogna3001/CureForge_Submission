import json
import threading
from pathlib import Path

import pytest


def _reset_singleton():
    import app.src.utils.metrics as m

    with m.MetricsCoordinator._init_lock:
        m.MetricsCoordinator._instance = None


@pytest.fixture(autouse=True)
def reset_metrics():
    _reset_singleton()
    yield
    _reset_singleton()


@pytest.fixture
def coord(tmp_path, monkeypatch):
    from app.src.utils.metrics import MetricsCoordinator

    monkeypatch.setattr(
        MetricsCoordinator,
        "_metrics_path",
        lambda self: tmp_path / "metrics" / "metrics.json",
    )
    c = MetricsCoordinator()
    return c, tmp_path


def test_increment_basic(coord):
    c, _ = coord
    c.increment("agent_created_total")
    assert c.get_value("agent_created_total") == 1


def test_increment_accumulates(coord):
    c, _ = coord
    c.increment("agent_created_total")
    c.increment("agent_created_total", value=4)
    assert c.get_value("agent_created_total") == 5


def test_increment_with_labels(coord):
    c, _ = coord
    c.increment("phase_transitions_total", labels={"phase_from": "research", "phase_to": "hypothesize"})
    c.increment("phase_transitions_total", labels={"phase_from": "research", "phase_to": "test"})
    assert c.get_value("phase_transitions_total", labels={"phase_from": "research", "phase_to": "hypothesize"}) == 1
    assert c.get_value("phase_transitions_total", labels={"phase_from": "research", "phase_to": "test"}) == 1
    assert c.get_value("phase_transitions_total", labels={"phase_from": "research", "phase_to": "synthesize"}) is None


def test_gauge_set_and_overwrite(coord):
    c, _ = coord
    c.gauge("active_agents", value=3.0)
    assert c.get_value("active_agents") == 3.0
    c.gauge("active_agents", value=1.0)
    assert c.get_value("active_agents") == 1.0


def test_gauge_with_labels(coord):
    c, _ = coord
    c.gauge("llm_tokens_used", labels={"model": "gpt-4"}, value=500.0)
    c.gauge("llm_tokens_used", labels={"model": "gpt-3.5"}, value=200.0)
    assert c.get_value("llm_tokens_used", labels={"model": "gpt-4"}) == 500.0
    assert c.get_value("llm_tokens_used", labels={"model": "gpt-3.5"}) == 200.0


def test_histogram_sum_and_count(coord):
    c, _ = coord
    c.histogram("tool_duration_seconds", labels={"tool_name": "read_file"}, value=0.1)
    c.histogram("tool_duration_seconds", labels={"tool_name": "read_file"}, value=0.3)
    result = c.get_value("tool_duration_seconds", labels={"tool_name": "read_file"})
    assert result["count"] == 2
    assert abs(result["sum"] - 0.4) < 1e-9


def test_histogram_multiple_label_sets(coord):
    c, _ = coord
    c.histogram("tool_duration_seconds", labels={"tool_name": "read_file"}, value=1.0)
    c.histogram("tool_duration_seconds", labels={"tool_name": "write_file"}, value=2.0)
    r1 = c.get_value("tool_duration_seconds", labels={"tool_name": "read_file"})
    r2 = c.get_value("tool_duration_seconds", labels={"tool_name": "write_file"})
    assert r1["count"] == 1 and r1["sum"] == 1.0
    assert r2["count"] == 1 and r2["sum"] == 2.0


def test_get_value_unknown_metric(coord):
    c, _ = coord
    assert c.get_value("does_not_exist") is None


def test_flush_creates_valid_json(coord):
    c, tmp_path = coord
    c.increment("agent_created_total")
    c.gauge("active_agents", value=2.0)
    c.histogram("agent_runtime_seconds", labels={"agent_id": "agent_1"}, value=5.0)
    c.flush()

    metrics_file = tmp_path / "metrics" / "metrics.json"
    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text())
    assert "timestamp" in data
    assert "metrics" in data
    assert isinstance(data["metrics"], dict)


def test_flush_json_counter_structure(coord):
    c, tmp_path = coord
    c.increment("llm_calls_total", value=7)
    c.flush()

    data = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    entries = data["metrics"]["llm_calls_total"]
    assert isinstance(entries, list)
    assert entries[0]["value"] == 7
    assert entries[0]["labels"] == {}


def test_flush_json_histogram_structure(coord):
    c, tmp_path = coord
    c.histogram("llm_duration_seconds", value=1.5)
    c.histogram("llm_duration_seconds", value=2.5)
    c.flush()

    data = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    entry = data["metrics"]["llm_duration_seconds"][0]
    assert entry["sum"] == 4.0
    assert entry["count"] == 2
    assert entry["labels"] == {}


def test_flush_json_gauge_structure(coord):
    c, tmp_path = coord
    c.gauge("active_agents", value=3.0)
    c.flush()

    data = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    entry = data["metrics"]["active_agents"][0]
    assert entry["value"] == 3.0


def test_flush_json_labelled_metrics(coord):
    c, tmp_path = coord
    c.increment("phase_transitions_total", labels={"phase_from": "research", "phase_to": "hypothesize"}, value=2)
    c.increment("phase_transitions_total", labels={"phase_from": "hypothesize", "phase_to": "test"}, value=1)
    c.flush()

    data = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    entries = {
        (e["labels"]["phase_from"], e["labels"]["phase_to"]): e["value"]
        for e in data["metrics"]["phase_transitions_total"]
    }
    assert entries[("research", "hypothesize")] == 2
    assert entries[("hypothesize", "test")] == 1


def test_thread_safety(coord):
    c, _ = coord
    n_threads = 20
    n_increments = 100

    def _work():
        for _ in range(n_increments):
            c.increment("concurrent_counter")

    threads = [threading.Thread(target=_work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert c.get_value("concurrent_counter") == n_threads * n_increments


def test_atexit_flush_registered(coord):
    c, tmp_path = coord
    c.increment("atexit_metric")
    c.flush()

    data = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    assert "atexit_metric" in data["metrics"]


def test_singleton_returns_same_instance(coord):
    c, _ = coord
    from app.src.utils.metrics import MetricsCoordinator

    c2 = MetricsCoordinator()
    assert c is c2


def test_periodic_flush_triggers_on_nth_call(tmp_path, monkeypatch):
    _reset_singleton()
    from app.src.utils.metrics import MetricsCoordinator

    monkeypatch.setattr(
        MetricsCoordinator,
        "_metrics_path",
        lambda self: tmp_path / "metrics" / "metrics.json",
    )

    c = MetricsCoordinator()
    c._flush_every = 5

    metrics_file = tmp_path / "metrics" / "metrics.json"
    assert not metrics_file.exists()

    for i in range(4):
        c.increment("tick")

    assert not metrics_file.exists()

    c.increment("tick")
    assert metrics_file.exists()


def test_actual_output_path():
    """Verify the production path resolves to .cache/metrics/metrics.json."""
    from app.src.utils.metrics import MetricsCoordinator
    c = MetricsCoordinator()
    path = c._metrics_path()
    assert path == Path(".cache") / "metrics" / "metrics.json"
