"""
Regression tests for the metrics collection module.

Validates counters, histograms, gauges, and the Timer context manager.
"""

import time

import pytest

from backend.core.metrics import MetricsCollector, Timer, metrics, track_agent_execution

pytestmark = pytest.mark.unit


class TestMetricsCollector:
    def test_counter_increment(self):
        collector = MetricsCollector()
        collector.increment_counter("requests", path="/ask")
        collector.increment_counter("requests", path="/ask")
        assert collector.counters["requests{path=/ask}"] == 2

    def test_counter_with_value(self):
        collector = MetricsCollector()
        collector.increment_counter("bytes_sent", value=1024, endpoint="/upload")
        assert collector.counters["bytes_sent{endpoint=/upload}"] == 1024

    def test_histogram_records_values(self):
        collector = MetricsCollector()
        collector.record_histogram("latency", 0.1, endpoint="/ask")
        collector.record_histogram("latency", 0.2, endpoint="/ask")
        collector.record_histogram("latency", 0.3, endpoint="/ask")
        stats = collector.get_metrics()["histograms"]["latency{endpoint=/ask}"]
        assert stats["count"] == 3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3
        assert 0.19 < stats["mean"] < 0.21

    def test_histogram_percentiles(self):
        collector = MetricsCollector()
        for i in range(100):
            collector.record_histogram("lat", float(i))
        stats = collector.get_metrics()["histograms"]["lat"]
        assert stats["p50"] == 50
        assert stats["p95"] == 95

    def test_gauge_set(self):
        collector = MetricsCollector()
        collector.set_gauge("active_sessions", 42)
        assert collector.gauges["active_sessions"] == 42

    def test_reset(self):
        collector = MetricsCollector()
        collector.increment_counter("test")
        collector.record_histogram("test_hist", 1.0)
        collector.set_gauge("test_gauge", 1.0)
        collector.reset()
        assert len(collector.counters) == 0
        assert len(collector.histograms) == 0
        assert len(collector.gauges) == 0

    def test_get_metrics_structure(self):
        collector = MetricsCollector()
        collector.increment_counter("c")
        collector.record_histogram("h", 1.0)
        collector.set_gauge("g", 1.0)
        result = collector.get_metrics()
        assert "counters" in result
        assert "histograms" in result
        assert "gauges" in result


class TestTimer:
    def test_timer_records_duration(self):
        # Timer records into the module-level collector singleton, not into a
        # locally constructed MetricsCollector.
        before = metrics.get_metrics()["histograms"].get("test_op{label=x}", {}).get("count", 0)

        with Timer("test_op", label="x"):
            time.sleep(0.01)

        stats = metrics.get_metrics()["histograms"]["test_op{label=x}"]
        assert stats["count"] == before + 1
        assert stats["min"] > 0


class TestTrackAgentExecution:
    def test_decorator_tracks_success(self):
        @track_agent_execution("test_agent")
        def my_agent():
            return "result"

        result = my_agent()
        assert result == "result"
        # Check metrics were recorded
        all_metrics = metrics.get_metrics()
        assert any("test_agent" in k for k in all_metrics["counters"])

    def test_decorator_tracks_errors(self):
        import pytest

        @track_agent_execution("failing_agent")
        def failing_agent():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            failing_agent()

        all_metrics = metrics.get_metrics()
        error_counters = {k: v for k, v in all_metrics["counters"].items() if "errors" in k}
        assert any("failing_agent" in k for k in error_counters)
