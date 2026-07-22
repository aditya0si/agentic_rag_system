"""
Metrics collection for monitoring and observability.

Tracks request counts, latencies, errors, and agent performance.
"""

import time
from collections import defaultdict
from typing import Any
import structlog

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Simple in-memory metrics collector."""
    
    def __init__(self):
        self.counters: dict[str, int] = defaultdict(int)
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.gauges: dict[str, float] = {}
    
    def increment_counter(self, name: str, value: int = 1, **labels) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, **labels)
        self.counters[key] += value
        logger.debug("metric_counter", name=name, value=value, **labels)
    
    def record_histogram(self, name: str, value: float, **labels) -> None:
        """Record a histogram value (for latencies, sizes, etc.)."""
        key = self._make_key(name, **labels)
        self.histograms[key].append(value)
        # Keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        logger.debug("metric_histogram", name=name, value=value, **labels)
    
    def set_gauge(self, name: str, value: float, **labels) -> None:
        """Set a gauge metric (for current state)."""
        key = self._make_key(name, **labels)
        self.gauges[key] = value
        logger.debug("metric_gauge", name=name, value=value, **labels)
    
    def _make_key(self, name: str, **labels) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": dict(self.counters),
            "histograms": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "mean": sum(v) / len(v) if v else 0,
                    "p50": sorted(v)[len(v) // 2] if v else 0,
                    "p95": sorted(v)[int(len(v) * 0.95)] if v else 0,
                    "p99": sorted(v)[int(len(v) * 0.99)] if v else 0,
                }
                for k, v in self.histograms.items()
            },
            "gauges": dict(self.gauges),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.histograms.clear()
        self.gauges.clear()
        logger.info("metrics_reset")


# Global metrics collector
metrics = MetricsCollector()


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str, **labels):
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        metrics.record_histogram(self.metric_name, duration, **self.labels)


def track_agent_execution(agent_name: str):
    """
    Decorator to track agent execution metrics.
    
    Records execution count, duration, and errors.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Increment execution counter
            metrics.increment_counter("agent_executions_total", agent=agent_name)
            
            # Time execution
            with Timer("agent_execution_duration_seconds", agent=agent_name):
                try:
                    result = func(*args, **kwargs)
                    # Increment success counter
                    metrics.increment_counter("agent_success_total", agent=agent_name)
                    return result
                except Exception as e:
                    # Increment error counter
                    metrics.increment_counter(
                        "agent_errors_total",
                        agent=agent_name,
                        error_type=type(e).__name__
                    )
                    raise
        
        return wrapper
    return decorator
