"""
Metrics endpoint — exposes collected metrics for monitoring.
"""

from fastapi import APIRouter
from ..core.metrics import metrics

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Returns all collected metrics in JSON format.
    
    Useful for monitoring dashboards and alerting.
    """
    return metrics.get_metrics()


@router.post("/metrics/reset")
async def reset_metrics():
    """
    Resets all metrics (useful for testing).
    """
    metrics.reset()
    return {"status": "metrics reset"}
