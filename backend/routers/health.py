"""
Health check endpoint — simple liveness probe for deployment monitoring.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Returns a simple status check to verify the API is running."""
    return {"status": "ok"}
