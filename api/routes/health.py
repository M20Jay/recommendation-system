"""Health check endpoint."""

from fastapi import APIRouter
from datetime import datetime
from api.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model="Item-Based Collaborative Filtering",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )