"""Response schemas for Recommendation API."""

from pydantic import BaseModel
from typing import List
from datetime import datetime


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    model: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    timestamp: str