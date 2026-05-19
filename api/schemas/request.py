"""Request schemas for Recommendation API."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for")
    n: int = Field(default=10, description="Number of recommendations to return")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 196,
                "n": 10
            }
        }