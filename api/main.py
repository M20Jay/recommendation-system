"""
Week 7 — Recommendation System
FastAPI Application

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026
"""

from fastapi import FastAPI
from api.routes import health, recommend

app = FastAPI(
    title="Movie Recommendation API",
    description="Collaborative filtering recommendations — MovieLens 100K. | MLOps Engineer | github.com/M20Jay",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(recommend.router)