"""
Week 7 — Recommendation System
Feature Engineering Pipeline

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026

Description:
    Creates new features from cleaned MovieLens data.
    Features improve model performance and interpretability.

Input:
    data/processed/ratings_clean.csv
    data/processed/movies_clean.csv
    data/processed/users_clean.csv

Output:
    data/processed/features.csv
    
Features Created:
    release_year        → extracted from movie title e.g. (1995)
    movie_age           → 2026 - release_year
    user_mean_rating    → average rating given by each user
    movie_mean_rating   → average rating received by each movie
    user_rating_count   → how many movies each user has rated
    movie_rating_count  → how many users have rated each movie
"""
import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = "data/processed/"


def load_processed_data():
    """Load cleaned data from data/processed/"""
    logger.info("Loading processed data...")

    ratings = pd.read_csv(PROCESSED_PATH + "ratings_clean.csv")
    movies = pd.read_csv(PROCESSED_PATH + "movies_clean.csv")
    users = pd.read_csv(PROCESSED_PATH + "users_clean.csv")

    logger.info(f"Ratings: {ratings.shape}")
    logger.info(f"Movies:  {movies.shape}")
    logger.info(f"Users:   {users.shape}")

    return ratings, movies, users


def extract_release_year(movies):
    """Extract release year from movie title e.g. Toy Story (1995)"""
    logger.info("Extracting release year from titles...")

    df = movies.copy()

    df['release_year'] = df['title'].str.extract(r'\((\d{4})\)')
    df['release_year'] = pd.to_numeric(
        df['release_year'], errors='coerce')
    df['movie_age'] = 2026 - df['release_year']

    missing = df['release_year'].isnull().sum()
    logger.info(f"Release year extracted. Missing: {missing}")

    return df


def create_rating_features(ratings):
    """Create user and movie level rating statistics."""
    logger.info("Creating rating features...")

    df = ratings.copy()

    # User level features
    user_stats = df.groupby('user_id')['rating'].agg(
        user_mean_rating='mean',
        user_rating_count='count'
    ).reset_index()

    # Movie level features
    movie_stats = df.groupby('item_id')['rating'].agg(
        movie_mean_rating='mean',
        movie_rating_count='count'
    ).reset_index()

    # Merge features onto ratings
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(movie_stats, on='item_id', how='left')

    logger.info(f"Rating features created: {df.shape}")
    return df


def build_features(ratings, movies):
    """Combine all features into one DataFrame."""
    logger.info("Building final features DataFrame...")

    # Create rating features
    df = create_rating_features(ratings)

    # Extract release year from movies
    movies_enriched = extract_release_year(movies)

    # Merge movie features onto ratings
    df = df.merge(
        movies_enriched[['item_id', 'release_year', 'movie_age']],
        on='item_id',
        how='left'
    )

    logger.info(f"Final features shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


def save_features(df):
    """Save features to data/processed/"""
    path = PROCESSED_PATH + "features.csv"
    df.to_csv(path, index=False)
    logger.info(f"✅ Features saved to {path}")


if __name__ == "__main__":
    ratings, movies, users = load_processed_data()
    features = build_features(ratings, movies)
    save_features(features)
    print(f"✅ Feature engineering complete")
    print(f"   Shape: {features.shape}")
    print(f"   Columns: {features.columns.tolist()}")