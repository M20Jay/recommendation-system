"""
Week 7 — Recommendation System Preprocessing Pipeline

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026

Description:
    Loads raw MovieLens 100K data from data/raw/
    Cleans and validates all three datasets.
    Saves processed data to data/processed/

Input:
    data/raw/u.data  → ratings
    data/raw/u.item  → movies
    data/raw/u.user  → users

Output:
    data/processed/ratings_clean.csv
    data/processed/movies_clean.csv
    data/processed/users_clean.csv
"""

import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_raw_data():
    """Load raw MovieLens files."""
    logger.info("Loading raw data...")

    ratings = pd.read_csv(RAW_PATH + "u.data", sep="\t",
                          names=["user_id", "item_id",
                                 "rating", "timestamp"])

    movies = pd.read_csv(RAW_PATH + "u.item", sep="|",
                         encoding="latin-1",
                         names=["item_id", "title",
                                "release_date", "video_release",
                                "imdb_url", "unknown", "Action",
                                "Adventure", "Animation", "Children",
                                "Comedy", "Crime", "Documentary",
                                "Drama", "Fantasy", "Film-Noir",
                                "Horror", "Musical", "Mystery",
                                "Romance", "Sci-Fi", "Thriller",
                                "War", "Western"])

    users = pd.read_csv(RAW_PATH + "u.user", sep="|",
                        names=["user_id", "age", "gender",
                               "occupation", "zip_code"])

    logger.info(f"Ratings: {ratings.shape}")
    logger.info(f"Movies: {movies.shape}")
    logger.info(f"Users: {users.shape}")

    return ratings, movies, users


def preprocess_ratings(ratings):
    """Clean and preprocess ratings DataFrame."""
    logger.info("Preprocessing ratings...")

    df = ratings.copy()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Ensure correct types
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['rating'] = df['rating'].astype(int)

    # Validate no missing values
    assert df.isnull().sum().sum() == 0, "Missing values found in ratings"

    logger.info(f"Ratings preprocessed: {df.shape}")
    return df


def preprocess_movies(movies):
    """Clean and preprocess movies DataFrame."""
    logger.info("Preprocessing movies...")

    df = movies.copy()

    # Keep only essential columns
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation',
                  'Children', 'Comedy', 'Crime', 'Documentary',
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']

    essential_cols = ['item_id', 'title', 'release_date'] + genre_cols
    df = df[essential_cols]

    # Convert release_date to datetime
    df['release_date'] = pd.to_datetime(
        df['release_date'], format='%d-%b-%Y', errors='coerce')

    logger.info(f"Movies preprocessed: {df.shape}")
    return df


def preprocess_users(users):
    """Clean and preprocess users DataFrame."""
    logger.info("Preprocessing users...")

    df = users.copy()

    # Encode gender
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})

    # Validate no missing values
    assert df.isnull().sum().sum() == 0, "Missing values found in users"

    logger.info(f"Users preprocessed: {df.shape}")
    return df


def save_processed_data(ratings, movies, users):
    """Save preprocessed data to data/processed/"""
    logger.info("Saving processed data...")

    ratings.to_csv(PROCESSED_PATH + "ratings_clean.csv", index=False)
    movies.to_csv(PROCESSED_PATH + "movies_clean.csv", index=False)
    users.to_csv(PROCESSED_PATH + "users_clean.csv", index=False)

    logger.info("✅ All processed data saved")


if __name__ == "__main__":
    ratings, movies, users = load_raw_data()
    ratings = preprocess_ratings(ratings)
    movies = preprocess_movies(movies)
    users = preprocess_users(users)
    save_processed_data(ratings, movies, users)
    print("✅ Preprocessing complete")