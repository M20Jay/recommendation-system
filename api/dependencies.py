"""Shared dependencies — model and data loaded once."""

import pickle
import os
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/production_model.pkl"
os.makedirs("models", exist_ok=True)


def train_and_save():
    """Train Item-CF and save if model does not exist."""
    logger.info("Training model for first time...")
    reader = Reader(rating_scale=(1, 5))
    train_raw = pd.read_csv("data/raw/u1.base", sep="\t",
                            names=["user_id", "item_id",
                                   "rating", "timestamp"])
    trainset = Dataset.load_from_df(
        train_raw[['user_id', 'item_id', 'rating']],
        reader).build_full_trainset()

    model = KNNWithMeans(k=40,
                         sim_options={'name': 'cosine',
                                      'user_based': False},
                         verbose=False)
    model.fit(trainset)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("✅ Model trained and saved")
    return model


# Load or train model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded from disk")
else:
    model = train_and_save()

ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies = pd.read_csv("data/processed/movies_clean.csv")