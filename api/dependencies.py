"""Shared dependencies — model and data loaded once."""

import pickle
import pandas as pd

with open("models/production_model.pkl", "rb") as f:
    model = pickle.load(f)

ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies = pd.read_csv("data/processed/movies_clean.csv")