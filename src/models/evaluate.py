"""
Week 7 — Recommendation System
Model Evaluation Script

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026

Description:
    Loads saved production model.
    Evaluates on test set.
    Reports RMSE, MAE and Precision@K.
    
Input:
    models/production_model.pkl
    configs/model.yaml
    data/raw/u1.test

Output:
    Evaluation metrics printed to console
    and logged via logger
"""

import pickle
import yaml
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    """Load config from configs/model.yaml."""
    with open("configs/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(path="models/production_model.pkl"):
    """Load saved production model."""
    logger.info(f"Loading model from {path}...")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded")
    return model


def load_testset(config):
    """Load and prepare test set."""
    raw_path = config['data']['raw_path']
    test_file = config['data']['test_file']

    reader = Reader(rating_scale=(1, 5))

    test_raw = pd.read_csv(
        raw_path + test_file, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"])

    testset = Dataset.load_from_df(
        test_raw[['user_id', 'item_id', 'rating']],
        reader).build_full_trainset().build_testset()

    logger.info(f"Test set loaded: {len(test_raw):,} ratings")
    return testset, test_raw


def evaluate_model(model, testset):
    """Calculate RMSE and MAE on test set."""
    logger.info("Evaluating model...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return predictions, rmse, mae


def precision_at_k(predictions, k=10, threshold=3.5):
    """
    Calculate Precision@K.
    
    For each user — of top K recommendations
    how many are actually relevant (above threshold)?
    
    threshold: minimum rating considered relevant
    """
    # Group predictions by user
    user_predictions = {}
    for pred in predictions:
        if pred.uid not in user_predictions:
            user_predictions[pred.uid] = []
        user_predictions[pred.uid].append(pred)

    precisions = []
    for uid, user_preds in user_predictions.items():
        # Sort by predicted rating — highest first
        user_preds.sort(key=lambda x: x.est, reverse=True)

        # Take top K
        top_k = user_preds[:k]

        # Count relevant in top K
        # Relevant = true rating above threshold
        n_relevant = sum(
            1 for p in top_k if p.r_ui >= threshold)

        precision = n_relevant / k
        precisions.append(precision)

    return round(np.mean(precisions), 4)


if __name__ == "__main__":
    config = load_config()
    model = load_model()
    testset, test_raw = load_testset(config)

    predictions, rmse, mae = evaluate_model(model, testset)

    k = config['api']['top_n']
    threshold = config['api']['min_rating']
    prec_at_k = precision_at_k(predictions, k=k,
                                threshold=threshold)

    print(f"\n{'='*45}")
    print(f"  Model Evaluation Results")
    print(f"{'='*45}")
    print(f"  Model     : Production (Item-CF)")
    print(f"  Test set  : {len(test_raw):,} ratings")
    print(f"{'='*45}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  P@{k}     : {prec_at_k:.4f}")
    print(f"{'='*45}")
    print(f"\n  Interpretation:")
    print(f"  RMSE {rmse:.4f} → predictions off by "
          f"{rmse:.2f} stars on average")
    print(f"  MAE  {mae:.4f} → absolute error "
          f"{mae:.2f} stars on average")
    print(f"  P@{k} {prec_at_k:.4f} → {prec_at_k*100:.1f}% of top "
          f"{k} recommendations are relevant")
    print(f"{'='*45}")