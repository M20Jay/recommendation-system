"""
Week 7 — Recommendation System
Model Training Script

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026

Description:
    Trains SVD, User-CF and Item-CF models on MovieLens 100K.
    All parameters loaded from configs/model.yaml.
    Saves best model as production_model.pkl.

Input:
    configs/model.yaml
    data/raw/u1.base
    data/raw/u1.test

Output:
    models/svd_model.pkl
    models/user_cf_model.pkl
    models/item_cf_model.pkl
    models/production_model.pkl
"""

import pickle
import os
import yaml
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise import accuracy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    """Load parameters from configs/model.yaml."""
    with open("configs/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("✅ Config loaded from configs/model.yaml")
    return config


def load_data(config):
    """Load and prepare train and test sets from config paths."""
    raw_path = config['data']['raw_path']
    train_file = config['data']['train_file']
    test_file = config['data']['test_file']

    logger.info(f"Loading {train_file} and {test_file}...")

    reader = Reader(rating_scale=(1, 5))

    train_raw = pd.read_csv(
        raw_path + train_file, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"])

    test_raw = pd.read_csv(
        raw_path + test_file, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"])

    trainset = Dataset.load_from_df(
        train_raw[['user_id', 'item_id', 'rating']],
        reader).build_full_trainset()

    testset = Dataset.load_from_df(
        test_raw[['user_id', 'item_id', 'rating']],
        reader).build_full_trainset().build_testset()

    logger.info(f"Train: {len(train_raw):,} ratings")
    logger.info(f"Test:  {len(test_raw):,} ratings")

    return trainset, testset


def train_svd(trainset, testset, config):
    """Train and evaluate SVD from config parameters."""
    params = config['models']['svd']
    logger.info(f"Training SVD → n_factors={params['n_factors']} "
                f"n_epochs={params['n_epochs']}")

    model = SVD(
        n_factors=params['n_factors'],
        n_epochs=params['n_epochs'],
        lr_all=params['lr_all'],
        reg_all=params['reg_all'],
        random_state=params['random_state'],
        verbose=False)

    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    logger.info(f"SVD → RMSE: {rmse:.4f} MAE: {mae:.4f}")
    return model, rmse, mae


def train_user_cf(trainset, testset, config):
    """Train and evaluate User-CF from config parameters."""
    params = config['models']['user_cf']
    logger.info(f"Training User-CF → k={params['k_neighbors']} "
                f"similarity={params['similarity']}")

    model = KNNWithMeans(
        k=params['k_neighbors'],
        sim_options={
            'name': params['similarity'],
            'user_based': True
        },
        verbose=False)

    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    logger.info(f"User-CF → RMSE: {rmse:.4f} MAE: {mae:.4f}")
    return model, rmse, mae


def train_item_cf(trainset, testset, config):
    """Train and evaluate Item-CF from config parameters."""
    params = config['models']['item_cf']
    logger.info(f"Training Item-CF → k={params['k_neighbors']} "
                f"similarity={params['similarity']}")

    model = KNNWithMeans(
        k=params['k_neighbors'],
        sim_options={
            'name': params['similarity'],
            'user_based': False
        },
        verbose=False)

    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    logger.info(f"Item-CF → RMSE: {rmse:.4f} MAE: {mae:.4f}")
    return model, rmse, mae


def save_model(model, path):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✅ Saved → {path}")


if __name__ == "__main__":
    config = load_config()

    os.makedirs("models/", exist_ok=True)

    trainset, testset = load_data(config)

    svd,     svd_rmse,  svd_mae  = train_svd(trainset, testset, config)
    user_cf, ucf_rmse,  ucf_mae  = train_user_cf(trainset, testset, config)
    item_cf, icf_rmse,  icf_mae  = train_item_cf(trainset, testset, config)

    save_model(svd,     "models/svd_model.pkl")
    save_model(user_cf, "models/user_cf_model.pkl")
    save_model(item_cf, "models/item_cf_model.pkl")

    results = {
        'SVD':     svd_rmse,
        'User-CF': ucf_rmse,
        'Item-CF': icf_rmse
    }

    best_name = min(results, key=results.get)
    best_models = {
        'SVD':     svd,
        'User-CF': user_cf,
        'Item-CF': item_cf
    }
    save_model(best_models[best_name], "models/production_model.pkl")

    print(f"\n{'='*45}")
    print(f"  Model Training Complete")
    print(f"{'='*45}")
    print(f"  SVD     → RMSE {svd_rmse:.4f}  MAE {svd_mae:.4f}")
    print(f"  User-CF → RMSE {ucf_rmse:.4f}  MAE {ucf_mae:.4f}")
    print(f"  Item-CF → RMSE {icf_rmse:.4f}  MAE {icf_mae:.4f}")
    print(f"{'='*45}")
    print(f"  Best model : {best_name}")
    print(f"  RMSE       : {results[best_name]:.4f}")
    print(f"  Saved      : models/production_model.pkl")
    print(f"{'='*45}")