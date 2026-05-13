# Recommendation System Pipeline 🎬

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-storage-blue)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)
![License](https://img.shields.io/badge/License-Research-yellow)

**Production recommendation engine — collaborative filtering, matrix factorization, and item-based recommendations trained on MovieLens 100K.**

Built by [Martin James Ng'ang'a](https://github.com/M20Jay) — MLOps Engineer | Nairobi, Kenya 🇰🇪

> 🔗 **Live API** — added on deployment

---

## Business Problem

Every organisation with users needs to answer one question:

**"What should this user engage with next?"**

- Banks → recommend financial products
- Telecom → recommend data bundles
- Healthcare → recommend relevant resources
- Environmental organisations → recommend relevant documents and reports

This pipeline answers that question automatically using three complementary approaches — collaborative filtering, item-based filtering, and matrix factorization (SVD).

---

## Dataset

**MovieLens 100K** — GroupLens Research Project, University of Minnesota

100,000 ratings (1-5) from 943 users on 1,682 movies
Each user has rated at least 20 movies
Collected: September 1997 — April 1998

**Citation:**
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015). DOI: http://dx.doi.org/10.1145/2827872

**Data files:**

u.data       → 100,000 ratings: user_id | item_id | rating | timestamp
u.item       → 1,682 movies: title | release_date | genres (19 columns)
u.user       → 943 users: age | gender | occupation | zip_code
u1.base      → 80% training split (pre-made)
u1.test      → 20% test split (pre-made)

---

## Models

| Model | Approach | Use Case |
|---|---|---|
| **User-Based CF** | Find similar users → recommend what they liked | Cold-start for new items |
| **Item-Based CF** | Find similar items → recommend related content | Because you liked X |
| **SVD (Matrix Factorization)** | Decompose user-item matrix into latent factors | Best accuracy |

---

## Live Results

| Model | RMSE | MAE | Notes |
|---|---|---|---|
| SVD | TBD | TBD | Best performer expected |
| Item-Based CF | TBD | TBD | Most interpretable |
| User-Based CF | TBD | TBD | Baseline |

*Results updated after training*

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Source | MovieLens 100K — GroupLens Research |
| Models | SVD · User-Based CF · Item-Based CF |
| API | FastAPI · Uvicorn · Pydantic |
| Storage | PostgreSQL — ratings + predictions |
| Dashboard | Streamlit · Plotly |
| Containerisation | Docker · docker-compose |
| Deployment | Render |
| Logging | Python logging · Rotating file handler |
| Testing | pytest — written for every file |

---

## Project Structure

```text
recommendation-system/
├── configs/                  Model parameters (YAML)
├── data/
│   ├── raw/                  MovieLens 100K files
│   └── processed/            Clean matrices and features
├── models/                   Trained model files
├── notebooks/                EDA — ratings analysis
├── screenshots/              Dashboard screenshots
├── src/
│   ├── data/                 Ingestion and preprocessing
│   ├── features/             Feature engineering
│   ├── models/               Train and evaluate
│   └── utils/                Logger and database helpers
├── api/
│   ├── main.py               FastAPI application
│   └── routes/               Recommend and health endpoints
├── tests/                    pytest — written for every file
├── streamlit_app.py          Interactive dashboard
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/recommend` | POST | Get top-N recommendations for a user |

---

## Pipeline Architecture

```text
MovieLens 100K CSV files
         ↓
src/data/ingestion.py
         ↓
src/data/preprocessing.py      → user-item matrix
         ↓
src/features/feature_engineering.py → similarity matrices
         ↓
src/models/train.py            → models/ SVD · User-CF · Item-CF
         ↓
src/models/evaluate.py         → RMSE · MAE · Precision@K · Recall@K
         ↓
FastAPI /recommend endpoint    → top-N recommendations
         ↓
PostgreSQL                     → ratings + predictions storage
         ↓
Streamlit dashboard            → interactive recommendations
         ↓
Docker + Render                → production deployment
```
---

## Running Locally

git clone https://github.com/M20Jay/recommendation-system.git
cd recommendation-system
pip install -r requirements.txt
python -m src.data.ingestion
python -m src.data.preprocessing
python -m src.models.train
python -m src.models.evaluate
uvicorn api.main:app --reload

## Running with Docker

docker-compose up --build

## Running Tests

pytest tests/ -v

---

## Data License

MovieLens data is used for research purposes under the GroupLens Research Project usage terms. This project does not redistribute the data commercially. Full license: https://grouplens.org/datasets/movielens/100k/

---

*Building from Nairobi. For the world. 🇰🇪*
*Week 7 of 15-Week MLOps Roadmap — github.com/M20Jay*