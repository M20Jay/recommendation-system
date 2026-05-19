# Recommendation System Pipeline 🎬

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-storage-blue)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)
![License](https://img.shields.io/badge/License-Research-yellow)

**Production recommendation engine — collaborative filtering, matrix factorization, and item-based recommendations trained on MovieLens 100K.**

Built by [Martin James Ng'ang'a](https://github.com/M20Jay) — MLOps Engineer | Nairobi, Kenya 🇰🇪

> 🌐 **Live Dashboard** → https://recommendation-system-dashboard.onrender.com
> 
> ⚡ **Live API** → https://recommendation-system-2gt5.onrender.com/docs
> 
> 📁 **GitHub** → https://github.com/M20Jay/recommendation-system

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
|-------|------|-----|-------|
| SVD | 0.9561 | 0.7524 | Strong baseline |
| Item-Based CF | 0.9540 | 0.7488 | ✅ Best performer — production model |
| User-Based CF | 0.9703 | 0.7654 | Baseline |

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
├── configs/
│   └── model.yaml                Model parameters — SVD · User-CF · Item-CF
├── data/
│   ├── raw/                      MovieLens 100K files — u.data · u.item · u.user
│   └── processed/                Cleaned data — ratings · movies · users · features
├── models/
│   ├── svd_model.pkl             Trained SVD model
│   ├── user_cf_model.pkl         Trained User-CF model
│   ├── item_cf_model.pkl         Trained Item-CF model
│   └── production_model.pkl      Best model — Item-CF RMSE 0.9540
├── notebooks/
│   └── 01_EDA.ipynb              Exploratory data analysis — 10 sections
├── reports/
│   └── figures/                  EDA visualisations
├── src/
│   ├── data/
│   │   ├── ingestion.py          Data validation and checks
│   │   └── preprocessing.py      Clean and validate raw data
│   ├── features/
│   │   └── feature_engineering.py  User · movie · temporal features
│   ├── models/
│   │   ├── train.py              Train all three models from config
│   │   └── evaluate.py           RMSE · MAE · Precision@K
│   └── utils/
│       ├── logger.py             Logging utility
│       └── database.py           PostgreSQL connection and storage
├── api/
│   ├── main.py                   FastAPI application entry point
│   ├── dependencies.py           Shared model and data loading
│   ├── schemas/
│   │   ├── request.py            RecommendRequest schema
│   │   └── response.py           RecommendResponse · HealthResponse
│   └── routes/
│       ├── health.py             GET /health endpoint
│       └── recommend.py          POST /recommend endpoint
├── tests/
│   └── test_recommend.py         pytest — 6/6 passing
├── streamlit_app.py              CineAI Netflix-standard dashboard
├── Dockerfile                    Production container
├── docker-compose.yml            PostgreSQL + API + Streamlit
├── requirements.txt              All dependencies
└── .env                          Environment variables (not committed)
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