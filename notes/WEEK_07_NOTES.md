# Week 7 — Movie Recommendation System
**Martin James Ng'ang'a · MLOps Engineer · Nairobi, Kenya 🇰🇪**
`github.com/M20Jay` · Week 7 of 15

---

## Overview

Production recommendation engine trained on 100,000 real ratings from 943 users on 1,682 movies. Three collaborative filtering approaches evaluated — Item-Based CF wins with RMSE 0.9540 and Precision@10 of 69.7%. Netflix-standard CineAI Streamlit dashboard with three tabs — recommendations, data insights, model performance.

**Business relevance:** Every organisation with users needs to answer one question: "What should this user engage with next?" Banks → financial products. Telecom → data bundles. UNEP → relevant environmental reports.

---

## Final Results

| Model | RMSE | MAE | Notes |
|-------|------|-----|-------|
| **Item-Based CF** | **0.9540** | **0.7488** | ✅ Best — production model |
| SVD | 0.9561 | 0.7524 | Strong baseline |
| User-Based CF | 0.9703 | 0.7654 | Weakest |

| Metric | Value |
|--------|-------|
| Dataset | MovieLens 100K · 943 users · 1,682 movies |
| Total ratings | 100,000 |
| Matrix sparsity | 93.7% — users rated only 6.3% of movies |
| Precision@10 | 69.7% — 7 in 10 recommendations genuinely relevant |
| Live Dashboard | https://disgrace-system-robust.ngrok-free.dev |
| Live API | http://18.184.3.203:8001/docs |
| Tests | 6/6 passing |

---

## 7-Day Build Plan

| Day | Task | Status |
|-----|------|--------|
| Day 1 | EDA — ratings distribution, sparsity, user behaviour | ✅ |
| Day 2 | User-Based CF + Item-Based CF implementation | ✅ |
| Day 3 | SVD matrix factorization + model evaluation | ✅ |
| Day 4 | FastAPI /recommend endpoint + PostgreSQL storage | ✅ |
| Day 5 | Streamlit CineAI dashboard — 3 tabs | ✅ |
| Day 6 | pytest — 6/6 tests passing | ✅ |
| Day 7 | README + Deploy to AWS EC2 | ✅ |

---

## Project Structure

```
recommendation-system/
├── configs/
│   └── model.yaml              SVD · User-CF · Item-CF hyperparameters
├── data/
│   ├── raw/                    MovieLens 100K — u.data · u.item · u.user
│   └── processed/              Cleaned ratings · movies · users · features
├── models/
│   ├── svd_model.pkl
│   ├── user_cf_model.pkl
│   ├── item_cf_model.pkl
│   └── production_model.pkl    Best model — Item-CF RMSE 0.9540
├── src/
│   ├── data/
│   │   ├── ingestion.py        Data validation and checks
│   │   └── preprocessing.py    Clean and validate raw data
│   ├── features/
│   │   └── feature_engineering.py  User · movie · temporal features
│   └── models/
│       ├── train.py            Train all three models from config
│       └── evaluate.py         RMSE · MAE · Precision@K · Recall@K
├── api/
│   ├── main.py                 FastAPI entry point — loads model at startup
│   ├── dependencies.py         Shared model and data loading
│   ├── schemas/
│   │   ├── request.py          RecommendRequest
│   │   └── response.py         RecommendResponse · HealthResponse
│   └── routes/
│       ├── health.py           GET /health
│       └── recommend.py        POST /recommend
├── tests/
│   └── test_recommend.py       pytest — 6/6 passing
├── streamlit_app.py            CineAI Netflix-standard dashboard
├── Dockerfile
├── docker-compose.yml          PostgreSQL + API + Streamlit
└── requirements.txt
```

---

## Pipeline Architecture

```
MovieLens 100K CSV files
    ↓
src/data/ingestion.py           → validated raw data
    ↓
src/data/preprocessing.py       → user-item matrix (943 × 1,682)
    ↓
src/features/feature_engineering.py  → similarity matrices
    ↓
src/models/train.py             → SVD · User-CF · Item-CF models saved
    ↓
src/models/evaluate.py          → RMSE · MAE · Precision@10 · Recall@10
    ↓
Item-CF selected as production model (RMSE 0.9540)
    ↓
api/main.py                     → loads production_model.pkl at startup
    ↓
POST /recommend                 → top-N recommendations for user
    ↓
PostgreSQL                      → ratings + predictions stored
    ↓
Streamlit CineAI dashboard      → recommendations · insights · performance
    ↓
Docker + AWS EC2                → live 24/7 via ngrok HTTPS tunnel
```

---

## Key Concepts

### Three Collaborative Filtering Approaches

```python
# 1. USER-BASED CF — find similar users
# "Users who rated the same movies as you also liked X"
# Steps:
#   - Build user-item matrix
#   - Calculate cosine similarity between users
#   - Find top-K most similar users
#   - Recommend movies they liked that you haven't seen

# 2. ITEM-BASED CF — find similar items (WINNER)
# "Because you liked Toy Story, you might like Aladdin"
# Steps:
#   - Build item-item similarity matrix
#   - For movies you've rated highly, find similar movies
#   - Recommend most similar unseen movies

# 3. SVD — matrix factorization
# Decompose 943×1682 sparse matrix into latent factors
# Each user and movie represented as vector of hidden features
# Prediction = dot product of user vector and movie vector
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
```

### Why Item-CF Beat SVD and User-CF

```
Item-CF wins because item relationships are MORE STABLE than user preferences.

User preferences drift over time:
  A user who loved action films in 2010 may prefer documentaries in 2024
  User similarity calculated on historical data becomes stale

Item relationships are stable:
  Toy Story will always attract the same audience as Aladdin
  Not because someone labelled them both animated — because the SAME PEOPLE watched both
  This pattern is durable across time

SVD advantages:
  Handles sparsity well — 93.7% of ratings missing
  Captures latent factors (genre preferences without explicit labels)
  Disadvantage: computationally expensive, less interpretable

With only 100,000 ratings across 943×1,682 matrix:
  93.7% sparsity means most user-movie pairs have no data
  Item-CF needs fewer data points to find stable similarities
  Hence Item-CF outperforms on this dataset
```

### The Sparsity Problem

```
User-item matrix: 943 users × 1,682 movies = 1,586,526 possible ratings
Actual ratings: 100,000
Sparsity: (1 - 100,000/1,586,526) = 93.7%

This means:
  Every user has rated only ~106 of 1,682 movies (6.3%)
  Most user-movie pairs have no data
  The model must predict ratings for the 93.7% it has never seen

Why this is hard:
  User A and User B may have only 3 movies in common
  Similarity score based on 3 points — unreliable
  Cold start problem: new user has 0 ratings → no similarity → no recommendations

How Item-CF handles sparsity better:
  Items have more ratings than users on average
  Popular movies rated by hundreds of users → stable similarity
  Less affected by individual sparsity
```

### Precision@K vs RMSE — Two Different Measurements

```
RMSE (Root Mean Square Error):
  Measures: how close predicted ratings are to actual ratings
  Scale: same as original rating scale (1-5)
  RMSE 0.9540 means: predictions are off by ~0.95 stars on average
  Good for: measuring prediction accuracy

Precision@K:
  Measures: of the top K recommendations shown, how many are relevant?
  Relevant = user would rate 4 or above
  Precision@10 = 69.7% means: 7 out of 10 recommendations are genuinely good

Why both matter:
  Low RMSE but low Precision@10 → accurate predictions but wrong recommendations
  High Precision@10 but high RMSE → good recommendations despite imperfect ratings

For a recommendation system, Precision@10 is the business metric.
RMSE is the technical metric used during model selection.
```

### FastAPI Recommendation Endpoint

```python
from fastapi import APIRouter, HTTPException
from api.schemas.request import RecommendRequest
from api.schemas.response import RecommendResponse
from api.dependencies import model, ratings_df, movies_df

router = APIRouter()

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    try:
        user_id = request.user_id
        n = request.n_recommendations

        # Get movies user has already rated
        rated = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()

        # Get all unseen movies
        all_movies = movies_df['movie_id'].tolist()
        unseen = [m for m in all_movies if m not in rated]

        # Predict ratings for unseen movies
        predictions = [(m, model.predict(user_id, m).est) for m in unseen]
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        top_n = predictions[:n]
        recommendations = [
            {"movie_id": m, "title": movies_df[movies_df['movie_id']==m]['title'].values[0],
             "predicted_rating": round(r, 2)}
            for m, r in top_n
        ]
        return RecommendResponse(user_id=user_id, recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## CLI Reference

### Training Pipeline

```bash
# Run from project root
python -m src.data.ingestion
python -m src.data.preprocessing
python -m src.models.train
python -m src.models.evaluate

# Check saved models
ls -lh models/
# svd_model.pkl
# user_cf_model.pkl
# item_cf_model.pkl
# production_model.pkl
```

### API Testing

```bash
# Health check
curl -s http://localhost:8001/health | python3 -m json.tool

# Get recommendations for user 196
curl -s -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 196, "n_recommendations": 5}' \
  | python3 -m json.tool
```

### Streamlit Dashboard

```bash
# Run locally
streamlit run streamlit_app.py --server.port 8502

# Dashboard tabs:
# Tab 1 — Recommendations: enter user ID → get top-N recommendations
# Tab 2 — Data Insights: rating distributions, top movies, user activity
# Tab 3 — Model Performance: RMSE, MAE, Precision@K comparison
```

### Docker Commands

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View API logs
docker compose logs api --tail=20

# Restart API
docker compose restart api

# Stop everything
docker compose down

# Check port on server
sudo ss -tlnp | grep 8001
```

### Run Tests

```bash
pytest tests/ -v
# 6/6 tests should pass
# test_health, test_recommend, test_invalid_user
# test_recommend_count, test_ratings_stored, test_response_schema
```

### Database Inspection

```bash
docker compose exec postgres psql -U recommendation -d recommendation

# Inside psql:
\dt
SELECT COUNT(*) FROM ratings;
SELECT COUNT(*) FROM predictions;
SELECT user_id, COUNT(*) as num_recommendations
  FROM predictions GROUP BY user_id ORDER BY num_recommendations DESC LIMIT 10;
\q
```

---

## Debugging Reference

### Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `User not found` | User ID must be between 1-943 for MovieLens 100K |
| `logger not defined` | Replace `logger.warning()` with `print()` in recommend.py |
| `PostgreSQL connection refused` | Check `DB_HOST=postgres` not `localhost` inside Docker |
| `Model file not found` | Run `python -m src.models.train` first to generate pkl files |
| `surprise ImportError` | Install: `pip install scikit-surprise` not `surprise` |
| `Port 8001 already in use` | `lsof -i :8001` then `kill -9 <PID>` |

### The Logger Bug Fix

```bash
# This bug caused 500 errors on every recommendation request
# api/routes/recommend.py used logger.warning() but logger was never imported

# Fix:
sed -i 's/logger.warning(f"DB save skipped: {e}")/print(f"DB save skipped: {e}")/' \
  api/routes/recommend.py
```

---

## AWS EC2 Deployment

```bash
# SSH to server
ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@18.184.3.203

# Start recommendation system
cd ~/recommendation-system
docker compose up -d

# Verify API running
docker ps | grep recommendation
curl -s http://localhost:8001/health

# Check ngrok tunnel for dashboard
sudo systemctl status ngrok
curl -s localhost:4040/api/tunnels | grep public_url

# Check logs
docker compose logs api --tail=20
```

---

## Deep Dives — Critical Concepts

### Collaborative Filtering vs Content-Based Filtering

Collaborative filtering (what this pipeline uses):
- Uses behaviour — ratings — not item attributes
- "Users who liked what you liked also liked X"
- Advantage: discovers unexpected connections — action fan might love a documentary others like
- Disadvantage: cold start — new users and new items have no ratings history

Content-based filtering:
- Uses item attributes — genre, director, cast
- "You liked action movies → recommend more action movies"
- Advantage: no cold start — works immediately for new items
- Disadvantage: filter bubble — only recommends more of what you already like

Hybrid systems (Netflix, Spotify, YouTube):
- Combine both approaches
- Collaborative for established users
- Content-based for new users and new items

### The Cold Start Problem

```
New user cold start:
  User has 0 ratings → no similarity to anyone → no recommendations
  Solutions:
    Ask for explicit preferences during onboarding
    Recommend globally popular items
    Use demographic information (age, location)

New item cold start:
  New movie has 0 ratings → never recommended by collaborative filtering
  Solutions:
    Content-based filtering for new items
    Promote new items explicitly
    Use item metadata (genre, director) until ratings accumulate

This pipeline:
  Minimum 20 ratings per user (MovieLens requirement)
  No cold start problem for this dataset
  Production system would need cold start strategy
```

### Matrix Factorization — Why SVD Works

```
User-item matrix: 943 rows × 1,682 columns = huge sparse matrix

SVD decomposes this into:
  User matrix:  943 × k   (k latent factors per user)
  Item matrix:  k × 1,682 (k latent factors per item)
  k = 100 in this pipeline

What are latent factors?
  Not explicitly defined — discovered by the algorithm
  Factor 1 might capture "likes action films"
  Factor 2 might capture "prefers 1990s films"
  Factor 3 might capture "rates generously"

Prediction:
  User vector × Movie vector = predicted rating
  If both user and movie score high on "action films" factor → high predicted rating

Why SVD handles sparsity:
  Missing ratings are not zeros — they are unknown
  SVD fills in the gaps by learning from available ratings
  Generalises patterns across the entire matrix
```

### Precision@K — The Real Business Metric

```
RMSE tells you: how accurate are your predicted ratings?
Precision@K tells you: are your recommendations actually good?

These can disagree:
  A model predicting exactly 3.5 for every movie:
    RMSE might be low (average rating is ~3.5)
    Precision@10 = 0% (never recommends anything the user loves)

Precision@10 = 0.697 means:
  Show user 10 recommendations
  7 of those 10 are movies the user would rate 4 or above
  3 are misses

What counts as "relevant":
  Threshold set at 4.0 (out of 5)
  Movies rated 1-3 → not relevant
  Movies rated 4-5 → relevant

Why 10:
  Netflix shows ~10 items per row
  Spotify shows ~10 items per playlist recommendation
  Industry standard for recommendation evaluation
```

---

*Week 7 of 15 · Movie Recommendation System · Built in Nairobi, Kenya 🇰🇪*
*Live Dashboard: https://disgrace-system-robust.ngrok-free.dev · Live API: http://18.184.3.203:8001/docs*
*Repository: https://github.com/M20Jay/recommendation-system*
