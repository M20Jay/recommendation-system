# Week 7 — Recommendation System


---

## 🔗 Links

| | URL |
|---|---|
| 🌐 Dashboard | https://recommendation-system-dashboard.onrender.com |
| ⚡ API | https://recommendation-system-2gt5.onrender.com/docs |
| 📁 GitHub | https://github.com/M20Jay/recommendation-system |

> ⚠️ APIs migrating from Render to AWS EC2. Live again by 31 May 2026.

---

## 1. What I Built

CineAI — a production movie recommendation engine trained on MovieLens 100K.

**Stack:** scikit-surprise · FastAPI · Streamlit · PostgreSQL · Docker · Render · pytest

**Results:**

| Model | RMSE | MAE | P@10 |
|---|---|---|---|
| **Item-CF ✅ Production** | **0.9540** | **0.7488** | **69.7%** |
| SVD | 0.9561 | 0.7524 | — |
| User-CF | 0.9703 | 0.7654 | — |

---

## 2. The Dataset — MovieLens 100K

| Stat | Value |
|---|---|
| Total Ratings | 100,000 |
| Users | 943 |
| Movies | 1,682 |
| Sparsity | 93.7% empty |
| Mean Rating | 3.53 |
| Train Split | u1.base (80%) |
| Test Split | u1.test (20%) |

**Key EDA Findings:**
- 61.4% of users have fewer than 50 ratings → cold start problem
- 31.5% of movies have fewer than 10 ratings → long tail problem
- Star Wars most rated with 583 ratings
- 71% male users, median age 31, students dominate
- Positive skew (mean 3.53) because of selection bias — users rate movies they expected to like

---

## 3. scikit-surprise Library

scikit-surprise = **Simple Python RecommendatIon System Engine**

Why not regular scikit-learn? scikit-learn has no built-in collaborative filtering. surprise provides SVD, KNNWithMeans, cross-validation, and accuracy metrics out of the box.

**Key classes:**

| Class | Purpose |
|---|---|
| `Reader(rating_scale=(1,5))` | Tells surprise the rating scale |
| `Dataset.load_from_df()` | Loads pandas DataFrame into surprise format |
| `build_full_trainset()` | Uses ALL data for training the final model |
| `build_testset()` | Converts data into test format for evaluation |
| `KNNWithMeans` | Item-CF and User-CF with mean normalisation |
| `SVD` | Matrix factorisation algorithm |
| `accuracy.rmse()` | Calculates RMSE from predictions |
| `accuracy.mae()` | Calculates MAE from predictions |

**build_full_trainset() vs build_testset():**

When training the production model we use `build_full_trainset()` — all 100,000 ratings. When evaluating during development we split into u1.base (train) and u1.test (test) and use `build_testset()` for the test portion.

---

## 4. The Three Models

### User-Based CF
Find users similar to you. Recommend what they liked but you have not seen.

Uses cosine similarity between user rating vectors. k=40 nearest neighbours.

**Weakness:** User preferences drift over time. Does not scale with many users.

### SVD — Singular Value Decomposition
Decompose the user-item matrix M into latent factors: **M = U × S × V^T**

- U = user factors
- V = item factors
- S = importance of each factor
- n_factors=100 → 100 latent dimensions, each representing a hidden concept

**Config:** n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02

### Item-Based CF — Production Model ✅
Find movies similar to what you already rated highly. Recommend similar movies.

**Why it won:**
Movie similarity patterns are stable. Star Wars will always attract the same audience as Return of the Jedi — not because both are Sci-Fi, but because the same people watched and rated both. Item relationships do not drift. User preferences do.

---

## 5. Metrics

### RMSE — Root Mean Squared Error
Measures how far predicted ratings are from actual ratings.

**Formula:** `√(average of all squared errors)`

**Why squared:** penalises large errors more than small ones.

**Interpretation:** RMSE 0.9540 means on average our predictions are 0.95 stars from the actual rating on a 1-5 scale. Lower is better.

### MAE — Mean Absolute Error
Average absolute difference between predicted and actual ratings. No squaring.

**Interpretation:** MAE 0.7488 means we are 0.75 stars away on average. More forgiving of outliers than RMSE.

### Precision@K — Full Explanation

**Question it answers:** Of the top K recommendations, what percentage does the user actually like?

K=10 in our system. Threshold=3.5 from configs/model.yaml.

**Step by step:**

```python
# Step 1: Group predictions by user
# After testing we have thousands of (predicted_rating, actual_rating) tuples per user

# Step 2: Sort by predicted rating — highest first
user_ratings.sort(key=lambda x: x[0], reverse=True)

# Step 3: Take top K
top_k = user_ratings[:10]

# Step 4: Count relevant items (actual rating >= threshold)
relevant = sum(1 for est, true_r in top_k if true_r >= 3.5)

# Step 5: Precision for this user
precision = relevant / k  # e.g. 7/10 = 0.70

# Step 6: Average across all 943 users
return sum(precisions) / len(precisions)  # = 0.697 = 69.7%
```

**Result:** 69.7% — 7 out of 10 recommendations are movies the user genuinely enjoys. Netflix benchmark: 70-80%.

---

## 6. Python Concepts

### lambda
A small anonymous function written in one line.

```python
# With lambda
user_ratings.sort(key=lambda x: x[0], reverse=True)

# Equivalent without lambda
def get_predicted_rating(item):
    return item[0]
user_ratings.sort(key=get_predicted_rating, reverse=True)
```

`x` is each element. `x[0]` means sort by the first item in the tuple (predicted_rating).

### enumerate()
Loop through a list and get both index and value.

```python
for i, rec in enumerate(recs, 1):
    # i = 1, 2, 3... (starts from 1 not 0)
    # rec = each movie recommendation
```

`enumerate(recs, 1)` starts counting from 1 — that is why our rankings show 01, 02, 03.

### .items() — dictionary method
Returns key-value pairs from a dictionary.

```python
for uid, user_ratings in user_recommendations.items():
    # uid = user ID (key)
    # user_ratings = list of (predicted, actual) tuples (value)
```

### List slicing [:k]
Takes first k items from a list.

```python
top_k = user_ratings[:10]  # first 10 items
```

Because we sorted highest first these are the top 10 recommendations.

### sum() with generator expression
Count items that meet a condition in one line.

```python
relevant = sum(1 for est, true_r in top_k if true_r >= 3.5)

# Equivalent to:
relevant = 0
for est, true_r in top_k:
    if true_r >= 3.5:
        relevant += 1
```

### Tuple unpacking
Unpack multiple values from a tuple in one line.

```python
for uid, iid, true_r, est, _ in predictions:
    # uid    = user ID
    # iid    = item ID
    # true_r = actual rating
    # est    = predicted rating
    # _      = unused value (convention)
```

### Dictionary default pattern
Build a dictionary where each key maps to a list.

```python
if uid not in user_recommendations:
    user_recommendations[uid] = []
user_recommendations[uid].append((est, true_r))
```

---

## 7. FastAPI Concepts

### What is FastAPI?
Modern Python framework for building REST APIs. Auto-generates /docs with Swagger UI. Based on Starlette and Pydantic.

### Pydantic Schemas
Validate incoming and outgoing data automatically.

```python
# Request — what the API accepts
class RecommendRequest(BaseModel):
    user_id: int = Field(...)       # required integer
    n: int = Field(default=10)      # optional, defaults to 10

# Response — what the API returns
class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float
```

If user sends user_id as a string FastAPI rejects it automatically with 422 error.

### Dependencies pattern
Load model ONCE on startup. Share across all routes. Without this pattern the model would reload on every request.

```python
# api/dependencies.py — runs ONCE when API starts
with open("models/production_model.pkl", "rb") as f:
    model = pickle.load(f)

ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies  = pd.read_csv("data/processed/movies_clean.csv")
```

### HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 404 | Not Found — user_id does not exist |
| 422 | Validation failed — invalid request data |
| 500 | Server error |

---

## 8. PostgreSQL Integration

### psycopg2
The most popular PostgreSQL adapter for Python. Allows Python to connect to PostgreSQL, run queries, and retrieve results.

### Tables created

| Table | Purpose |
|---|---|
| `recommendation_requests` | Logs every API call — user_id, model_used, timestamp |
| `recommendations` | Logs every recommended movie — title, predicted_rating, rank |

### .env and python-dotenv

```bash
# .env file (never committed to GitHub)
DATABASE_URL=postgresql://martin:martin123@localhost:5432/recommendations
```

```python
# database.py
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
```

`load_dotenv()` reads the .env file. `os.getenv()` retrieves the value. Passwords never appear in code.

---

## 9. Docker

### Why Docker?
The problem: code that works on your Mac may not work on a Linux server. Different Python versions, different library versions, different OS. Docker packages everything into a container that runs identically everywhere.

### Dockerfile explained

```dockerfile
FROM python:3.12-slim
# Start from minimal Python 3.12 Linux image

RUN apt-get update && apt-get install -y gcc g++
# Install C compilers — scikit-surprise has C extensions
# Without gcc the pip install of scikit-surprise FAILS

RUN pip install --upgrade pip
# Upgrade pip to avoid old setuptools issues

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# --host 0.0.0.0 makes it accessible outside the container
```

### Key commands

| Command | What it does |
|---|---|
| `docker build -t recommendation-api .` | Build image from Dockerfile |
| `docker compose up -d postgres` | Start only postgres in background |
| `docker compose up --build` | Build and start all services |
| `docker compose down` | Stop and remove containers |

---

## 10. Streamlit Key Functions

| Function | Purpose |
|---|---|
| `st.set_page_config()` | First command. Sets tab title, icon, layout |
| `st.markdown(unsafe_allow_html=True)` | Renders HTML and CSS — how we inject Netflix theme |
| `st.sidebar` | Left panel content |
| `st.tabs()` | Creates multiple tabs |
| `st.columns()` | Side-by-side layout |
| `st.button()` | Returns True when clicked |
| `st.spinner()` | Loading animation |
| `st.plotly_chart()` | Display Plotly chart |

### Data flow — button to chart

```
User enters User ID → clicks button
→ recommend_btn = True
→ get_recommendations(196, 10) called
→ requests.post sends to FastAPI
→ FastAPI runs Item-CF model
→ JSON response returns
→ Python loops through movies
→ Plotly chart created
→ st.plotly_chart() displays it
```

### requests library — GET vs POST

```python
# GET — retrieve data, no body
requests.get(f"{API_URL}/health", timeout=5)

# POST — send data, get response
requests.post(f"{API_URL}/recommend",
              json={"user_id": 196, "n": 10},
              timeout=30)
```

---

## 11. pytest — Testing

### What is TestClient?
Simulates HTTP requests without a running server. Tests run fast and work in CI/CD.

```python
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post("/recommend", json={"user_id": 196, "n": 5})
assert response.status_code == 200
```

### 6 Tests Written

| Test | What it checks |
|---|---|
| `test_health_endpoint` | GET /health returns 200 with all fields |
| `test_recommend_valid_user` | User 196 gets 5 recommendations |
| `test_recommend_returns_correct_fields` | Each rec has movie_id, title, rating 1.0-5.0 |
| `test_recommend_invalid_user` | User 99999 returns 404 |
| `test_recommend_default_n` | No n parameter returns 10 recommendations |
| `test_recommend_different_users` | Users 196 and 405 get different recommendations |

---

## 12. Pipeline Structure

| File | Purpose |
|---|---|
| `configs/model.yaml` | All parameters. Never hardcode in Python |
| `src/data/preprocessing.py` | Load, clean, save raw data |
| `src/features/feature_engineering.py` | Create derived features |
| `src/models/train.py` | Train all 3 models from config |
| `src/models/evaluate.py` | RMSE, MAE, Precision@K |
| `src/utils/logger.py` | Logging utility |
| `src/utils/database.py` | PostgreSQL connection and storage |
| `api/main.py` | FastAPI entry point |
| `api/dependencies.py` | Load model ONCE on startup |
| `api/schemas/request.py` | Validate incoming data |
| `api/schemas/response.py` | Define response shape |
| `api/routes/recommend.py` | POST /recommend |
| `api/routes/health.py` | GET /health |
| `streamlit_app.py` | CineAI dashboard |
| `tests/test_recommend.py` | 6 pytest tests |
| `Dockerfile` | Container definition |
| `docker-compose.yml` | All services together |
| `.env` | Secrets — never committed |

---

## 13. Key Concepts Summary

| Concept | One Line Explanation |
|---|---|
| Collaborative Filtering | People who agreed in the past will agree in the future |
| Sparsity | 93.7% of ratings are missing — CF predicts the gaps |
| Cosine Similarity | Angle between two rating vectors — direction not distance |
| KNNWithMeans | KNN that subtracts mean rating to correct for user bias |
| Cold Start | New user has no ratings — cannot recommend |
| Long Tail | 31.5% of movies have fewer than 10 ratings |
| Selection Bias | Users only rate movies they chose to watch — positive skew |
| RMSE | Root mean squared prediction error — penalises large errors |
| MAE | Mean absolute prediction error — treats all errors equally |
| Precision@K | Of top K recs what % does the user actually like |

---

## 14. Mistakes Made

| Mistake | Lesson |
|---|---|
| Built notebooks instead of src/ files | EDA only in notebooks. Production code in src/ from Day 1 |
| No configs/model.yaml from start | Config file Day 1. Read all parameters from config |
| Mac Postgres.app conflict | Remove Postgres.app. Docker PostgreSQL only |
| API_URL hardcoded to localhost | Always use os.getenv('API_URL', 'localhost') |
| Models not committed to GitHub | Commit models — Render loads instantly, no retraining |
| No gcc in Dockerfile | scikit-surprise needs gcc/g++ for C compilation |

---

*Week 7 of 15 — Building in public — No shortcuts 🇰🇪*
