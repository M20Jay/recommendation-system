"""Recommendation endpoint."""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas.request import RecommendRequest
from api.schemas.response import RecommendResponse, MovieRecommendation
from api.dependencies import model, movies, ratings

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        # Get movies user has already rated
        rated = ratings[
            ratings['user_id'] == request.user_id
        ]['item_id'].tolist()

        if not rated:
            raise HTTPException(
                status_code=404,
                detail=f"User {request.user_id} not found")

        # Get unrated movies
        all_movies = movies['item_id'].tolist()
        unrated = [m for m in all_movies if m not in rated]

        # Predict ratings
        predictions = [
            model.predict(request.user_id, mid)
            for mid in unrated
        ]

        # Sort by predicted rating
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n = predictions[:request.n]

        # Build response
        recs = []
        for pred in top_n:
            title = movies[
                movies['item_id'] == pred.iid
            ]['title'].values[0]
            recs.append(MovieRecommendation(
                movie_id=pred.iid,
                title=title,
                predicted_rating=round(pred.est, 2)
            ))
        
        # Save to database
        try:
            from src.utils.database import save_recommendations
            recs_dict = [r.dict() for r in recs]
            save_recommendations(request.user_id, recs_dict, "Item-CF")
        except Exception as e:
            logger.warning(f"DB save skipped: {e}")

        return RecommendResponse(
            user_id=request.user_id,
            recommendations=recs,
            model="Item-CF",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))