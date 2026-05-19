"""
Week 7 — Recommendation System
Database Utility

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026

Description:
    PostgreSQL connection and table management.
    Stores recommendation requests and results.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://martin:martin123@localhost:5432/recommendations"
)


def get_connection():
    """Get PostgreSQL connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def create_tables():
    """Create tables if they do not exist."""
    conn = get_connection()
    if not conn:
        logger.warning("Skipping table creation — no DB connection")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_requests (
                    id          SERIAL PRIMARY KEY,
                    user_id     INTEGER NOT NULL,
                    n_requested INTEGER NOT NULL,
                    model_used  VARCHAR(50),
                    created_at  TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id               SERIAL PRIMARY KEY,
                    request_id       INTEGER REFERENCES
                                     recommendation_requests(id),
                    user_id          INTEGER NOT NULL,
                    movie_id         INTEGER NOT NULL,
                    title            VARCHAR(255),
                    predicted_rating FLOAT,
                    rank             INTEGER,
                    created_at       TIMESTAMP DEFAULT NOW()
                );
            """)

            conn.commit()
            logger.info("✅ Tables created successfully")

    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        conn.rollback()
    finally:
        conn.close()


def save_recommendations(user_id, recommendations, model_used):
    """Save recommendation request and results to database."""
    conn = get_connection()
    if not conn:
        logger.warning("Skipping DB save — no connection")
        return None

    try:
        with conn.cursor() as cur:
            # Save request
            cur.execute("""
                INSERT INTO recommendation_requests
                    (user_id, n_requested, model_used)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (user_id, len(recommendations), model_used))

            request_id = cur.fetchone()[0]

            # Save each recommendation
            for rank, rec in enumerate(recommendations, 1):
                cur.execute("""
                    INSERT INTO recommendations
                        (request_id, user_id, movie_id,
                         title, predicted_rating, rank)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    request_id,
                    user_id,
                    rec['movie_id'],
                    rec['title'],
                    rec['predicted_rating'],
                    rank
                ))

            conn.commit()
            logger.info(
                f"✅ Saved {len(recommendations)} recommendations "
                f"for user {user_id}")
            return request_id

    except Exception as e:
        logger.error(f"Failed to save recommendations: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def get_user_history(user_id, limit=50):
    """Get recommendation history for a user."""
    conn = get_connection()
    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.title, r.predicted_rating,
                       r.rank, r.created_at
                FROM recommendations r
                JOIN recommendation_requests rq
                    ON r.request_id = rq.id
                WHERE r.user_id = %s
                ORDER BY r.created_at DESC
                LIMIT %s;
            """, (user_id, limit))

            return cur.fetchall()

    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    create_tables()
    print("✅ Database tables ready")