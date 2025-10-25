# app/main.py
# both Redis and in-memory caching
# Run: uvicorn app.main:app --reload

import hashlib
import json
import os
from functools import lru_cache
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Flexible import for your folder style (keep connection similar to your originals)
try:
    from app.schemas import HeartInput, PredictionOutput
except Exception:
    from schemas import HeartInput, PredictionOutput  # fallback if run differently

import joblib
import numpy as np
from pathlib import Path

# ----------------------------
# Redis
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")  # use "redis" insted of "localhost" while deploying
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
PREDICT_CACHE_TTL = int(os.getenv("PREDICT_CACHE_TTL", "300"))  # seconds

r = None
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        socket_timeout=2,
        socket_connect_timeout=2,
        decode_responses=True,  # store/read JSON as str
    )
    r.ping()
    print("✅ Connected to Redis")
except Exception as e:
    print("⚠️ Redis not available:", e)
    r = None  # app still runs without cache, but Redis preferred

# ----------------------------
# App & Model
# ----------------------------
app = FastAPI(
    title="Heart Disease Classifier API",
    description="Logistic Regression pipeline on heart.csv with Redis + LRU in-memory caching",
    version="1.1.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "app") else Path.cwd()
MODEL_PATH = PROJECT_ROOT / "model" / "heart_model.joblib"

# Consistent class-name mapping (0 -> absence, 1 -> presence)
CLASS_NAMES = {0: "absence", 1: "presence"}

_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. Please run: python model/model_run.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model

# ----------------------------
# Utilities
# ----------------------------
def make_cache_key(payload: Dict[str, Any]) -> str:
    """Create a deterministic cache key from the JSON payload."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "predict:" + hashlib.sha256(body.encode("utf-8")).hexdigest()

def to_feature_row(hi: HeartInput) -> np.ndarray:
    """Order must match schemas.py and training (model_run.py)."""
    return np.array([[
        hi.age,
        hi.trestbps,
        hi.chol,
        hi.thalach,
        hi.oldpeak,
        hi.sex,
        hi.cp,
        hi.fbs,
        hi.restecg,
        hi.exang,
        hi.slope,
        hi.ca,
        hi.thal,
    ]], dtype=float)

# ----------------------------
# In-Memory Cache (@lru_cache)
# ----------------------------
@lru_cache(maxsize=2048)
def memory_cached_prediction(payload_json: str) -> Dict[str, Any]:
    """Cached version of the prediction process (used by /predict)."""
    model = load_model()
    data = json.loads(payload_json)
    hi = HeartInput(**data)
    X = to_feature_row(hi)

    try:
        proba_presence = float(model.predict_proba(X)[0, 1])
    except Exception:
        # Fallback: use sigmoid of decision_function
        decision = float(model.decision_function(X)[0])
        proba_presence = 1.0 / (1.0 + np.exp(-decision))

    pred_int = int(model.predict(X)[0])
    pred_label = CLASS_NAMES.get(pred_int, str(pred_int))

    return {
        "predicted_class": pred_label,
        "probability": round(proba_presence, 6),
    }

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    status = {"status": "ok", "redis": bool(r), "in_memory_cache": "lru_cache(2048)"}
    return JSONResponse(status_code=200, content=status)

@app.get("/info")
def info():
    return {
        "model_path": str(MODEL_PATH),
        "model_type": "LogisticRegression (scikit-learn Pipeline)",
        "target": "heart-disease presence (binary 0/1)",
        "class_names": CLASS_NAMES,
        "cache_order": "LRU → Redis → Model",
        "redis_ttl_seconds": PREDICT_CACHE_TTL if r else 0,
        "memory_cache": "enabled (lru_cache, maxsize=2048)",
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: HeartInput):
    # Payload as dict (for caching)
    payload = input_data.model_dump()
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = make_cache_key(payload)

    # 1️⃣ Try Redis cache
    if r:
        cached = r.get(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except Exception:
                pass  # invalid cache entry

    # 2️⃣ Try in-memory LRU cache
    try:
        result = memory_cached_prediction(payload_json)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    # 3️⃣ Save to Redis for cross-instance caching
    if r:
        try:
            r.setex(key, PREDICT_CACHE_TTL, json.dumps(result))
        except Exception as e:
            print("Redis setex failed:", e)

    return result

