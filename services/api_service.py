# ===============================
# 🧠 IMPORTS
# ===============================
import os
import sys
import uuid
import time
import logging
import hmac
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse

# ===============================
# 🔧 ENV
# ===============================
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ===============================
# 🔧 IMPORT PREDICTOR
# ===============================
try:
    from services.predictor import create_predictor
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from services.predictor import create_predictor

# ===============================
# 🧠 INIT MODEL (SAFE)
# ===============================
try:
    predictor = create_predictor("model_registry")
except Exception as e:
    raise RuntimeError(f"Failed to initialize predictor: {e}")

# ===============================
# 📝 LOGGING (ENTERPRISE)
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("health_api")

# ===============================
# 🚀 FASTAPI INIT
# ===============================
app = FastAPI(
    title="AI Health Intelligence API",
    description="AI-powered disease risk prediction system"
)


# ===============================
# 🔐 SECURITY
# ===============================
def verify_api_key(x_api_key: str = Header(...)):

    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")

    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# ===============================
# 📥 REQUEST MODEL
# ===============================

class PredictionRequest(BaseModel):
    disease: str = Field(...)
    payload: Dict[str, Any]

    @validator("disease")
    def validate_disease(cls, v):
        allowed = {"breast_cancer", "diabetes", "heart"}
        if v not in allowed:
            raise ValueError(f"Disease must be one of {allowed}")
        return v


# ===============================
# 📤 RESPONSE MODEL (MATCHED)
# ===============================
class PredictionResponse(BaseModel):
    status: str
    trace_id: str

    prediction: Dict[str, Any]
    confidence: Dict[str, Any]

    medical_report: Dict[str, Any]

    insights: Dict[str, Any]

    advanced_analysis: Dict[str, Any]
    decision_support: Dict[str, Any]

    system: Dict[str, Any]

    meta: Dict[str, Any]

    disclaimer: str


# ===============================
# 🏥 ROOT
# ===============================
@app.get("/")
def root():
    return {
        "service": "AI Health Intelligence API",
        "status": "running"
    }


# ===============================
# ❤️ HEALTH
# ===============================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "available_diseases": ["breast_cancer", "diabetes", "heart"],
        "timestamp": datetime.utcnow().isoformat()
    }


# ===============================
# 🚀 PREDICT
# ===============================
@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key)
):
    # Validation
    if not request.payload:
        raise HTTPException(status_code=400, detail="Payload cannot be empty")

    if len(request.payload) < 3:
        raise HTTPException(status_code=400, detail="Insufficient input features")

    trace_id = str(uuid.uuid4())
    start_time = time.time()

    client_ip = http_request.client.host

    try:
        logger.info(
            f"[{trace_id}] Request",
            extra={
                "ip": client_ip,
                "disease": request.disease
            }
        )

        # ===============================
        # 🧠 CALL PREDICTOR
        # ===============================
        try:
            result = predictor.predict(request.disease, request.payload)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Prediction timeout")

        if result.get("status") != "success":
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Prediction failed")
            )

        latency = (time.time() - start_time) * 1000

        logger.info(f"[{trace_id}] Success | Latency={latency:.2f}ms")

        # ===============================
        # ✅ ENRICH RESPONSE (ENTERPRISE)
        # ===============================
        if "meta" not in result:
            result["meta"] = {}

        result["trace_id"] = trace_id
        result["meta"]["latency_ms"] = round(latency, 2)

        return JSONResponse(
            content=result,
            headers={"X-Trace-ID": trace_id}
        )

    except HTTPException as e:
        logger.warning(f"[{trace_id}] Client Error: {str(e.detail)}")
        raise

    except Exception as e:
        logger.error(f"[{trace_id}] Server Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail={
                "trace_id": trace_id,
                "error": "Internal server error"
            }
        )