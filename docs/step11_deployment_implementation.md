# Step 11: Deployment Implementation & REST API Documentation — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 2**

---

## 1. REST API Endpoints & Specifications
Our production server (`app_server.py`) exposes REST API endpoints conforming to enterprise JSON specifications:

### `GET /api/v1/health`
- **Purpose:** Liveness and readiness probe for Cloud Run, ECS, or Kubernetes.
- **Sample Response (HTTP 200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "nutriscore-api",
    "model_version": "best_nutriscore_model.pkl",
    "is_ml_model": true,
    "uptime_seconds": 124.52
  }
  ```

### `POST /api/v1/predict`
- **Purpose:** Predicts NutriScore health score (1.0–10.0), letter grade (`A`–`E`), and feature analytics from nutrition facts JSON payload.
- **Request Headers:** `Content-Type: application/json`
- **Sample Request Body:**
  ```json
  {
    "calories": 210,
    "protein_g": 18.0,
    "carbs_g": 15.0,
    "fiber_g": 5.0,
    "fat_g": 4.0,
    "sugar_g": 2.0,
    "sodium_mg": 140,
    "ingredients": "Organic Greek Yogurt, Blueberries, Honey"
  }
  ```
- **Sample Response (HTTP 200 OK):**
  ```json
  {
    "success": true,
    "score": 8.7,
    "grade": "A",
    "model_used": "best_nutriscore_model.pkl",
    "features_analyzed": {
      "calories": 210.0,
      "protein_g": 18.0,
      "carbs_g": 15.0,
      "fiber_g": 5.0,
      "fat_g": 4.0,
      "sugar_g": 2.0,
      "sodium_mg": 140.0,
      "has_high_fructose_corn_syrup": 0,
      "has_healthy_evoo_oil": 0
    },
    "insights": {
      "positive_factors": ["High Protein Density", "Rich in Dietary Fiber"],
      "negative_factors": [],
      "confidence_interval": [8.35, 9.05]
    }
  }
  ```

### `GET /api/v1/barcode/<barcode>`
- **Purpose:** Fetches metadata from Open Food Facts API and returns combined real-time ML score prediction.
- **Sample Request:** `GET /api/v1/barcode/049000006346`

### `GET /api/v1/metrics`
- **Purpose:** Exposes structured latency percentiles (P50/P95/P99), request volume, and letter grade distribution (`A`–`E`) for cloud monitoring and Prometheus scraping.

---

## 2. Containerization & Cloud Deployment Guide

### Build & Run Locally via Docker Compose
To deploy the entire production stack locally:
```bash
docker-compose up --build -d
```
The application server and interactive UI are immediately accessible at `http://localhost:5000`.

### Cloud 1-Command Deployment (Google Cloud Run / AWS App Runner)
```bash
# Build OCI image
docker build -t nutriscore-api:latest .

# Deploy to Google Cloud Run
gcloud run deploy nutriscore-api --image nutriscore-api:latest --platform managed --region us-central1 --allow-unauthenticated
```

---

## 3. Automated Test Suite Verification (`pytest`)
Our automated test suite in `tests/` (`test_api.py`, `test_inference.py`) tests:
- Numerical boundary conditions and grade mapping accuracy (`A`–`E`).
- REST API payload validation and error handling (HTTP 400/404/500).
- Open Food Facts barcode lookup integration.
- Telemetry metrics aggregation.

To run the complete automated test suite:
```bash
pytest -v
```
