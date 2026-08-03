# Step 10: Design Your Deployment Solution — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 2**

---

## 1. End-to-End Production Architecture Diagram

```mermaid
graph TD
    Client[Consumer Mobile / Web Browser PWA] -->|1. Barcode Scan / Manual Macros| APIGateway[Cloud Load Balancer / HTTPS]
    APIGateway -->|2. POST /api/v1/predict or GET /api/v1/barcode| FlaskApp[Flask / Gunicorn Container App Server]
    
    subgraph Containerized Backend Service
        FlaskApp -->|3. Feature Vector & Sanitization| MLEngine[NutriScoreInferenceEngine]
        MLEngine -->|4a. Lookup Metadata| OpenFoodFacts[Open Food Facts API]
        MLEngine -->|4b. Predict Score & Grade A-E| StackingModel[best_nutriscore_model.pkl / StackingRegressor]
    end
    
    FlaskApp -->|5. Structured JSON Telemetry| CloudLogging[Cloud Logging / AWS CloudWatch]
    FlaskApp -->|6. Prometheus / Cloud Metrics| CloudMonitor[Monitoring Dashboard / /api/v1/metrics]
    
    subgraph Data Pipeline & Training
        RawData[Open Food Facts Dump + Synthetic] -->|7. Offline Batch ETL| GenerateDS[generate_dataset.py]
        GenerateDS -->|8. 5-Fold Stratified CV| TrainPipeline[experiment_pipeline.py & ensemble_model.py]
        TrainPipeline -->|9. Export Serialized Artifact| StackingModel
    end
```

---

## 2. Monitoring, Logging, and Debugging Plan
1. **Structured JSON Telemetry (`src/monitoring/logger.py`):**
   - Every API request is logged as structured JSON containing `timestamp`, `path`, `status_code`, `latency_ms`, `model_version`, `prediction`, and `grade`.
   - Allows instant log querying in Google Cloud Logging or AWS CloudWatch.
2. **Real-Time Prometheus & Health Probes:**
   - **`/api/v1/health`:** Liveness probe checked every 30 seconds by container orchestration to verify model uptime and readiness.
   - **`/api/v1/metrics`:** Exposes P50/P95/P99 latency percentiles, total request counts, error rates, and grade distribution (`A`–`E`) to detect data drift or prediction skew.
3. **Debugging Strategy:**
   - If `best_nutriscore_model.pkl` is missing or fails version checks, `NutriScoreInferenceEngine` automatically logs a warning and falls back to our deterministic analytical heuristic scoring engine without crashing client sessions.
