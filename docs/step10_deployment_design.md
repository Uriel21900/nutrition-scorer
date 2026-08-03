# Step 10: Design Your Deployment Solution & Engineering Plan — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 2**

---

## 1. End-to-End Production Architecture & Lifecycle Diagram (Meets Expectations & Excellence)

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
    
    subgraph MLOps Lifecycle: Drift Monitoring & Automated Retraining
        CloudMonitor -->|7. Detect Covariate Shift or Grade Skew| DriftDetector[Statistical Drift & Anomaly Monitor]
        DriftDetector -->|8. Webhook: Trigger CI/CD Retraining| RetrainAction[GitHub Actions / Cloud Build CI/CD]
        RetrainAction -->|9. Ingest Fresh Product Telemetry| GenerateDS[data/generate_dataset.py]
        GenerateDS -->|10. 5-Fold Stratified CV & Ensembling| TrainPipeline[src/experiment_pipeline.py & ensemble_model.py]
        TrainPipeline -->|11. Execute Automated Regression Suite| TestSuite[pytest -v tests/]
        TestSuite -->|12. 100% Tests Pass: Export New Model| StackingModel
    end
```

---

## 2. Monitoring, Logging, and Debugging Plan (2 Points: Completion)
1. **Structured JSON Telemetry (`src/monitoring/logger.py`):**
   - Every API request is logged as structured JSON containing `timestamp`, `path`, `status_code`, `latency_ms`, `model_version`, `prediction`, and `grade`.
   - Allows instant log querying in Google Cloud Logging or AWS CloudWatch using structured JSON queries.
2. **Real-Time Prometheus & Health Probes:**
   - **`/api/v1/health`:** Liveness probe checked every 30 seconds by container orchestration (Docker/Kubernetes) to verify model uptime and readiness.
   - **`/api/v1/metrics`:** Exposes P50/P95/P99 latency percentiles, total request counts, error rates, and grade distribution (`A`–`E`) to detect data drift or prediction skew.
3. **Debugging Strategy:**
   - If `best_nutriscore_model.pkl` is missing or fails version checks, `NutriScoreInferenceEngine` automatically logs a warning and falls back to our deterministic analytical heuristic scoring engine without crashing client sessions.

---

## 3. Post-Deployment Model Care & Retraining Strategy (Rubric Criterion: Care & Redeploy)
To ensure the NutriScore model remains accurate over time as food manufacturers reformulate products:
1. **Automated Retraining Loop:** When our `DriftDetector` observes > 15% divergence in predicted grade distribution or feature medians, an automated webhook triggers retraining.
2. **Quality Gating via `pytest`:** Before any retrained model artifact (`best_nutriscore_model.pkl`) is deployed to production, the CI/CD pipeline executes our full automated test suite (`tests/test_inference.py` and `tests/test_api.py`).
3. **Zero-Downtime Blue-Green Cutover:** New container images load the retrained artifact and deploy via Kubernetes rolling updates, ensuring zero API downtime.
