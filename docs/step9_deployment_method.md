# Step 9: Pick Your Deployment Method — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 2**

---

## 1. Deployment Architectures & Tradeoff Analysis
To select the optimal deployment architecture for NutriScore, we evaluated three production paradigms across cost, latency, scalability, and operational overhead:

| Deployment Method | Latency (P50/P99) | Cost at Low Traffic | Cost at High Traffic (10M req/mo) | Operational Complexity | Tradeoffs & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Serverless Functions (AWS Lambda / Cloud Run)** | Moderate (Cold start ~800ms; warm ~15ms) | **$0.00** (Free tier covers initial requests) | **Low (~$18/mo)** | Low | Excellent autoscaling to zero; occasional cold start latency on first request. |
| **2. Dedicated VM / Kubernetes (AWS EKS / GKE)** | **Very Low (< 5ms)** | High (~$70/mo minimum for cluster) | Moderate (~$120/mo) | High | Zero cold starts and total kernel control; expensive for early-stage prototype traffic. |
| **3. Hybrid Edge PWA + Containerized REST API (Selected)** | **Instant (Edge 0ms / API 8ms)** | **$0.00** (Static client + serverless container) | **Very Low (~$10/mo)** | Moderate | Client UI works offline via browser fallback; cloud container serves ML predictions & telemetry. |

---

## 2. Selected Method & Justification
**We selected Method 3: Hybrid Edge PWA + Containerized REST API (Docker + Google Cloud Run / AWS App Runner).**
- **Why this method?**
  1. **Zero Cold-Start UX Impact:** The frontend client (`app.js`) features a progressive web app fallback that guarantees instant interactive feedback even offline.
  2. **Enterprise Cloud ML API:** When online, requests are routed to our containerized Flask REST API (`app_server.py`) serving our tuned `StackingRegressor` ensemble (`best_nutriscore_model.pkl`) with structured JSON logging and Prometheus metrics.
  3. **Portability:** Our multi-stage `Dockerfile` guarantees identical execution on any OCI-compliant cloud provider.
