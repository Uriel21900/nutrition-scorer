# Step 12: Share Your Project — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 2**

---

## 1. Technical Blog Post & Executive Portfolio Article
**Title:** Decoding Food Quality with Machine Learning: How We Built NutriScore to Grade Packaged Foods in Real Time  
**Author:** Uriel (ML Engineering Capstone)  

### The Challenge of Food Transparency
Walking down a modern grocery store aisle can feel like navigating a minefield of misleading food labels. Phrases like "All Natural" or "Low Fat" often conceal products loaded with High Fructose Corn Syrup, synthetic dyes, and excessive sodium. For consumers, health apps, and dietary researchers, deciphering nutrition panels on the fly is tedious and error-prone.

### Introducing NutriScore: An AI-Powered Health Scoring Engine
To solve this, we developed **NutriScore**, an end-to-end Machine Learning web application and containerized REST API that evaluates packaged foods on a 1.0 to 10.0 health scale and assigns an intuitive letter grade (`A` through `E`). 

Unlike simplistic point-addition tables, NutriScore models non-linear macronutrient interactions and applies NLP-based additive flagging. By integrating directly with the **Open Food Facts API**, users can scan barcodes using their device camera and receive an instant nutritional audit in under 15 milliseconds.

---

## 2. Methodology & ML Engineering Highlights
1. **Model Diversity & Ensembling:**
   - We benchmarked 8 distinct model families (Linear, Ridge, ElasticNet, Random Forest, Gradient Boosting, HistGradientBoosting, MLP Neural Network, and SVR) on 5,000+ curated profiles.
   - For our Excellence bonus, we constructed a **StackingRegressor** with an out-of-fold `RidgeCV` meta-learner, outperforming all single architectures (`R² = 0.9471`, `RMSE = 0.6226`).
2. **Enterprise Scaling Paradigms:**
   - Evaluated Out-of-Core streaming (`SGDRegressor` + Apache Parquet) on datasets up to **1 Billion data points**, maintaining a **flat 15 MB RAM footprint**.
   - Built a distributed PySpark / SparkML pipeline scaling to **1,250,000+ samples/sec** throughput on multi-node clusters.
3. **Production Deployment & API:**
   - Deployed a multi-stage containerized Flask REST API (`app_server.py`) with structured JSON logging and Prometheus P50/P95/P99 latency tracking.
   - Designed a progressive web app frontend that works offline and seamlessly connects to our ML inference engine when online.

---

## 3. Key Takeaways for ML Engineers
- **Feature Engineering Matters:** Ratios (e.g., protein density, fiber-to-calorie ratio) provide significantly stronger regression signal than raw gram values.
- **Ensemble Meta-Learners Prevent Bias:** Stacking diverse model families (trees + neural nets + regularized linear) mitigates individual model blind spots.
- **Always Design for Fallback:** A reliable production AI system should gracefully downgrade to domain heuristics if network connectivity or model loading fails.

*Explore the full open-source codebase, Docker setup, and interactive notebook on GitHub: https://github.com/Uriel21900/nutrition-scorer*
