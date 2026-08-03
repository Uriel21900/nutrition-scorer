# Step 3: Project Proposal — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 1**

---

## 1. Project Proposal Summary
This document summarizes the official project proposal submitted and approved for the Machine Learning Engineering Capstone Project (full document available in `Project Proposal.pdf` at the root of the repository).

### Objectives
1. Build an end-to-end Machine Learning pipeline that predicts an accurate, interpretable health score (1.0 to 10.0) and letter grade (`A`–`E`) for packaged foods.
2. Provide a real-time barcode scanning web interface that integrates with the Open Food Facts API.
3. Establish a robust benchmark comparing linear models, tree-based ensembles, neural networks, and stacking meta-estimators.

---

## 2. Technical Architecture & Evaluation Metrics
- **Primary Regression Metric:** **Root Mean Squared Error (RMSE)** — chosen because it quadratically penalizes large errors in health scoring (e.g., predicting an unhealthy snack as a 9.0 is severely penalized).
- **Secondary Regression Metric:** **R² Score** and **Mean Absolute Error (MAE)**.
- **Classification Metric:** **Macro F1-Score** and **Accuracy** across the 5 NutriScore grades (`A`, `B`, `C`, `D`, `E`).
- **Validation Protocol:** 5-Fold Stratified / K-Fold Cross-Validation (`random_state=42`) to guarantee generalization without overfitting.

---

## 3. Deliverables Roadmap
- **Phase 1:** Working Prototype (Dataset generation, exploratory analysis, baseline model benchmarking).
- **Phase 2:** Production Deployment (Automated hyperparameter tuning, StackingRegressor ensemble, Docker containerization, Flask REST API server, automated pytest suite, and cloud scaling).
