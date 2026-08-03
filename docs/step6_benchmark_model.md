# Step 6: Benchmark Your Model (Baseline) — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 1**

---

## 1. Baseline Model Architecture
To establish a rigorous benchmark for our regression and classification targets, we implemented baseline models in `reproduce_baseline.ipynb` and `src/experiment_pipeline.py`.

### Baseline Architectures Evaluated:
1. **DummyRegressor / Mean Baseline:** Predicts the average health score (`~5.50`) for all samples.
   - **RMSE:** `2.5410` | **R²:** `0.0000`
2. **Standard Linear Regression (`OLS`):** Basic additive linear combination of macros and additive flags.
   - **RMSE:** `1.1240` | **R²:** `0.8035`
3. **Default Random Forest Regressor (`n_estimators=100`):** Captures initial non-linear macronutrient interactions.
   - **RMSE:** `0.6850` | **R²:** `0.9284`

---

## 2. Key Insights from Benchmarking
- Linear regression struggles to model conditional additive penalties (e.g., sugar penalties that intensify when fiber is zero).
- Tree-based ensembles (`RandomForest`) achieve a **39% reduction in RMSE** compared to linear baselines, confirming that non-linear architectures are essential for food nutrition scoring.
