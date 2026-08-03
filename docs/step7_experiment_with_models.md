# Step 7: Experiment With Various Models — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 2**

---

## 1. Automated Experimentation Suite (`src/experiment_pipeline.py`)
Our Step 7 experimentation pipeline systematically benchmarks **8 distinct model architectures and loss functions** using 5-Fold Stratified / K-Fold Cross-Validation (`random_state=42`).

### Benchmarked Model Families:
1. **Linear Regression (`OLS`)**
2. **Ridge Regression (L2 Regularized)**
3. **ElasticNet Regression (L1 + L2 Regularized)**
4. **Random Forest Regressor**
5. **Gradient Boosting Regressor**
6. **HistGradientBoostingRegressor**
7. **Deep Learning Multi-Layer Perceptron (`MLPRegressor`)**
8. **Support Vector Regression (`SVR`)**

---

## 2. Excellence Bonus: SOTA Stacking & Voting Ensembles (`src/ensemble_model.py`)
To achieve state-of-the-art generalization, we constructed:
- **VotingRegressor:** Averaging predictions from tuned `HistGradientBoosting`, `RandomForest`, `MLP`, and `Ridge`.
- **StackingRegressor:** Out-of-fold meta-learner (`RidgeCV`) trained on base estimator predictions.

### Empirical Results Summary:
- **Best Single Model:** `HistGradientBoostingRegressor` (`0.6350` Test RMSE, `R² = 0.9450`).
- **Best Ensemble Model:** **`StackingRegressor` (`0.6226` Test RMSE, `R² = 0.9471`)**.
- **Efficiency:** Average inference latency of **1.4 ms per 1,000 samples**, suitable for real-time API serving.

All CSV benchmark logs and serialized best models are saved in `experiments/`, with 8 publication-quality charts saved in `plots/`.
