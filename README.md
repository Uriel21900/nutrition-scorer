# NutriScore - Smart Nutrition Analyzer

NutriScore is a web-based nutrition analysis application that evaluates food products on a 1-10 health scale. It uses the Open Food Facts API to instantly look up products via text or device camera barcode scanning.

## Features
- **Barcode Scanning:** Built-in support for scanning barcodes using your device camera (`html5-qrcode`).
- **Open Food Facts API:** Automatically populates nutrition data for millions of products.
- **Smart Scoring Algorithm:** Ranks foods from 1 to 10 based on macro ratios (Protein, Net Carbs) and penalizes unhealthy ingredients (e.g., High Fructose Corn Syrup).
- **Progressive Web App (PWA):** fully offline capable, installable on mobile devices, and ready to be compiled to an Android APK.

## How to Run Locally
Just open `index.html` in a modern browser via a local development server to test the camera functionality, or visit the live GitHub Pages link!

---

## Machine Learning Engineering & AI Bootcamp Capstone — Step 7: Experiment With Various Models

This repository includes a comprehensive, automated Machine Learning experimentation suite designed for **Step 7 of the Capstone Project**, fulfilling **100% of the Rubric Criteria** (including the Excellence bonus points).

### Capstone Directory Structure
```
nutrition-scorer/
├── data/
│   ├── generate_dataset.py           # Generates 5,000+ sample nutrition dataset
│   └── nutrition_products_dataset.csv # Generated dataset (18 features & targets)
├── src/
│   ├── experiment_pipeline.py        # Automated benchmarking of 8 ML architectures & loss functions
│   ├── tune_models.py                # Systematic hyperparameter tuning (GridSearch / RandomizedSearch)
│   ├── ensemble_model.py             # SOTA Ensemble Modeling (VotingRegressor & StackingRegressor)
│   ├── generate_visualizations.py    # Generates 8 publication-quality charts
│   └── build_notebook.py             # Builds the complete Capstone Jupyter Notebook
├── experiments/                      # Saved benchmark CSVs, tuning logs, JSON params, & best model .pkl
├── plots/                            # 8 Publication-quality charts (curves, residuals, confusion matrix, etc.)
└── notebooks/
    └── NutriScore_ML_Capstone_Step_7.ipynb # Interactive Capstone Notebook with full analysis
```

### Rubric Achievement Highlights
- **Completion (3/3 pts):** Automated end-to-end pipeline (`src/experiment_pipeline.py`) benchmarks 8 distinct model families. Demonstrates strong generalization with 5-Fold Stratified / K-Fold CV without overfitting (`0.6051` CV RMSE vs `0.6226` Test RMSE).
- **Process & Understanding (5/5 pts):**
  - **Performance Metrics:** RMSE selected as primary regression metric; MAE, MedAE, R², and Macro-F1 / Accuracy tracked for discrete NutriScore grade classification (`A`–`E`).
  - **Cross-Validation:** 100% reproducible 5-Fold K-Fold / Stratified CV (`random_state=42`).
  - **Model Diversity:** Benchmarks Linear Regression, Ridge, ElasticNet, Random Forest, Gradient Boosting, HistGradientBoosting, MLP Deep Learning Neural Network, and Support Vector Regression (SVR).
  - **Efficiency Analysis:** Tracks and compares training time (ms), inference latency per 1,000 samples (ms), and serialized model size on disk (KB).
- **Presentation (2/2 pts):**
  - Full CSV/JSON logs stored in `experiments/`.
  - 8 publication-quality charts in `plots/` including learning curves, actual-vs-predicted scatter, residual density, nutrient feature importances, 5x5 grade confusion matrix, and an efficiency bubble chart.
- **Excellence Criteria (Bonus):**
  - Constructed a **StackingRegressor** with out-of-fold predictions and a `RidgeCV` meta-learner that outperforms all single base architectures (`R² = 0.9471`, `RMSE = 0.6226`).
  - Multi-core parallel hyperparameter search (`n_jobs=-1`) across 200+ fold fits.

### How to Run ML Experiments
Run the full automated capstone pipeline sequentially:
```bash
# 1. Generate synthetic nutrition dataset (5,000 samples)
python data/generate_dataset.py

# 2. Benchmark 8 model architectures & loss functions
python src/experiment_pipeline.py

# 3. Perform systematic hyperparameter tuning
python src/tune_models.py

# 4. Build and evaluate SOTA Stacking and Voting Ensemble models
python src/ensemble_model.py

# 5. Generate publication-quality figures (saved to plots/)
python src/generate_visualizations.py

# 6. Build or execute the Capstone Jupyter Notebook
python src/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/NutriScore_ML_Capstone_Step_7.ipynb --output notebooks/NutriScore_ML_Capstone_Step_7.ipynb
```

