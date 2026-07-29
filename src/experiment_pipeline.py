#!/usr/bin/env python3
"""
experiment_pipeline.py
======================
Automated Machine Learning Experimentation Suite for NutriScore.
Evaluates 8 diverse model architectures across multiple families (Linear, Bagged Trees,
Boosting Trees, Deep Learning MLP, SVR), tests various loss functions, applies
reproducible 5-Fold Cross-Validation, and measures training time, inference latency,
and serialized disk size.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    f1_score
)

# Model architectures
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

def score_to_grade(score: float) -> str:
    """Maps continuous NutriScore (1.0 - 10.0) to discrete grade A-E."""
    if score >= 8.0:
        return 'A'
    elif score >= 6.5:
        return 'B'
    elif score >= 5.0:
        return 'C'
    elif score >= 3.5:
        return 'D'
    return 'E'

def run_experiment_pipeline(data_path: str, output_dir: str):
    print("=" * 70)
    print("NUTRISCORE AUTOMATED ML EXPERIMENTATION PIPELINE — STEP 7 CAPSTONE")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns.")
    
    features = [
        'calories', 'protein_g', 'carbs_g', 'fiber_g', 'fat_g',
        'sugar_g', 'sodium_mg', 'sat_fat_g',
        'has_high_fructose_corn_syrup', 'has_hydrogenated_oils',
        'has_artificial_sweeteners', 'has_artificial_colors',
        'has_healthy_evoo_oil'
    ]
    
    X = df[features]
    y_reg = df['health_score']
    y_clf = df['nutriscore_grade']
    
    # Train / Test split (80/20) for held-out evaluation
    X_train, X_test, y_train, y_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )
    
    # 5-Fold Cross Validation setup (Reproducible)
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define 8 diverse candidate models
    models = {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "ElasticNet (L1+L2)": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Random Forest (100 Trees)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting (Exact)": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting (LightGBM approx)": HistGradientBoostingRegressor(random_state=42),
        "MLP Neural Net (64-32)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42),
        "Support Vector Regressor (RBF)": SVR(kernel='rbf', C=1.0)
    }
    
    results = []
    
    print("\n--- BENCHMARKING 8 ML MODEL ARCHITECTURES ---")
    for name, model in models.items():
        # Build pipeline with standard scaler
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Measure 5-Fold Cross-Validation RMSE
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()
        
        # Measure training time on full training set
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0
        
        # Measure inference latency on 1000 samples
        sample_batch = X_test.iloc[:1000] if len(X_test) >= 1000 else X_test
        t0 = time.perf_counter()
        y_pred = pipe.predict(sample_batch)
        inf_time_ms = ((time.perf_counter() - t0) / len(sample_batch)) * 1000.0
        
        # Measure serialized size on disk
        serialized = pickle.dumps(pipe)
        size_kb = len(serialized) / 1024.0
        
        # Evaluate on test set
        y_test_pred = pipe.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_medae = median_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Evaluate Grade Classification Accuracy & F1
        pred_grades = [score_to_grade(s) for s in y_test_pred]
        acc = accuracy_score(y_clf_test, pred_grades)
        f1 = f1_score(y_clf_test, pred_grades, average='macro')
        
        results.append({
            "Model Name": name,
            "CV RMSE (Mean)": round(cv_rmse_mean, 4),
            "CV RMSE (Std)": round(cv_rmse_std, 4),
            "Test RMSE": round(test_rmse, 4),
            "Test MAE": round(test_mae, 4),
            "Test MedAE": round(test_medae, 4),
            "Test R2": round(test_r2, 4),
            "Grade Acc (%)": round(acc * 100, 2),
            "Grade Macro-F1": round(f1, 4),
            "Train Time (ms)": round(train_time_ms, 2),
            "Inference Time / 1k (ms)": round(inf_time_ms * 1000, 2),
            "Model Size (KB)": round(size_kb, 2)
        })
        print(f"[{name}] -> CV RMSE: {cv_rmse_mean:.4f} | Test R2: {test_r2:.4f} | Size: {size_kb:.1f} KB | Train Time: {train_time_ms:.1f} ms")
        
    results_df = pd.DataFrame(results)
    
    # Sort by lowest Test RMSE
    results_df = results_df.sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved model comparison benchmark to: {out_csv}")
    
    # ---------------------------------------------------------
    # LOSS FUNCTION EXPERIMENTATION
    # ---------------------------------------------------------
    print("\n--- EXPERIMENTING WITH DIFFERENT LOSS FUNCTIONS ---")
    loss_results = []
    
    # We test HistGradientBoostingRegressor with different loss functions
    hgb_losses = [
        ('squared_error', 'MSE (Squared Error)'),
        ('absolute_error', 'MAE (Absolute Error)'),
        ('poisson', 'Poisson Deviance')
    ]
    
    for loss_code, loss_label in hgb_losses:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', HistGradientBoostingRegressor(loss=loss_code, random_state=42))
        ])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        tt_ms = (time.perf_counter() - t0) * 1000.0
        preds = pipe.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        loss_results.append({
            "Model": "HistGradientBoosting",
            "Loss Function": loss_label,
            "Test RMSE": round(rmse, 4),
            "Test MAE": round(mae, 4),
            "Test R2": round(r2, 4),
            "Train Time (ms)": round(tt_ms, 2)
        })
        print(f"[HistGradientBoosting | Loss: {loss_label}] -> Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f} | R2: {r2:.4f}")
        
    loss_df = pd.DataFrame(loss_results)
    loss_csv = os.path.join(output_dir, "loss_function_comparison.csv")
    loss_df.to_csv(loss_csv, index=False)
    print(f"Saved loss function comparison to: {loss_csv}")
    
    print("\n" + "="*70)
    print("TOP 3 BENCHMARKED MODELS (BY TEST RMSE)")
    print("="*70)
    print(results_df[["Model Name", "CV RMSE (Mean)", "Test RMSE", "Test R2", "Grade Acc (%)", "Train Time (ms)", "Model Size (KB)"]].head(3).to_string(index=False))
    return results_df, loss_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "nutrition_products_dataset.csv")
    exp_dir = os.path.join(base_dir, "experiments")
    
    run_experiment_pipeline(data_file, exp_dir)
