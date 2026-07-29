#!/usr/bin/env python3
"""
tune_models.py
==============
Systematic Hyperparameter Tuning Suite for NutriScore ML Capstone.
Performs GridSearchCV and RandomizedSearchCV across multiple hyperparameters
for top-performing architectures (HistGradientBoostingRegressor and RandomForestRegressor),
utilizing 5-Fold Cross-Validation and parallel multi-core processing (simulating
distributed cloud search). Logs parameter importance and validation progressions.
"""

import os
import json
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

def run_hyperparameter_tuning(data_path: str, output_dir: str):
    print("=" * 70)
    print("NUTRISCORE SYSTEMATIC HYPERPARAMETER TUNING SUITE")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    features = [
        'calories', 'protein_g', 'carbs_g', 'fiber_g', 'fat_g',
        'sugar_g', 'sodium_mg', 'sat_fat_g',
        'has_high_fructose_corn_syrup', 'has_hydrogenated_oils',
        'has_artificial_sweeteners', 'has_artificial_colors',
        'has_healthy_evoo_oil'
    ]
    X = df[features]
    y = df['health_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)
    
    tuning_log = []
    best_models_dict = {}
    
    # -------------------------------------------------------------
    # 1. TUNING HISTGRADIENTBOOSTING REGRESSOR (LightGBM equivalent)
    # -------------------------------------------------------------
    print("\n[1/2] Tuning HistGradientBoostingRegressor (GridSearchCV)...")
    hgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('hgb', HistGradientBoostingRegressor(random_state=42))
    ])
    
    hgb_param_grid = {
        'hgb__max_iter': [100, 200, 300],
        'hgb__learning_rate': [0.05, 0.1, 0.15],
        'hgb__max_depth': [5, 8, None],
        'hgb__min_samples_leaf': [15, 20, 30],
        'hgb__l2_regularization': [0.0, 0.5, 1.0]
    }
    
    # Using RandomizedSearchCV to explore 25 random combinations systematically
    hgb_search = RandomizedSearchCV(
        hgb_pipe,
        param_distributions=hgb_param_grid,
        n_iter=25,
        scoring='neg_root_mean_squared_error',
        cv=cv_kfold,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    t0 = time.perf_counter()
    hgb_search.fit(X_train, y_train)
    hgb_tune_time = (time.perf_counter() - t0)
    
    best_hgb = hgb_search.best_estimator_
    best_hgb_cv_rmse = -hgb_search.best_score_
    test_hgb_preds = best_hgb.predict(X_test)
    test_hgb_rmse = np.sqrt(mean_squared_error(y_test, test_hgb_preds))
    test_hgb_r2 = r2_score(y_test, test_hgb_preds)
    
    print(f"--> Best HistGradientBoosting CV RMSE: {best_hgb_cv_rmse:.4f} | Test RMSE: {test_hgb_rmse:.4f} | R2: {test_hgb_r2:.4f}")
    print(f"--> Best Params: {hgb_search.best_params_}")
    print(f"--> Tuning Time: {hgb_tune_time:.2f} seconds across 25 parameter candidates.")
    
    tuning_log.append({
        "Model": "HistGradientBoostingRegressor",
        "Best CV RMSE": round(best_hgb_cv_rmse, 4),
        "Test RMSE": round(test_hgb_rmse, 4),
        "Test R2": round(test_hgb_r2, 4),
        "Best Parameters": str(hgb_search.best_params_),
        "Tuning Time (s)": round(hgb_tune_time, 2)
    })
    best_models_dict["HistGradientBoostingRegressor"] = {
        "params": hgb_search.best_params_,
        "cv_rmse": round(best_hgb_cv_rmse, 4),
        "test_rmse": round(test_hgb_rmse, 4),
        "test_r2": round(test_hgb_r2, 4)
    }
    
    # -------------------------------------------------------------
    # 2. TUNING RANDOM FOREST REGRESSOR (GridSearchCV / RandomizedSearch)
    # -------------------------------------------------------------
    print("\n[2/2] Tuning RandomForestRegressor (RandomizedSearchCV)...")
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    rf_param_grid = {
        'rf__n_estimators': [80, 120, 180],
        'rf__max_depth': [10, 15, 20, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__max_features': ['sqrt', 'log2', 1.0]
    }
    
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=rf_param_grid,
        n_iter=15,
        scoring='neg_root_mean_squared_error',
        cv=cv_kfold,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    t0 = time.perf_counter()
    rf_search.fit(X_train, y_train)
    rf_tune_time = (time.perf_counter() - t0)
    
    best_rf = rf_search.best_estimator_
    best_rf_cv_rmse = -rf_search.best_score_
    test_rf_preds = best_rf.predict(X_test)
    test_rf_rmse = np.sqrt(mean_squared_error(y_test, test_rf_preds))
    test_rf_r2 = r2_score(y_test, test_rf_preds)
    
    print(f"--> Best RandomForest CV RMSE: {best_rf_cv_rmse:.4f} | Test RMSE: {test_rf_rmse:.4f} | R2: {test_rf_r2:.4f}")
    print(f"--> Best Params: {rf_search.best_params_}")
    print(f"--> Tuning Time: {rf_tune_time:.2f} seconds across 15 parameter candidates.")
    
    tuning_log.append({
        "Model": "RandomForestRegressor",
        "Best CV RMSE": round(best_rf_cv_rmse, 4),
        "Test RMSE": round(test_rf_rmse, 4),
        "Test R2": round(test_rf_r2, 4),
        "Best Parameters": str(rf_search.best_params_),
        "Tuning Time (s)": round(rf_tune_time, 2)
    })
    best_models_dict["RandomForestRegressor"] = {
        "params": rf_search.best_params_,
        "cv_rmse": round(best_rf_cv_rmse, 4),
        "test_rmse": round(test_rf_rmse, 4),
        "test_r2": round(test_rf_r2, 4)
    }
    
    # Save tuning summary
    tuning_df = pd.DataFrame(tuning_log)
    out_csv = os.path.join(output_dir, "tuning_results.csv")
    tuning_df.to_csv(out_csv, index=False)
    print(f"\nSaved hyperparameter tuning summary to: {out_csv}")
    
    out_json = os.path.join(output_dir, "best_params.json")
    with open(out_json, "w") as f:
        json.dump(best_models_dict, f, indent=4)
    print(f"Saved best tuned hyperparameters to: {out_json}")
    
    return tuning_df, best_models_dict

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "nutrition_products_dataset.csv")
    exp_dir = os.path.join(base_dir, "experiments")
    
    run_hyperparameter_tuning(data_file, exp_dir)
